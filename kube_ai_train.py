#!/usr/bin/env python3
"""
KUBE-AI Training Script
Aerial Intelligence for Livestock & Wildlife Detection
Framework: MindSpore + Huawei ModelArts
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, Callback
import logging
import time
import json

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

CLASS_LIST = [
    'cattle', 'goat', 'sheep', 'elephant', 'zebra',
    'giraffe', 'buffalo', 'antelope', 'lion', 'leopard'
]


class AerialAnimalDataset:
    """VOC-format aerial imagery dataset."""

    def __init__(self, root_dir, class_list=None):
        self.class_list = class_list or CLASS_LIST
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}
        self.images, self.annotations = [], []

        img_dir = os.path.join(root_dir, 'JPEGImages')
        ann_dir = os.path.join(root_dir, 'Annotations')

        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    xml = os.path.join(ann_dir, os.path.splitext(f)[0] + '.xml')
                    if os.path.exists(xml):
                        self.images.append(os.path.join(img_dir, f))
                        self.annotations.append(xml)

        logger.info(f"Found {len(self.images)} training images")

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        w, h = img.size
        img = img.resize((224, 224))
        arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))

        boxes, labels = self._parse_xml(self.annotations[idx], w, h)
        label = labels[0] if labels else 0
        bbox = boxes[0] if boxes else [0.0, 0.0, 1.0, 1.0]

        return arr, np.array(label, dtype=np.int32), np.array(bbox, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def _parse_xml(self, xml_path, img_w, img_h):
        tree = ET.parse(xml_path)
        boxes, labels = [], []
        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text.lower()
            if name not in self.class_to_idx:
                continue
            bb = obj.find('bndbox')
            boxes.append([
                float(bb.find('xmin').text) / img_w,
                float(bb.find('ymin').text) / img_h,
                float(bb.find('xmax').text) / img_w,
                float(bb.find('ymax').text) / img_h,
            ])
            labels.append(self.class_to_idx[name])
        return boxes, labels


class KubeAIModel(nn.Cell):
    """Dual-head CNN: classification + bounding-box regression."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.SequentialCell([
            nn.Conv2d(3, 64, 3, padding=1, pad_mode='pad'), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, pad_mode='pad'), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1, pad_mode='pad'), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1, pad_mode='pad'), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3, padding=1, pad_mode='pad'), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        ])
        feat = 1024 * 7 * 7
        self.classifier = nn.SequentialCell([
            nn.Dense(feat, 2048), nn.ReLU(), nn.Dropout(keep_prob=0.5),
            nn.Dense(2048, 1024), nn.ReLU(), nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, num_classes),
        ])
        self.bbox_regressor = nn.SequentialCell([
            nn.Dense(feat, 2048), nn.ReLU(), nn.Dropout(keep_prob=0.5),
            nn.Dense(2048, 1024), nn.ReLU(), nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, 4), nn.Sigmoid(),
        ])
        self.flatten = nn.Flatten()

    def construct(self, x):
        f = self.flatten(self.backbone(x))
        return self.classifier(f), self.bbox_regressor(f)


class DetectionLoss(nn.Cell):
    """Classification + bounding-box regression loss."""

    def __init__(self):
        super().__init__()
        self.cls_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.bbox_loss = nn.MSELoss(reduction='mean')

    def construct(self, cls_pred, bbox_pred, cls_target, bbox_target):
        return self.cls_loss(cls_pred, cls_target) + 2.0 * self.bbox_loss(bbox_pred, bbox_target)


class TrainLogger(Callback):
    """Log epoch-level metrics to file."""

    def __init__(self):
        super().__init__()
        self.epoch_start = 0

    def on_train_epoch_begin(self, run_context):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, run_context):
        cb = run_context.original_args()
        elapsed = time.time() - self.epoch_start
        logger.info(f"Epoch {cb.cur_epoch_num}/{cb.batch_num} | Loss: {cb.net_outputs.asnumpy():.4f} | Time: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='KUBE-AI Training')
    parser.add_argument('--data_url', type=str, default='./data')
    parser.add_argument('--train_url', type=str, default='./models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.train_url, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logger.info("=" * 60)
    logger.info("KUBE-AI: Aerial Intelligence Training")
    logger.info("=" * 60)

    ds_gen = AerialAnimalDataset(args.data_url)
    dataset = GeneratorDataset(ds_gen, ["image", "label", "bbox"], shuffle=True)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)

    model_net = KubeAIModel(num_classes=args.num_classes)
    loss_fn = DetectionLoss()
    optimizer = nn.Adam(model_net.trainable_params(), learning_rate=args.lr)

    from mindspore.train import Model
    model = Model(model_net, loss_fn, optimizer)

    ck_cfg = CheckpointConfig(save_checkpoint_steps=50, keep_checkpoint_max=5)
    callbacks = [
        ModelCheckpoint(prefix="kube_ai", directory=args.train_url, config=ck_cfg),
        LossMonitor(),
    ]

    logger.info("Starting training...")
    t0 = time.time()
    model.train(args.epochs, dataset, callbacks=callbacks)
    logger.info(f"Training completed in {(time.time() - t0) / 60:.2f} minutes")

    ckpt_path = os.path.join(args.train_url, "kube_ai_final.ckpt")
    ms.save_checkpoint(model_net, ckpt_path)
    logger.info(f"Final model saved: {ckpt_path}")


if __name__ == '__main__':
    main()
