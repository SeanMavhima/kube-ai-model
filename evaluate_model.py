#!/usr/bin/env python3
"""
KUBE-AI Model Evaluation
Runs inference on the test set and reports per-class metrics.
"""

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import json
import time
import logging

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/evaluation.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

CLASS_LIST = [
    'cattle', 'goat', 'sheep', 'elephant', 'zebra',
    'giraffe', 'buffalo', 'antelope', 'lion', 'leopard'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LIST)}


class KubeAIModel(nn.Cell):
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


def get_true_label(xml_path):
    """Extract first object label from VOC XML."""
    tree = ET.parse(xml_path)
    for obj in tree.getroot().findall('object'):
        name = obj.find('name').text.lower()
        if name in CLASS_TO_IDX:
            return CLASS_TO_IDX[name], name
    return None, None


def predict_image(model, img_path):
    """Run inference on a single image, return predicted class index and confidence."""
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
    tensor = Tensor(arr[np.newaxis, :], ms.float32)
    cls_logits, _ = model(tensor)
    probs = nn.Softmax(axis=1)(cls_logits).asnumpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


def evaluate(model, test_img_dir, test_ann_dir):
    """Evaluate model on the test set."""
    true_labels, pred_labels = [], []
    total_time = 0

    img_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    logger.info(f"Evaluating on {len(img_files)} test images...")

    for i, img_file in enumerate(img_files):
        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(test_ann_dir, xml_file)
        if not os.path.exists(xml_path):
            continue

        true_idx, true_name = get_true_label(xml_path)
        if true_idx is None:
            continue

        t0 = time.time()
        pred_idx, conf = predict_image(model, os.path.join(test_img_dir, img_file))
        total_time += time.time() - t0

        true_labels.append(true_idx)
        pred_labels.append(pred_idx)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(img_files)} images...")

    return np.array(true_labels), np.array(pred_labels), total_time


def compute_metrics(true, pred):
    """Compute per-class precision, recall, F1 and overall accuracy."""
    overall_acc = float(np.mean(true == pred))
    per_class = {}

    present_classes = sorted(set(true.tolist()) | set(pred.tolist()))
    for idx in present_classes:
        tp = int(np.sum((true == idx) & (pred == idx)))
        fp = int(np.sum((true != idx) & (pred == idx)))
        fn = int(np.sum((true == idx) & (pred != idx)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        name = CLASS_LIST[idx] if idx < len(CLASS_LIST) else f'class_{idx}'
        per_class[name] = {'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
                           'support': int(np.sum(true == idx))}

    return {'overall_accuracy': round(overall_acc, 4), 'per_class': per_class, 'total_samples': len(true)}


def main():
    parser = argparse.ArgumentParser(description='KUBE-AI Evaluation')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_img_dir', default='./data/TestImages')
    parser.add_argument('--test_ann_dir', default='./data/TestAnnotations')
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    model = KubeAIModel(num_classes=args.num_classes)
    ms.load_param_into_net(model, ms.load_checkpoint(args.model_path))
    model.set_train(False)
    logger.info("Model loaded.")

    true, pred, elapsed = evaluate(model, args.test_img_dir, args.test_ann_dir)

    if len(true) == 0:
        logger.error("No test samples found. Check test data paths.")
        return

    metrics = compute_metrics(true, pred)
    avg_ms = (elapsed / len(true)) * 1000

    logger.info("=" * 50)
    logger.info("KUBE-AI EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total samples:    {metrics['total_samples']}")
    logger.info(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info(f"Avg inference:    {avg_ms:.1f} ms/image")
    logger.info("-" * 50)
    for name, m in metrics['per_class'].items():
        logger.info(f"  {name:12s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  n={m['support']}")
    logger.info("=" * 50)

    os.makedirs('results', exist_ok=True)
    out_path = 'results/evaluation_results.json'
    metrics['avg_inference_ms'] = round(avg_ms, 1)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Results saved: {out_path}")


if __name__ == '__main__':
    main()
