#!/usr/bin/env python3
"""
KUBE-AI Inference Script
Real-time Aerial Animal Detection — MindSpore
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import json
import time

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

CLASS_LIST = [
    'cattle', 'goat', 'sheep', 'elephant', 'zebra',
    'giraffe', 'buffalo', 'antelope', 'lion', 'leopard'
]

LIVESTOCK = {'cattle', 'goat', 'sheep'}
PREDATORS = {'lion', 'leopard'}
WILDLIFE = {'elephant', 'zebra', 'giraffe', 'buffalo', 'antelope'}


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


def get_module(animal):
    if animal in LIVESTOCK:
        return 'KUBE-Farm'
    if animal in PREDATORS:
        return 'KUBE-Park (Critical)'
    if animal in WILDLIFE:
        return 'KUBE-Park'
    return 'KUBE-Land'


def get_alert(confidence, animal):
    if animal in PREDATORS and confidence > 0.7:
        return 'CRITICAL - Predator Alert'
    if confidence > 0.8:
        return 'HIGH - Confirmed Detection'
    if confidence > 0.6:
        return 'MEDIUM - Probable Detection'
    return 'LOW - Possible Detection'


def detect(model, image_path):
    """Run detection on a single image."""
    t0 = time.time()
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    arr = np.transpose(np.array(img.resize((224, 224)), dtype=np.float32) / 255.0, (2, 0, 1))
    tensor = Tensor(arr[np.newaxis, :], ms.float32)

    cls_logits, bbox_preds = model(tensor)
    probs = nn.Softmax(axis=1)(cls_logits).asnumpy()[0]
    pred_cls = int(np.argmax(probs))
    conf = float(probs[pred_cls])
    bbox = bbox_preds.asnumpy()[0]

    animal = CLASS_LIST[pred_cls] if pred_cls < len(CLASS_LIST) else f'unknown_{pred_cls}'
    px_bbox = [int(bbox[0] * orig_w), int(bbox[1] * orig_h), int(bbox[2] * orig_w), int(bbox[3] * orig_h)]

    return {
        'detection_id': f'kube_{int(time.time())}',
        'animal_type': animal,
        'confidence': round(conf, 4),
        'bbox': px_bbox,
        'image_size': [orig_w, orig_h],
        'inference_time_ms': round((time.time() - t0) * 1000, 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'kube_module': get_module(animal),
        'alert_level': get_alert(conf, animal),
    }, img


def visualize(img, result, output_path):
    draw = ImageDraw.Draw(img)
    bbox = result['bbox']
    colors = {'KUBE-Farm': 'green', 'KUBE-Park (Critical)': 'red', 'KUBE-Park': 'blue', 'KUBE-Land': 'orange'}
    color = colors.get(result['kube_module'], 'yellow')

    draw.rectangle(bbox, outline=color, width=4)
    label = f"{result['animal_type']}: {result['confidence']:.2f}"
    draw.text((bbox[0], max(bbox[1] - 20, 0)), label, fill=color)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='KUBE-AI Inference')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--output_path', default='results/detection_result.jpg')
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    model = KubeAIModel(num_classes=args.num_classes)
    ms.load_param_into_net(model, ms.load_checkpoint(args.model_path))
    model.set_train(False)

    result, img = detect(model, args.image_path)

    print("\n" + "=" * 50)
    print("KUBE-AI DETECTION RESULTS")
    print("=" * 50)
    print(f"Animal:     {result['animal_type']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Location:   {result['bbox']}")
    print(f"Module:     {result['kube_module']}")
    print(f"Alert:      {result['alert_level']}")
    print(f"Time:       {result['inference_time_ms']:.1f} ms")
    print("=" * 50)

    visualize(img, result, args.output_path)
    print(f"Visualization saved: {args.output_path}")

    json_path = args.output_path.rsplit('.', 1)[0] + '_result.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"JSON saved: {json_path}")


if __name__ == '__main__':
    main()
