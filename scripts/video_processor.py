#!/usr/bin/env python3
"""
KUBE-AI Video Processing Module
Process drone video streams for animal detection — MindSpore
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from kube_ai_inference import KubeAIModel, CLASS_LIST

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")


def process_video(model, video_path, output_path=None, frame_skip=5):
    """Detect animals in a drone video file."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} — {total} frames @ {fps} FPS")

    detections = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((224, 224))
            arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
            tensor = Tensor(arr[np.newaxis, :], ms.float32)

            cls_logits, bbox_preds = model(tensor)
            probs = nn.Softmax(axis=1)(cls_logits).asnumpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

            if conf > 0.7:
                det = {
                    'frame': frame_num,
                    'timestamp_s': round(frame_num / fps, 2),
                    'animal': CLASS_LIST[pred],
                    'confidence': round(conf, 4),
                }
                detections.append(det)
                print(f"  Frame {frame_num}: {det['animal']} ({conf:.2f})")

        frame_num += 1

    cap.release()

    if output_path:
        with open(output_path, 'w') as f:
            json.dump({'total_detections': len(detections), 'detections': detections}, f, indent=2)
        print(f"Results saved: {output_path}")

    return detections


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KUBE-AI Video Processor')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--output_path', default='results/video_results.json')
    parser.add_argument('--frame_skip', type=int, default=5)
    args = parser.parse_args()

    model = KubeAIModel(num_classes=10)
    ms.load_param_into_net(model, ms.load_checkpoint(args.model_path))
    model.set_train(False)

    process_video(model, args.video_path, args.output_path, args.frame_skip)
