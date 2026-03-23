#!/usr/bin/env python3
"""
KUBE-AI REST API Server
Serves aerial animal detection via HTTP — MindSpore
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from kube_ai_inference import KubeAIModel, CLASS_LIST, get_module, get_alert

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

app = Flask(__name__)

MODEL = None


def load_model(path, num_classes=10):
    global MODEL
    net = KubeAIModel(num_classes=num_classes)
    ms.load_param_into_net(net, ms.load_checkpoint(path))
    net.set_train(False)
    MODEL = net
    print(f"Model loaded from {path}")


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        t0 = time.time()
        img = Image.open(request.files['image'].stream).convert('RGB')
        arr = np.transpose(np.array(img.resize((224, 224)), dtype=np.float32) / 255.0, (2, 0, 1))
        tensor = Tensor(arr[np.newaxis, :], ms.float32)

        cls_logits, bbox_preds = MODEL(tensor)
        probs = nn.Softmax(axis=1)(cls_logits).asnumpy()[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        animal = CLASS_LIST[pred] if pred < len(CLASS_LIST) else f'unknown_{pred}'

        return jsonify({
            'detection_id': f'kube_{int(time.time())}',
            'animal_type': animal,
            'confidence': round(conf, 4),
            'kube_module': get_module(animal),
            'alert_level': get_alert(conf, animal),
            'inference_time_ms': round((time.time() - t0) * 1000, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model_loaded': MODEL is not None})


if __name__ == '__main__':
    model_path = os.environ.get('KUBE_MODEL_PATH', '../models/kube_ai_final.ckpt')
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        print(f"Warning: {model_path} not found — API will run in no-model mode")

    app.run(host='0.0.0.0', port=5000)
