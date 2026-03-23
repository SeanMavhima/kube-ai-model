#!/usr/bin/env python3
"""Quick smoke-test for the KUBE-AI REST API."""

import requests
import json

BASE = 'http://localhost:5000'


def test_health():
    r = requests.get(f'{BASE}/health')
    print("Health:", r.json())


def test_predict(image_path='../data/JPEGImages/img_000001.jpg'):
    with open(image_path, 'rb') as f:
        r = requests.post(f'{BASE}/predict', files={'image': f})
    if r.ok:
        print("Prediction:", json.dumps(r.json(), indent=2))
    else:
        print(f"Error {r.status_code}: {r.text}")


if __name__ == '__main__':
    test_health()
    test_predict()
