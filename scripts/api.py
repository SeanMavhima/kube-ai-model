from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import base64
import time

app = Flask(__name__)

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256 * 28 * 28, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load model
model = Model()
model.load_state_dict(torch.load('../models/kube_pytorch.pth'))
model.eval()

classes = ['cattle', 'goat', 'sheep', 'elephant', 'zebra', 'giraffe', 'buffalo', 'antelope', 'lion', 'leopard']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        
        # Preprocess
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Predict
        start_time = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, 1)[0][predicted].item()
        
        processing_time = (time.time() - start_time) * 1000
        
        animal = classes[predicted.item()]
        
        # Determine alert level
        alert_level = "CRITICAL" if animal in ['lion', 'leopard'] else "STANDARD"
        module = "KUBE-Park" if animal in ['elephant', 'zebra', 'giraffe', 'buffalo', 'antelope', 'lion', 'leopard'] else "KUBE-Farm"
        
        return jsonify({
            'detection_id': f'kube_{int(time.time())}',
            'animal_type': animal,
            'confidence': round(confidence, 4),
            'kube_module': module,
            'alert_level': alert_level,
            'processing_time_ms': round(processing_time, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'KUBE-AI API is running', 'version': '1.0'})

if __name__ == '__main__':
    print("🚁 KUBE-AI API Starting...")
    print("📡 Endpoint: http://localhost:5000/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)