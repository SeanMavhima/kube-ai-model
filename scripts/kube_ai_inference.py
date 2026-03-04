#!/usr/bin/env python3
"""
KUBE-AI Inference Script
Real-time Aerial Animal Detection
Framework: MindSpore
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import json
import time

# Set MindSpore context
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class KubeAIModel(nn.Cell):
    """KUBE-AI: Enhanced CNN for Aerial Animal Detection"""
    
    def __init__(self, num_classes):
        super(KubeAIModel, self).__init__()
        self.num_classes = num_classes
        
        # Enhanced backbone for aerial imagery
        self.backbone = nn.SequentialCell([
            # Block 1 - Edge detection for aerial views
            nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2 - Pattern recognition
            nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3 - Feature extraction
            nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4 - High-level features
            nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5 - Deep features for small objects
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        ])
        
        # Animal classification head
        self.classifier = nn.SequentialCell([
            nn.Dense(1024 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(2048, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, num_classes)
        ])
        
        # Bounding box regression for animal location
        self.bbox_regressor = nn.SequentialCell([
            nn.Dense(1024 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(2048, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, 4),
            nn.Sigmoid()
        ])
        
        self.flatten = nn.Flatten()
    
    def construct(self, x):
        # Extract aerial features
        features = self.backbone(x)
        features = self.flatten(features)
        
        # Predict animal type and location
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        
        return class_logits, bbox_preds

class KubeAIDetector:
    """KUBE-AI Real-time Detector"""
    
    def __init__(self, model_path, num_classes=10):
        self.num_classes = num_classes
        self.class_list = [
            'cattle', 'goat', 'sheep', 'elephant', 'zebra',
            'giraffe', 'buffalo', 'antelope', 'lion', 'leopard'
        ]
        
        # Create and load model
        self.model = KubeAIModel(num_classes=num_classes)
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(self.model, param_dict)
        self.model.set_train(False)
        
        print("🚁 KUBE-AI: Aerial Intelligence System Ready")
        print(f"🐾 Detecting: {', '.join(self.class_list)}")
    
    def detect(self, image_path):
        """Detect animals in aerial imagery"""
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize and normalize
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_tensor = Tensor(image_array[np.newaxis, :], ms.float32)
        
        # AI Inference
        class_logits, bbox_preds = self.model(image_tensor)
        
        # Get predictions
        class_probs = nn.Softmax(axis=1)(class_logits)
        confidence = float(np.max(class_probs.asnumpy()))
        predicted_class = int(np.argmax(class_probs.asnumpy()))
        bbox = bbox_preds.asnumpy()[0]
        
        # Convert to pixel coordinates
        xmin = int(bbox[0] * orig_width)
        ymin = int(bbox[1] * orig_height)
        xmax = int(bbox[2] * orig_width)
        ymax = int(bbox[3] * orig_height)
        
        inference_time = time.time() - start_time
        
        # KUBE Platform result
        result = {
            'detection_id': f"kube_{int(time.time())}",
            'animal_type': self.class_list[predicted_class] if predicted_class < len(self.class_list) else f'unknown_{predicted_class}',
            'confidence': confidence,
            'bbox': [xmin, ymin, xmax, ymax],
            'image_size': [orig_width, orig_height],
            'inference_time_ms': inference_time * 1000,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'kube_module': self.get_kube_module(self.class_list[predicted_class] if predicted_class < len(self.class_list) else 'unknown'),
            'alert_level': self.get_alert_level(confidence, self.class_list[predicted_class] if predicted_class < len(self.class_list) else 'unknown')
        }
        
        return result, image
    
    def get_kube_module(self, animal_type):
        """Route to appropriate KUBE module"""
        livestock = ['cattle', 'goat', 'sheep']
        predators = ['lion', 'leopard']
        wildlife = ['elephant', 'zebra', 'giraffe', 'buffalo', 'antelope']
        
        if animal_type in livestock:
            return 'KUBE-Farm'
        elif animal_type in predators:
            return 'KUBE-Park (Critical)'
        elif animal_type in wildlife:
            return 'KUBE-Park'
        else:
            return 'KUBE-Land'
    
    def get_alert_level(self, confidence, animal_type):
        """Determine alert priority"""
        predators = ['lion', 'leopard']
        
        if animal_type in predators and confidence > 0.7:
            return 'CRITICAL - Predator Alert'
        elif confidence > 0.8:
            return 'HIGH - Confirmed Detection'
        elif confidence > 0.6:
            return 'MEDIUM - Probable Detection'
        else:
            return 'LOW - Possible Detection'
    
    def visualize(self, image, result, output_path):
        """Create detection visualization"""
        draw = ImageDraw.Draw(image)
        bbox = result['bbox']
        
        # Color by module
        colors = {
            'KUBE-Farm': 'green',
            'KUBE-Park (Critical)': 'red', 
            'KUBE-Park': 'blue',
            'KUBE-Land': 'orange'
        }
        color = colors.get(result['kube_module'], 'yellow')
        
        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=4)
        
        # Draw labels
        label = f"{result['animal_type']}: {result['confidence']:.2f}"
        module = f"Module: {result['kube_module']}"
        alert = f"Alert: {result['alert_level']}"
        
        draw.text((bbox[0], bbox[1] - 60), label, fill=color, stroke_width=2, stroke_fill='white')
        draw.text((bbox[0], bbox[1] - 40), module, fill=color, stroke_width=1, stroke_fill='white')
        draw.text((bbox[0], bbox[1] - 20), alert, fill=color, stroke_width=1, stroke_fill='white')
        
        image.save(output_path)
        print(f"🖼️ Visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='KUBE-AI Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to aerial image')
    parser.add_argument('--output_path', type=str, default='detection_result.jpg', help='Output path')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = KubeAIDetector(args.model_path, args.num_classes)
    
    # Run detection
    print(f"\n🔍 Analyzing: {args.image_path}")
    result, image = detector.detect(args.image_path)
    
    # Display results
    print("\n" + "="*50)
    print("🚁 KUBE-AI DETECTION RESULTS")
    print("="*50)
    print(f"🐾 Animal: {result['animal_type']}")
    print(f"📊 Confidence: {result['confidence']:.4f}")
    print(f"📍 Location: {result['bbox']}")
    print(f"🎯 Module: {result['kube_module']}")
    print(f"🚨 Alert: {result['alert_level']}")
    print(f"⚡ Time: {result['inference_time_ms']:.1f}ms")
    print("="*50)
    
    # Create visualization
    detector.visualize(image, result, args.output_path)
    
    # Save JSON results
    json_path = args.output_path.replace('.jpg', '_result.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"📄 Results saved: {json_path}")

if __name__ == '__main__':
    main()