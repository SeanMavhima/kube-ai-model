#!/usr/bin/env python3
"""
KUBE-AI Local Testing with Video Support
PyTorch version for testing before MindSpore deployment
"""

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import time
import cv2
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AerialAnimalDataset(Dataset):
    """Dataset for aerial animal images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_list = ['cattle', 'goat', 'sheep', 'elephant', 'zebra', 'giraffe', 'buffalo', 'antelope', 'lion', 'leopard']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        
        self.images = []
        self.annotations = []
        
        img_dir = os.path.join(root_dir, 'JPEGImages')
        ann_dir = os.path.join(root_dir, 'Annotations')
        
        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = os.path.splitext(img_file)[0]
                    xml_file = os.path.join(ann_dir, img_name + '.xml')
                    if os.path.exists(xml_file):
                        self.images.append(os.path.join(img_dir, img_file))
                        self.annotations.append(xml_file)
        
        logger.info(f"Found {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        if self.transform:
            image = self.transform(image)
        
        xml_path = self.annotations[idx]
        boxes, labels = self.parse_annotation(xml_path, orig_width, orig_height)
        
        if len(labels) > 0:
            label = labels[0]
            bbox = torch.tensor(boxes[0], dtype=torch.float32)
        else:
            label = 0
            bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        
        return image, torch.tensor(label, dtype=torch.long), bbox
    
    def parse_annotation(self, xml_file, img_width, img_height):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower()
            if class_name not in self.class_to_idx:
                continue
            
            class_id = self.class_to_idx[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / img_width
            ymin = float(bbox.find('ymin').text) / img_height
            xmax = float(bbox.find('xmax').text) / img_width
            ymax = float(bbox.find('ymax').text) / img_height
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)
        
        return boxes, labels

class KubeAIModel(nn.Module):
    """KUBE-AI Detection Model"""
    
    def __init__(self, num_classes):
        super(KubeAIModel, self).__init__()
        self.num_classes = num_classes
        
        # Backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # Bbox regression head
        self.bbox_regressor = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        
        return class_logits, bbox_preds

def process_video(model, video_path, class_list):
    """Process video for animal detection"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    detections = []
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    logger.info(f"Processing video at {fps} FPS")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 10th frame for speed
        if frame_count % 10 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            image_tensor = transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                class_logits, bbox_preds = model(image_tensor)
                probs = torch.softmax(class_logits, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
                
                if confidence.item() > 0.7:
                    detection = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'animal': class_list[predicted_class.item()],
                        'confidence': confidence.item()
                    }
                    detections.append(detection)
                    logger.info(f"Frame {frame_count}: {detection['animal']} ({detection['confidence']:.2f})")
        
        frame_count += 1
    
    cap.release()
    return detections

def main():
    parser = argparse.ArgumentParser(description='KUBE-AI Local Testing')
    parser.add_argument('--data_url', type=str, default='./datasets_clean/kaggle_cows', help='Data path')
    parser.add_argument('--train_url', type=str, default='./models', help='Model save path')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Classes')
    parser.add_argument('--video_path', type=str, help='Video file to process')
    
    args = parser.parse_args()
    
    os.makedirs(args.train_url, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚁 KUBE-AI Local Testing")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = AerialAnimalDataset(args.data_url, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KubeAIModel(num_classes=args.num_classes).to(device)
    
    # Training
    if len(dataset) > 0:
        cls_criterion = nn.CrossEntropyLoss()
        bbox_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        logger.info("🚀 Starting training...")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            
            for batch_idx, (images, labels, bboxes) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)
                
                optimizer.zero_grad()
                
                class_logits, bbox_preds = model(images)
                
                cls_loss = cls_criterion(class_logits, labels)
                bbox_loss = bbox_criterion(bbox_preds, bboxes)
                
                total_loss_batch = cls_loss + 2.0 * bbox_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save(model.state_dict(), os.path.join(args.train_url, 'kube_ai_local.pth'))
        logger.info("✅ Training completed!")
    
    # Video processing demo
    if args.video_path and os.path.exists(args.video_path):
        logger.info("🎥 Processing video...")
        detections = process_video(model, args.video_path, dataset.class_list)
        
        # Save video results
        with open('video_detections.json', 'w') as f:
            json.dump(detections, f, indent=2)
        
        logger.info(f"Found {len(detections)} animal detections in video")

if __name__ == '__main__':
    main()