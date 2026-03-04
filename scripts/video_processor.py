#!/usr/bin/env python3
"""
KUBE-AI Video Processing Module
Process drone videos for animal detection
"""

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import os
import time

class VideoProcessor:
    """Process videos for animal detection"""
    
    def __init__(self, model, class_list):
        self.model = model
        self.class_list = class_list
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def process_video(self, video_path, output_path=None, frame_skip=5):
        """Process video and detect animals"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections = []
        frame_count = 0
        
        print(f"🎥 Processing video: {video_path}")
        print(f"📊 FPS: {fps}, Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Run detection
            detection = self.detect_frame(pil_image, frame_count, fps)
            if detection:
                detections.append(detection)
                print(f"Frame {frame_count}: {detection['animal']} ({detection['confidence']:.2f})")
            
            frame_count += 1
        
        cap.release()
        
        # Save results
        if output_path:
            self.save_results(detections, output_path)
        
        return detections
    
    def detect_frame(self, pil_image, frame_num, fps):
        """Detect animals in a single frame"""
        # Preprocess
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            class_logits, bbox_preds = self.model(image_tensor)
            
            probs = torch.softmax(class_logits, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            bbox = bbox_preds[0].cpu().numpy()
        
        # Only return if confidence is high enough
        if confidence > 0.7:
            return {
                'frame': frame_num,
                'timestamp': frame_num / fps,
                'animal': self.class_list[predicted_class],
                'confidence': confidence,
                'bbox': bbox.tolist()
            }
        
        return None
    
    def save_results(self, detections, output_path):
        """Save detection results to JSON"""
        results = {
            'total_detections': len(detections),
            'animals_found': list(set([d['animal'] for d in detections])),
            'detections': detections
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_path}")

def main():
    """Demo video processing"""
    print("🎥 KUBE-AI Video Processing Demo")
    
    # This would use your trained model
    # For demo purposes, we'll show the structure
    video_path = "drone_footage.mp4"
    
    if os.path.exists(video_path):
        # processor = VideoProcessor(model, class_list)
        # detections = processor.process_video(video_path, "video_results.json")
        print(f"Would process: {video_path}")
    else:
        print("No video file found for demo")

if __name__ == '__main__':
    main()