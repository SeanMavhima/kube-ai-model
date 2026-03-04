import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import argparse

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

def predict(image_path, model_path):
    classes = ['cattle', 'goat', 'sheep', 'elephant', 'zebra', 'giraffe', 'buffalo', 'antelope', 'lion', 'leopard']
    
    # Load model
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, 1)[0][predicted].item()
    
    animal = classes[predicted.item()]
    print(f"🐾 KUBE-AI Detection:")
    print(f"Animal: {animal}")
    print(f"Confidence: {confidence:.2%}")
    
    return animal, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='../data/JPEGImages/img_000001.jpg')
    parser.add_argument('--model', default='../models/kube_pytorch.pth')
    args = parser.parse_args()
    
    predict(args.image, args.model)