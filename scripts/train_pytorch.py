import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import argparse

class AnimalDataset(Dataset):
    def __init__(self, data_dir):
        self.images = []
        self.labels = []
        self.classes = ['cattle', 'goat', 'sheep', 'elephant', 'zebra', 'giraffe', 'buffalo', 'antelope', 'lion', 'leopard']
        
        img_dir = os.path.join(data_dir, 'JPEGImages')
        ann_dir = os.path.join(data_dir, 'Annotations')
        
        if os.path.exists(img_dir):
            for img_file in os.listdir(img_dir):
                if img_file.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(img_dir, img_file))
                    xml_path = os.path.join(ann_dir, img_file.replace('.jpg', '.xml'))
                    self.labels.append(self._get_label(xml_path))
    
    def _get_label(self, xml_path):
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            obj = tree.find('.//object/name')
            if obj is not None:
                name = obj.text.lower()
                return self.classes.index(name) if name in self.classes else 0
        return 0
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB').resize((224, 224))
        img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
    
    def __len__(self):
        return len(self.images)

class KubeModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 28 * 28, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(self.flatten(x))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    dataset = AnimalDataset('../data')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = KubeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/kube_pytorch.pth')
    print("Training complete!")

if __name__ == '__main__':
    train()