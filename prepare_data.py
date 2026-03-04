#!/usr/bin/env python3
"""
KUBE-AI Data Preparation Script
Prepare datasets for aerial animal detection training
"""

import os
import shutil
from PIL import Image
import xml.etree.ElementTree as ET

def prepare_data():
    """Prepare training data from multiple dataset sources"""
    
    # Source directories
    datasets = [
        'datasets_clean/african_wildlife',
        'datasets_clean/animals_detection', 
        'datasets_clean/animals_wild',
        'datasets_clean/kaggle_cows'
    ]
    
    # Target structure
    target_dir = 'data'
    img_dir = os.path.join(target_dir, 'JPEGImages')
    ann_dir = os.path.join(target_dir, 'Annotations')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    
    count = 0
    
    for dataset in datasets:
        if not os.path.exists(dataset):
            continue
            
        print(f"Processing {dataset}...")
        
        # Look for images in common subdirectories
        for root, dirs, files in os.walk(dataset):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(root, file)
                    
                    # Copy image
                    dst_name = f"img_{count:06d}.jpg"
                    dst_path = os.path.join(img_dir, dst_name)
                    
                    try:
                        # Convert to RGB and resize
                        img = Image.open(src_path).convert('RGB')
                        img = img.resize((640, 480))  # Standard size
                        img.save(dst_path, 'JPEG', quality=95)
                        
                        # Create simple annotation
                        create_annotation(dst_name, ann_dir, dataset)
                        
                        count += 1
                        if count % 100 == 0:
                            print(f"Processed {count} images...")
                            
                    except Exception as e:
                        print(f"Error processing {src_path}: {e}")
    
    print(f"✅ Data preparation complete! {count} images prepared.")

def create_annotation(img_name, ann_dir, dataset_path):
    """Create VOC-style XML annotation"""
    
    # Determine animal class from dataset path
    if 'cow' in dataset_path.lower():
        animal_class = 'cattle'
    elif 'wildlife' in dataset_path.lower():
        animal_class = 'elephant'  # Default for wildlife
    elif 'wild' in dataset_path.lower():
        animal_class = 'zebra'     # Default for wild animals
    else:
        animal_class = 'cattle'    # Default fallback
    
    # Create XML annotation
    annotation = ET.Element('annotation')
    
    # Basic info
    ET.SubElement(annotation, 'filename').text = img_name
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = '640'
    ET.SubElement(size, 'height').text = '480'
    ET.SubElement(size, 'depth').text = '3'
    
    # Object (simplified - center of image)
    obj = ET.SubElement(annotation, 'object')
    ET.SubElement(obj, 'name').text = animal_class
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'
    
    bbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = '160'  # 25% from left
    ET.SubElement(bbox, 'ymin').text = '120'  # 25% from top
    ET.SubElement(bbox, 'xmax').text = '480'  # 75% from left
    ET.SubElement(bbox, 'ymax').text = '360'  # 75% from top
    
    # Save XML
    xml_name = img_name.replace('.jpg', '.xml')
    xml_path = os.path.join(ann_dir, xml_name)
    
    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)

if __name__ == '__main__':
    prepare_data()