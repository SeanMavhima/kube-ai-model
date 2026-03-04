import os
import shutil
from PIL import Image
import xml.etree.ElementTree as ET

def prepare_data():
    datasets = ['../datasets_clean/african_wildlife', '../datasets_clean/animals_detection', 
                '../datasets_clean/animals_wild', '../datasets_clean/kaggle_cows']
    
    img_dir = '../data/JPEGImages'
    ann_dir = '../data/Annotations'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    
    count = 0
    for dataset in datasets:
        if not os.path.exists(dataset):
            continue
        
        for root, dirs, files in os.walk(dataset):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, file)
                    dst = os.path.join(img_dir, f"img_{count:06d}.jpg")
                    
                    try:
                        Image.open(src).convert('RGB').resize((640, 480)).save(dst, 'JPEG')
                        create_xml(f"img_{count:06d}.jpg", ann_dir, dataset)
                        count += 1
                    except:
                        continue
    
    print(f"Prepared {count} images")

def create_xml(img_name, ann_dir, dataset_path):
    animal = 'cattle'
    if 'wildlife' in dataset_path: animal = 'elephant'
    elif 'wild' in dataset_path: animal = 'zebra'
    
    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = img_name
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = '640'
    ET.SubElement(size, 'height').text = '480'
    ET.SubElement(size, 'depth').text = '3'
    
    obj = ET.SubElement(root, 'object')
    ET.SubElement(obj, 'name').text = animal
    
    bbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = '160'
    ET.SubElement(bbox, 'ymin').text = '120'
    ET.SubElement(bbox, 'xmax').text = '480'
    ET.SubElement(bbox, 'ymax').text = '360'
    
    ET.ElementTree(root).write(os.path.join(ann_dir, img_name.replace('.jpg', '.xml')))

if __name__ == '__main__':
    prepare_data()