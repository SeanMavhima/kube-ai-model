#!/usr/bin/env python3
# download_datasets.py - Download KUBE-AI datasets

import os
import subprocess
import sys
from roboflow import Roboflow

def download_kaggle_cows():
    """Download Kaggle cows dataset"""
    print("Downloading Kaggle Cows Dataset...")
    os.makedirs('./datasets/kaggle_cows', exist_ok=True)
    
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download', 
            'trainingdatapro/cows-detection-dataset',
            '-p', './datasets/kaggle_cows',
            '--unzip'
        ], check=True)
        print("Kaggle cows dataset downloaded")
    except:
        print("Setup kaggle.json first")

def download_roboflow_datasets():
    """Download Roboflow wildlife & livestock datasets"""
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if not api_key:
        print("Set ROBOFLOW_API_KEY environment variable first")
        return
    rf = Roboflow(api_key=api_key)
    
    datasets = [
        ('project-vbv5j', 'wildlife-yj7t1', 1),
        ('e055', 'wildlife-4satw', 1), 
        ('swiftdynamics', 'livestock-qocf0', 1)
    ]
    
    for workspace, project_name, version in datasets:
        print(f"Downloading {project_name}...")
        try:
            project = rf.workspace(workspace).project(project_name)
            project.version(version).download("yolov8", location=f"./datasets/{project_name}")
            print(f"{project_name} downloaded")
        except Exception as e:
            print(f"Failed: {e}")

def main():
    print("KUBE-AI Dataset Downloader")
    os.makedirs('./datasets', exist_ok=True)
    
    download_kaggle_cows()
    download_roboflow_datasets()
    
    print("KUBE datasets downloaded!")

if __name__ == '__main__':
    main()