#!/usr/bin/env python3
import os
import sys
sys.path.append('..')

# Quick training setup
def quick_train():
    print("🚁 KUBE-AI Quick Training")
    
    # 1. Prepare data
    print("📁 Preparing data...")
    os.system('python prepare_data.py')
    
    # 2. Train model
    print("🔥 Training model...")
    os.system('python train_minimal.py')
    
    print("✅ Quick training complete!")

if __name__ == '__main__':
    quick_train()