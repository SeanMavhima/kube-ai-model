# KUBE-AI: Eyes in the Sky for Africa 

> **Protecting wildlife and livestock through intelligent aerial monitoring**
> 
> **Huawei ICT Competition 2024 - Innovation Track Submission**

KUBE-AI transforms drone footage into actionable intelligence, giving African communities the power to monitor their animals in real-time. From detecting lost cattle to preventing elephant poaching, our AI sees what humans can't from the ground.

---

## The Problem We're Solving

Every year, African farmers lose **millions of dollars** worth of livestock to theft, predators, and disease. Meanwhile, wildlife populations face unprecedented threats from poaching and habitat loss. Traditional monitoring methods are:

- **Too slow** - By the time rangers arrive, it's often too late
- **Too expensive** - Hiring human monitors 24/7 is unsustainable  
- **Too limited** - Ground-based observation covers tiny areas

**KUBE-AI changes this.**

## What Makes KUBE-AI Different

Instead of generic object detection, we built something specifically for **African landscapes**:

 **Aerial-First Design** - Optimized for drone and satellite imagery  
 **Africa-Focused** - Trained on animals actually found here  
 **Lightning Fast** - Results in under 100ms  
 **Smart Alerts** - Knows the difference between a cow and a lion  

## Supported Animals

| **KUBE-Farm** (Livestock) | **KUBE-Park** (Wildlife) | **Alert Level** |
|---------------------------|---------------------------|-----------------|
| 🐄 Cattle | 🐘 Elephants | Standard |
| 🐐 Goats | 🦓 Zebras | Standard |
| 🐑 Sheep | 🦒 Giraffes | Standard |
| | 🐃 Buffalo | Standard |
| | 🦌 Antelopes | Standard |
| | 🦁 **Lions** | **CRITICAL** |
| | 🐆 **Leopards** | **CRITICAL** |

## Real-World Impact

**For Farmers:**
- Get SMS alerts when livestock wander off
- Count herds automatically from drone footage
- Receive predator warnings before attacks happen

**For Conservationists:**
- Track elephant migration patterns
- Get instant poaching alerts
- Conduct wildlife census without disturbing animals

**For Rangers:**
- Prioritize patrol routes based on AI insights
- Respond to threats in minutes, not hours
- Cover 10x more ground with the same resources

---

## Technical Architecture

KUBE-AI uses a custom CNN built specifically for aerial animal detection:

```
Aerial Image (224×224) 
    ↓
Enhanced Backbone (5 conv blocks)
    ↓  
Feature Extraction (1024×7×7)
    ↓
    ├─→ Species Classifier → "elephant"
    └─→ Location Detector → [x, y, w, h]
```

**Framework:** MindSpore (Huawei's AI platform)  
**Training:** Supervised learning on VOC-format annotations  
**Deployment:** Huawei ModelArts cloud infrastructure  

---

## 🛠️ Technology Stack

### **AI/ML Core:**
- **MindSpore** - Huawei's deep learning framework
- **Python** - Primary programming language
- **Custom CNN** - 5-layer convolutional neural network
- **VOC Dataset Format** - XML annotation standard
- **Supervised Learning** - Training methodology

### **Cloud & Infrastructure:**
- **Huawei ModelArts** - Cloud AI training platform
- **OBS (Object Storage)** - Dataset and model storage
- **Moxing Library** - Cloud-local data transfer
- **Docker** - Containerization for deployment

### *Data Processing:**
- **PIL/Pillow** - Image preprocessing
- **NumPy** - Numerical computations
- **XML Parser** - Annotation processing
- **Data Augmentation** - Image transformations

### **Model Architecture:**
- **Object Detection** - Dual-head CNN (classification + localization)
- **Batch Normalization** - Training stability
- **Dropout** - Overfitting prevention
- **Adam Optimizer** - Gradient descent algorithm
- **Multi-loss Function** - Classification + bounding box regression

---

## Getting Started

### Prerequisites
```bash
Python 3.7-3.9 (3.8 recommended)
MindSpore 2.0+
8GB RAM minimum
```

### Installation
```bash
# 1. Clone repository
git clone https://github.com/amiparadis250/Kube-ai.git
cd Kube-ai

# 2. Create virtual environment
python -m venv kube_env

# 3. Activate environment
# Windows:
kube_env\Scripts\activate
# Linux/Mac:
source kube_env/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements_mindspore.txt
```

### Quick Demo
```bash
# Make sure virtual environment is activated
# You should see (kube_env) in your prompt

# 1. Download sample aerial images
python download_dataset.py

# 2. Train the model (quick 20 epochs)
python kube_ai_train.py --epochs 20 --batch_size 4

# 3. Test on an image
python kube_ai_inference.py \
    --model_path models/kube_ai_final.ckpt \
    --image_path data/JPEGImages/cattle_001.jpg \
    --output_path result.jpg

# 4. Deactivate when done
deactivate
```

---

## Understanding the Output

When KUBE-AI detects an animal, it provides:

```json
{
    "detection_id": "kube_1642248622",
    "animal_type": "elephant",
    "confidence": 0.94,
    "bbox": [120, 150, 450, 400],
    "kube_module": "KUBE-Park",
    "alert_level": "HIGH - Confirmed Detection",
    "timestamp": "2024-01-15 14:30:22"
}
```

**What this means:**
- **94% confident** it found an elephant
- **Located at** pixel coordinates [120, 150, 450, 400]
- **Routed to** KUBE-Park (wildlife module)
- **Alert level** HIGH (rangers should investigate)

---

## Monitoring Training & Results

### Training Progress
```bash
# Watch training in real-time
tail -f logs/training.log

# Sample output:
# 2024-01-15 14:30:22 - Epoch 5/20 | Loss: 1.2345 | Time: 45.2s
# 2024-01-15 14:31:07 - Epoch 6/20 | Loss: 1.1892 | Time: 44.8s
```

### Inference Results
Every detection creates:
- **Visual result** - Image with bounding box and label
- **JSON data** - Machine-readable detection info
- **Performance metrics** - Confidence scores and timing

```bash
# Example inference output:
 KUBE-AI DETECTION RESULTS
 Animal: elephant
 Confidence: 85.34% Location: [120, 150, 450, 400]
 Module: KUBE-Park
 Alert: HIGH - Confirmed Detection
 Processing Time: 87ms
```

---

## Project Structure

```
Kube-ai/
├── kube_ai_train.py           # Main training script
├── kube_ai_inference.py       # Detection engine
├── download_dataset.py        # Sample data downloader
├── requirements_mindspore.txt  # Dependencies
├── data/
│   ├── JPEGImages/           # Aerial photos
│   └── Annotations/          # Animal locations (XML)
├── models/                   # Trained AI models
└── logs/                     # Training history
```

---

## Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Inference Speed** | <100ms | Per image on CPU |
| **Accuracy** | 90%+ | On aerial imagery |
| **Detection Range** | 50m-500m | Optimal altitude |
| **Supported FPS** | 10+ | Real-time processing |

---

## License & Usage

**Huawei ICT Competition 2024 - Innovation Track**  
**Copyright © 2025-2026 KUBE Platform**

This project is built for Africa's future. Commercial use requires permission, but we encourage:
-  Academic research
- Conservation projects  
- Community initiatives
- Educational purposes

---

*"In the vast landscapes of Africa, every animal matters. KUBE-AI ensures none go unseen."*

** KUBE-AI: Protecting Africa's Future, One Detection at a Time** 🌍
