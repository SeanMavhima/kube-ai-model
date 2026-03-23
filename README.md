# KUBE-AI: Eyes in the Sky for Africa

> **Protecting wildlife and livestock through intelligent aerial monitoring**
>
> **Huawei ICT Competition 2025–2026 — Innovation Track**

KUBE-AI transforms drone footage into actionable intelligence, giving African communities the power to monitor animals in real-time — from detecting lost cattle to preventing elephant poaching.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Supported Animals](#supported-animals)
4. [Technical Architecture](#technical-architecture)
5. [Technology Stack](#technology-stack)
6. [Reproduction Steps](#reproduction-steps)
7. [Training](#training)
8. [Inference](#inference)
9. [Evaluation](#evaluation)
10. [Project Structure](#project-structure)
11. [Utility Scripts](#utility-scripts)
12. [Performance Benchmarks](#performance-benchmarks)
13. [Real-World Impact](#real-world-impact)
14. [License](#license)

---

## Problem Statement

Every year, African farmers lose **millions of dollars** worth of livestock to theft, predators, and disease. Wildlife populations face unprecedented threats from poaching and habitat loss. Traditional monitoring methods are too slow, too expensive, and too limited in coverage.

## Solution Overview

KUBE-AI is an aerial-first AI system built specifically for **African landscapes**:

- **Aerial-First Design** — Optimized for drone and satellite imagery
- **Africa-Focused** — Trained on animals found across the continent
- **Lightning Fast** — Inference in under 100 ms per image
- **Smart Alerts** — Distinguishes livestock from predators and triggers appropriate alerts

## Supported Animals

| KUBE-Farm (Livestock) | KUBE-Park (Wildlife) | Alert Level |
|---|---|---|
| Cattle | Elephants | Standard |
| Goats | Zebras | Standard |
| Sheep | Giraffes | Standard |
| | Buffalo | Standard |
| | Antelopes | Standard |
| | **Lions** | **CRITICAL** |
| | **Leopards** | **CRITICAL** |

---

## Technical Architecture

Custom dual-head CNN for simultaneous classification and localization:

```
Aerial Image (224×224)
    ↓
Enhanced Backbone (5 conv blocks → 1024 channels)
    ↓
Feature Map (1024×7×7)
    ↓
    ├── Species Classifier  → animal class
    └── Location Regressor  → [xmin, ymin, xmax, ymax]
```

## Technology Stack

| Layer | Technology |
|---|---|
| Framework | MindSpore 2.0+ (Huawei) |
| Language | Python 3.7–3.9 |
| Model | 5-block CNN with BatchNorm, Dropout, dual heads |
| Dataset Format | Pascal VOC (XML annotations) |
| Training | Adam optimizer, multi-loss (CE + MSE) |
| Cloud | Huawei ModelArts, OBS, Moxing |
| Preprocessing | PIL/Pillow, NumPy |

---

## Reproduction Steps

### Prerequisites

- Python 3.7–3.9 (3.8 recommended)
- MindSpore 2.0+
- 8 GB RAM minimum

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/amiparadis250/Kube-ai.git
cd Kube-ai

python -m venv kube_env

# Windows
kube_env\Scripts\activate
# Linux / macOS
source kube_env/bin/activate

pip install --upgrade pip
pip install -r requirements_mindspore.txt
```

### 2. Prepare the Dataset

The repository includes partial annotations under `data/Annotations/` and `data/TestAnnotations/`. To download the full image datasets:

```bash
# Download source datasets (requires Roboflow & Kaggle API keys)
python download_datasets.py

# Convert and prepare VOC-format data
python prepare_data.py
```

After preparation, the `data/` directory will contain:

```
data/
├── JPEGImages/       # ~16 600 training images
├── Annotations/      # VOC XML annotations
├── TestImages/       # ~1 000 test images
└── TestAnnotations/  # Test XML annotations
```

### 3. Train the Model

```bash
python kube_ai_train.py --epochs 20 --batch_size 4 --lr 0.0001
```

Key arguments:

| Flag | Default | Description |
|---|---|---|
| `--data_url` | `./data` | Path to dataset |
| `--train_url` | `./models` | Output directory for checkpoints |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--num_classes` | `10` | Number of animal classes |

Training logs are saved to `logs/training.log`. Checkpoints are saved to `models/`.

### 4. Run Inference

```bash
python kube_ai_inference.py \
    --model_path models/kube_ai_final.ckpt \
    --image_path data/JPEGImages/img_000001.jpg \
    --output_path results/detection_result.jpg
```

Output includes a visualized image with bounding boxes and a JSON file:

```json
{
    "detection_id": "kube_1642248622",
    "animal_type": "elephant",
    "confidence": 0.94,
    "bbox": [120, 150, 450, 400],
    "kube_module": "KUBE-Park",
    "alert_level": "HIGH - Confirmed Detection",
    "inference_time_ms": 87.3,
    "timestamp": "2024-01-15 14:30:22"
}
```

### 5. Evaluate Accuracy

```bash
python evaluate_model.py --model_path models/kube_ai_final.ckpt
```

This runs inference on the test set and prints per-class precision, recall, F1-score, and overall accuracy. Results are saved to `logs/evaluation.log` and `results/evaluation_results.json`.

---

## Project Structure

```
Kube-ai/
├── kube_ai_train.py            # Training entry point
├── kube_ai_inference.py        # Inference entry point
├── evaluate_model.py           # Evaluation entry point
├── download_datasets.py        # Dataset downloader
├── prepare_data.py             # VOC data preparation
├── requirements_mindspore.txt  # MindSpore dependencies
├── requirements.txt            # General dependencies
├── data/
│   ├── JPEGImages/             # Training images
│   ├── Annotations/            # Training annotations (VOC XML)
│   ├── TestImages/             # Test images
│   └── TestAnnotations/        # Test annotations (VOC XML)
├── models/                     # Saved checkpoints
├── logs/                       # Training & evaluation logs
├── results/                    # Inference outputs
├── visualizations/             # Accuracy plots
└── scripts/                    # Additional utilities (see scripts/SCRIPTS.md)
    ├── SCRIPTS.md              # Scripts documentation
    ├── api.py                  # REST API server
    ├── test_api.py             # API smoke test
    ├── video_processor.py      # Video stream processing
    ├── evaluate_accuracy.py    # Accuracy plot generator
    ├── visualize_data.py       # Dataset visualization
    └── visualize_training.py   # Training curve plots
```

## Utility Scripts

The `scripts/` directory contains additional tools for deployment, visualization, and analysis.

### REST API Server

Serve the model as an HTTP endpoint for real-time detection:

```bash
cd scripts
set KUBE_MODEL_PATH=../models/kube_ai_final.ckpt   # Windows
python api.py
```

| Method | URL | Description |
|---|---|---|
| `POST` | `/predict` | Upload an image, receive detection JSON |
| `GET` | `/health` | Health check |

Test the API:

```bash
python test_api.py
```

### Video Processing

Detect animals in drone video footage frame-by-frame:

```bash
cd scripts
python video_processor.py \
    --model_path ../models/kube_ai_final.ckpt \
    --video_path drone_footage.mp4 \
    --output_path ../results/video_results.json \
    --frame_skip 5
```

### Visualization Tools

```bash
cd scripts

# Generate accuracy plots from evaluation results
python evaluate_accuracy.py

# Visualize dataset distribution and sample images
python visualize_data.py

# Plot training loss and accuracy curves
python visualize_training.py
```

All visualizations are saved to the `visualizations/` directory.

---

## Performance Benchmarks

| Metric | Value | Notes |
|---|---|---|
| Inference Speed | < 100 ms | Per image on CPU |
| Accuracy | 90%+ | On aerial test imagery |
| Detection Range | 50–500 m | Optimal drone altitude |
| Supported FPS | 10+ | Real-time capable |

---

## Real-World Impact

**For Farmers:** SMS alerts when livestock wander off, automatic herd counting, predator warnings.

**For Conservationists:** Elephant migration tracking, instant poaching alerts, non-invasive wildlife census.

**For Rangers:** AI-prioritized patrol routes, minute-level threat response, 10× ground coverage.

---

## License

**Huawei ICT Competition 2025–2026 — Innovation Track**
**© 2025 KUBE Platform**

Open for academic research, conservation projects, community initiatives, and educational purposes. Commercial use requires permission.

---

*"In the vast landscapes of Africa, every animal matters. KUBE-AI ensures none go unseen."*
