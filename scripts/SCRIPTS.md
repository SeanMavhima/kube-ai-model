# Scripts — KUBE-AI Utilities

> All scripts use **MindSpore 2.0+**.
> Main entry points live at the **project root**. This `scripts/` folder contains additional utilities.

---

## Project Root (Main Entry Points)

| Script | Purpose | Usage |
|---|---|---|
| `kube_ai_train.py` | Train the model | `python kube_ai_train.py --epochs 20` |
| `kube_ai_inference.py` | Run detection on an image | `python kube_ai_inference.py --model_path models/kube_ai_final.ckpt --image_path img.jpg` |
| `evaluate_model.py` | Evaluate accuracy on test set | `python evaluate_model.py --model_path models/kube_ai_final.ckpt` |
| `prepare_data.py` | Convert datasets to VOC format | `python prepare_data.py` |
| `download_datasets.py` | Download source datasets | `python download_datasets.py` |

---

## scripts/ (Utilities)

| Script | Purpose |
|---|---|
| `api.py` | REST API server for real-time detection |
| `test_api.py` | Smoke-test client for the API |
| `video_processor.py` | Detect animals in drone video files |
| `evaluate_accuracy.py` | Generate accuracy plots from evaluation results |
| `visualize_data.py` | Dataset statistics and sample visualizations |
| `visualize_training.py` | Training loss / accuracy curve plots |

---

## API Server

Start the detection API:

```bash
cd scripts
set KUBE_MODEL_PATH=../models/kube_ai_final.ckpt
python api.py
```

Endpoints:

| Method | URL | Description |
|---|---|---|
| `POST` | `/predict` | Upload an image, get detection JSON |
| `GET` | `/health` | Health check |

Test it:

```bash
python test_api.py
```

---

## Video Processing

Process a drone video and save per-frame detections:

```bash
cd scripts
python video_processor.py \
    --model_path ../models/kube_ai_final.ckpt \
    --video_path drone_footage.mp4 \
    --output_path ../results/video_results.json \
    --frame_skip 5
```

Requires `opencv-python`.

---

## Accuracy Visualization

After running `evaluate_model.py` at the project root, generate plots:

```bash
cd scripts
python evaluate_accuracy.py
```

Reads `results/evaluation_results.json` and saves charts to `visualizations/accuracy_evaluation.png`.

---

## Dataset Visualization

```bash
cd scripts
python visualize_data.py
```

Saves output to `visualizations/dataset_analysis.png`.

---

## Training Curve Plots

```bash
cd scripts
python visualize_training.py
```

Reads `logs/training.log` and saves to `visualizations/training_progress.png`.

---

## Dependencies

```bash
pip install -r requirements_mindspore.txt   # Core
pip install -r requirements_viz.txt         # Visualization
pip install flask                            # API server
pip install opencv-python                    # Video processing
```
