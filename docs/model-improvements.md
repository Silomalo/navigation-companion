# Model Improvements Guide

> Referenced by: `src/detector.py` → `Detector.load()`
> Referenced by: `src/config.py` → `OBSTACLE_MAP`

This document covers how to improve the speed and accuracy of the YOLOv8n
object detection model used in the Navigation Companion.

---

## Current Capabilities (YOLOv8n + COCO-80)

| Target | COCO status | Quality |
|---|---|---|
| Pedestrians | ✅ `person` class | Good |
| Cyclists | ✅ `bicycle` + `person` | Good |
| Cars / trucks / buses / trains | ✅ native | Good |
| Animals — large (dog, horse, cow, bear…) | ✅ 8 classes mapped | Good |
| Animals — small (cat, bird) | ✅ mapped | Moderate |
| Traffic lights / stop signs | ✅ native | Good |
| Suitcases / skateboards / balls | ✅ mapped as path hazards | Moderate |
| **Staircases** | ❌ not in COCO | **Needs custom training** |
| **Stones / pebbles on path** | ❌ not in COCO | **Needs custom training** |
| **Road surface / tarmac** | ❌ requires segmentation | **Needs YOLOv8-seg** |
| **Road signs (general)** | ⚠️ stop sign + traffic light only | Partial — needs training |
| **Children crawling** | ⚠️ detected as `person`, no pose | **Needs pose model** |
| **Cycle drivers** | ⚠️ `person` + `bicycle` separate | No combined label |

---

## 1 — Speed: ONNX (Already Active ✅)

ONNX export is now **fully automatic**. On first run `Detector.load()`:

1. Downloads `yolov8n.pt` if not cached
2. Exports it to `data/models/yolov8n.onnx` (~10 s, one-time)
3. Loads the ONNX model on every subsequent run

No manual steps are required. Just run `pip install -r requirements.txt` and start.

| Model | RPi 5 | macOS (Intel) |
|---|---|---|
| `yolov8n.pt` PyTorch | ~15 fps | ~30 fps |
| `yolov8n.onnx` **active** | ~28 fps | ~55 fps |

---

## 2 — Accuracy: Upgrade Model Variant

Change `YOLO_MODEL_NAME` and `YOLO_ONNX_PATH` in `src/config.py`:

| Model | Size | mAP50 | RPi 5 (ONNX) | Best for |
|---|---|---|---|---|
| `yolov8n` | 6 MB | 37.3 | ~28 fps | Current — real-time on RPi |
| `yolov8s` | 22 MB | 44.9 | ~15 fps | Balanced accuracy / speed |
| `yolov8m` | 52 MB | 50.2 | ~8 fps | High accuracy, slower |

**Recommended next step:** switch to `yolov8s` — +7.6 mAP, still real-time via ONNX:

```python
# src/config.py
YOLO_MODEL_NAME: str = "yolov8s.pt"
YOLO_ONNX_PATH:  Path = MODELS_DIR / "yolov8s.onnx"
```

The ONNX export happens automatically on the next run.

---

## 3 — Staircases (Custom Training)

Stairs are the highest-priority missing class. `"stairs"` exists in `OBSTACLE_MAP`
but the base COCO model never produces that label — it requires fine-tuning.

### Recommended dataset
- **Roboflow Universe — Stairs Detection**
  https://universe.roboflow.com/search?q=stairs
  ~3,000 labelled images, multiple angles, indoor/outdoor.

### Fine-tuning steps

```bash
# 1. Download dataset (free Roboflow account required)
pip install roboflow
python - <<'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace().project("stairs-detection")
dataset = project.version(1).download("yolov8", location="data/datasets/stairs")
EOF

# 2. Fine-tune from yolov8n checkpoint (transfer learning — fast, ~1 h on GPU)
yolo detect train \
  model=data/models/yolov8n.pt \
  data=data/datasets/stairs/data.yaml \
  epochs=50 \
  imgsz=640 \
  project=data/runs \
  name=stairs_ft

# 3. Move best weights and update config
cp data/runs/stairs_ft/weights/best.pt data/models/yolov8n_custom.pt
```

```python
# src/config.py
YOLO_MODEL_NAME: str = "yolov8n_custom.pt"
YOLO_ONNX_PATH:  Path = MODELS_DIR / "yolov8n_custom.onnx"  # auto-exported on next run
```

---

## 4 — Stones & Pebbles on Path (Custom Dataset)

No public dataset covers small ground-level hazards. You must collect and
annotate your own data from the target environment.

### Data collection
1. Mount the chest camera and walk your regular routes.
2. Record 20–30 min of footage at different times of day and lighting.
3. Extract one frame every 0.5 s:
   ```bash
   ffmpeg -i video.mp4 -vf fps=2 data/datasets/pebbles/frames/%04d.jpg
   ```

### Annotation tools
| Tool | Hosting | Limit | Link |
|---|---|---|---|
| Roboflow | Cloud | 1,000 images free | https://roboflow.com |
| CVAT | Self-hosted | Unlimited | https://cvat.ai |
| Label Studio | Local | Unlimited | https://labelstud.io |

### Suggested classes
```
stone_small    # < 5 cm — trip hazard
stone_large    # > 5 cm — step-around hazard
puddle         # wet / slippery surface
pothole        # road defect
```

### Training
Same `yolo detect train` command as Section 3, pointing at your dataset.

---

## 5 — Road Signs (Comprehensive)

COCO covers only `stop sign` and `traffic light`. For full coverage:

### Public datasets
| Dataset | Classes | Region | Link |
|---|---|---|---|
| GTSRB | 43 | Germany | http://benchmark.ini.rub.de |
| LISA | 47 | USA | https://cvrr.ucsd.edu/LISA |
| Roboflow Road Signs | Mixed | Global | https://universe.roboflow.com/search?q=road+signs |

### Approach
Fine-tune `yolov8s.pt` on road signs. Run two models in parallel — obstacles
(model A) and signs (model B) — then merge `DetectionResult` lists before
passing to the navigator.

---

## 6 — Children Crawling (Pose Estimation)

A crawling child has the same bounding-box label (`person`) as a standing adult.
Detection alone cannot distinguish them — pose estimation is required.

### Solution: YOLOv8-Pose

```python
from ultralytics import YOLO
pose_model = YOLO("yolov8n-pose.pt")   # auto-downloads (~7 MB)
results = pose_model(frame)
# keypoints[15] = left ankle,  keypoints[16] = right ankle
# keypoints[11] = left hip,    keypoints[12] = right hip
# Crawling check: ankle.y ≈ hip.y  (within POSE_CRAWL_THRESH)
```

Add to `src/config.py`:
```python
YOLO_POSE_MODEL_PATH: Path = MODELS_DIR / "yolov8n-pose.pt"
POSE_CRAWL_THRESH: float = 0.15   # ankle-to-hip vertical normalised ratio
```

---

## 7 — Road Surface / Tarmac

Object detection draws bounding boxes — it cannot label every pixel.
Road surface requires **semantic segmentation**.

### Solution: YOLOv8-Seg

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
```

The segmentation mask identifies driveable surface vs. obstacles and can
feed into a path-planning or obstacle-avoidance module.

---

## 8 — Upgrade Path

```
Phase 1 ✅ DONE     ONNX auto-export        2–3× speed, zero effort
Phase 2  (week 1)   Switch to yolov8s       +7 mAP, still real-time via ONNX
Phase 3  (week 2–4) Fine-tune + stairs      actual staircase detection
Phase 4  (month 2)  Custom pebble dataset   small ground-hazard detection
Phase 5  (month 3)  Add pose model          crawling child detection
Phase 6  (month 4)  Road sign dataset       comprehensive sign recognition
```
