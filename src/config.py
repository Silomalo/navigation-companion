"""
config.py — Central configuration for the Navigation Assistant.

Edit this file to match your hardware.  Every other module imports
from here — no magic numbers anywhere else.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
MAPS_DIR = DATA_DIR / "maps"

# Create directories on first import.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# ── Execution mode ─────────────────────────────────────────────────────────────
# Set SIMULATE=1 in the environment to run in simulation mode (no RPi hardware).
SIMULATE: bool = os.getenv("SIMULATE", "0") == "1"

# ── Camera ─────────────────────────────────────────────────────────────────────
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
CAMERA_FPS: int = 30  # capture FPS from hardware
DETECT_FPS: int = 10  # inference FPS (sub-sample to save CPU)
CAMERA_FORMAT: str = "MJPEG"  # picamera2 format

# Simulation: index of the OpenCV VideoCapture source (0 = default webcam).
SIM_CAMERA_INDEX: int = 0

# ── YOLOv8 Object Detection ────────────────────────────────────────────────────
# Model variant: yolov8n (nano, 6 MB) is ideal for RPi 5 (~15 fps).
# Options: yolov8n, yolov8s, yolov8m  (larger = more accurate, slower)
YOLO_MODEL_NAME: str = "yolov8n.pt"
YOLO_MODEL_PATH: Path = MODELS_DIR / YOLO_MODEL_NAME
YOLO_CONF_THRESH: float = 0.45  # minimum detection confidence
YOLO_IOU_THRESH: float = 0.45  # NMS IoU threshold
YOLO_IMG_SIZE: int = 640  # inference resolution (square)
YOLO_DEVICE: str = "cpu"  # "cpu" on RPi; "cuda" if GPU available

# ── Obstacle zones (fraction of frame width) ──────────────────────────────────
# The frame is divided into 3 horizontal columns.
ZONE_LEFT_EDGE: float = 0.33  # 0.0 → 0.33 = left zone
ZONE_RIGHT_EDGE: float = 0.67  # 0.67 → 1.0 = right zone
# 0.33 → 0.67 = centre zone (path ahead)

# Distance estimation from bounding-box height.
# These thresholds are in fraction of frame height.
DIST_CLOSE_THRESH: float = 0.55  # bbox_h > this → "very close"
DIST_MEDIUM_THRESH: float = 0.30  # bbox_h > this → "ahead"
# bbox_h ≤ 0.30 → "in the distance"

# ── Navigation feedback ────────────────────────────────────────────────────────
# Minimum seconds between two spoken navigation instructions.
# Prevents audio spam while the user is walking.
FEEDBACK_MIN_INTERVAL_S: float = 2.5

# Urgency thresholds: if an obstacle is this close AND in centre, say it immediately.
URGENT_CLOSE_FRACTION: float = 0.55  # same as DIST_CLOSE_THRESH

# ── Microphone / Speech-to-Text ───────────────────────────────────────────────
MIC_SAMPLE_RATE: int = 16_000  # Hz — Whisper requires 16 kHz
MIC_CHANNELS: int = 1  # mono
MIC_DTYPE: str = "int16"  # sample format
MIC_BLOCK_FRAMES: int = 8_000  # 0.5 s per capture block
MIC_SILENCE_THRESH: float = 300.0  # RMS threshold for voice activity detection

# Whisper model size: "tiny" (39 MB) runs in real-time on RPi 5.
# Options: tiny, base, small, medium, large  (larger = more accurate, slower)
WHISPER_MODEL: str = "tiny"

# ── Text-to-Speech ────────────────────────────────────────────────────────────
TTS_RATE: int = 165  # words per minute (lower = clearer)
TTS_VOLUME: float = 0.95  # 0.0 – 1.0
# Voice selection: None = system default.  Set to a voice name string to override.
# List available voices:  python -c "import pyttsx3; e=pyttsx3.init(); print([v.name for v in e.getProperty('voices')])"
TTS_VOICE: str | None = None

# ── Topological Map ────────────────────────────────────────────────────────────
TOPO_MAP_FILE: Path = MAPS_DIR / "routes.json"
# A location is considered "known" if its feature vector cosine-similarity
# with a stored node exceeds this threshold.
TOPO_SIMILARITY_THRESH: float = 0.72
# Minimum seconds between recording new topological nodes.
TOPO_RECORD_INTERVAL_S: float = 5.0

# ── COCO class → navigation category mapping ──────────────────────────────────
# Keys are COCO class names (YOLOv8 default labels).
# Values are (nav_label, priority) where priority 1=urgent, 2=normal, 3=info.
OBSTACLE_MAP: dict[str, tuple[str, int]] = {
    # --- Dynamic / moving ---
    "person": ("person", 1),
    "bicycle": ("bicycle", 1),
    "car": ("car", 1),
    "motorcycle": ("motorcycle", 1),
    "bus": ("bus", 1),
    "truck": ("truck", 1),
    # --- Static path hazards ---
    "chair": ("chair", 2),
    "bench": ("bench", 2),
    "dining table": ("table", 2),
    "potted plant": ("plant", 2),
    "fire hydrant": ("fire hydrant", 2),
    "parking meter": ("parking meter", 2),
    "stop sign": ("stop sign", 2),
    "traffic light": ("traffic light", 2),
    # --- Elevation hazards ---
    "stairs": ("stairs", 1),  # custom label (not COCO default)
    # --- Indoor furniture ---
    "couch": ("couch", 2),
    "bed": ("bed", 2),
    "toilet": ("toilet", 3),
    "refrigerator": ("refrigerator", 3),
    "door": ("door", 2),  # custom label
}
