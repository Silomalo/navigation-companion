"""
detector.py — YOLO obstacle detection and scene understanding.

Loads the configured Ultralytics YOLO pretrained model (COCO, 80 classes).
Provides:
  - Per-frame detection with bounding boxes and confidence scores.
  - Classification of detections into navigation-relevant categories.
  - Spatial zone tagging (LEFT / CENTRE / RIGHT).
  - Proximity estimation from bounding-box size.
  - A structured DetectionResult for the navigator to consume.

The model is downloaded automatically on first run, immediately exported
to ONNX (2-3× CPU speedup), and cached to config.MODELS_DIR.
Subsequent starts load the ONNX directly — no manual steps needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from ultralytics import YOLO  # type: ignore

from config import (
    DIST_CLOSE_THRESH,
    DIST_MEDIUM_THRESH,
    OBSTACLE_MAP,
    YOLO_CONF_THRESH,
    YOLO_DEVICE,
    YOLO_IMG_SIZE,
    YOLO_IOU_THRESH,
    YOLO_MODEL_NAME,
    YOLO_MODEL_PATH,
    YOLO_ONNX_PATH,
    ZONE_LEFT_EDGE,
    ZONE_RIGHT_EDGE,
)

log = logging.getLogger(__name__)


# ── Domain types ──────────────────────────────────────────────────────────────


class Zone(str, Enum):
    LEFT = "left"
    CENTRE = "ahead"  # "ahead" reads more naturally in speech
    RIGHT = "right"


class Proximity(str, Enum):
    CLOSE = "very close"
    MEDIUM = "ahead"
    FAR = "in the distance"


@dataclass
class Detection:
    """A single detected object with navigation metadata."""

    class_name: str  # raw COCO label (e.g. "person")
    nav_label: str  # human-friendly label (e.g. "person")
    priority: int  # 1 = urgent, 2 = normal, 3 = info
    confidence: float  # 0–1
    zone: Zone  # spatial position in frame
    proximity: Proximity  # estimated distance category
    bbox_xyxy: tuple[float, float, float, float]  # pixel coords
    bbox_norm: tuple[float, float, float, float]  # normalised 0–1

    @property
    def is_urgent(self) -> bool:
        return self.priority == 1 and self.proximity == Proximity.CLOSE

    @property
    def speech_phrase(self) -> str:
        """Short phrase suitable for TTS output."""
        return f"{self.nav_label} {self.zone.value} {self.proximity.value}"


@dataclass
class DetectionResult:
    """All detections from a single frame, with summary statistics."""

    frame_id: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    # Frame dimensions (needed by navigator for relative calculations)
    frame_w: int = 0
    frame_h: int = 0

    @property
    def urgent(self) -> list[Detection]:
        return [d for d in self.detections if d.is_urgent]

    @property
    def by_priority(self) -> list[Detection]:
        return sorted(self.detections, key=lambda d: (d.priority, -d.confidence))

    @property
    def has_centre_obstacle(self) -> bool:
        return any(d.zone == Zone.CENTRE for d in self.detections)

    @property
    def path_clear(self) -> bool:
        return not self.has_centre_obstacle

    def feature_vector(self) -> np.ndarray:
        """
        Returns a 80-element float32 vector of per-class confidence scores.
        Used by the topological map for location fingerprinting.
        Zero for classes not detected in this frame.
        """
        vec = np.zeros(80, dtype=np.float32)
        for d in self.detections:
            idx = _COCO_INDEX.get(d.class_name, -1)
            if 0 <= idx < 80:
                vec[idx] = max(vec[idx], d.confidence)
        return vec


# ── ONNX export helper ───────────────────────────────────────────────────────


def _export_onnx() -> None:
    """Export the .pt model to ONNX format (runs once, ~10 s).

    Writes to YOLO_ONNX_PATH. Called automatically by Detector.load()
    on first run so no manual step is ever needed.
    """
    from pathlib import Path as _Path

    log.info("[detector] exporting to ONNX — one-time setup, please wait …")
    pt_model = YOLO(str(YOLO_MODEL_PATH))
    out = pt_model.export(
        format="onnx",
        imgsz=YOLO_IMG_SIZE,
        simplify=True,  # fuse ops for faster CPU execution
        dynamic=False,  # fixed batch=1 for embedded/RPi deployment
        opset=17,
    )
    exported = _Path(str(out))
    if exported.resolve() != YOLO_ONNX_PATH.resolve():
        YOLO_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        exported.replace(YOLO_ONNX_PATH)
    log.info("[detector] ONNX model ready → %s", YOLO_ONNX_PATH)


# ── Detector class ────────────────────────────────────────────────────────────


class Detector:
    """
    Wraps a YOLO model with navigation-specific postprocessing.

    Usage:
        det = Detector()
        det.load()
        result = det.detect(bgr_frame, frame_id=0)
    """

    def __init__(self) -> None:
        self._model: Optional[YOLO] = None
        self._frame_count: int = 0

    def load(self) -> None:
        """Download the .pt weights (if missing), auto-export to ONNX on first
        run, then load the ONNX model for 2-3x faster CPU inference.

        Flow on every startup:
          1. If the configured .onnx exists → load it directly (fast path).
          2. If only the configured .pt exists → export to ONNX, then load ONNX.
          3. If neither exists        → download .pt, export to ONNX, load ONNX.

        See docs/model-improvements.md for tuning and custom-training details.
        """
        # ── Step 1: ensure the .pt weights are on disk ─────────────────────
        if not YOLO_ONNX_PATH.exists():
            if not YOLO_MODEL_PATH.exists():
                log.info("[detector] downloading %s …", YOLO_MODEL_NAME)
            YOLO(str(YOLO_MODEL_PATH))  # triggers download if missing

            # ── Step 2: export to ONNX (runs once, ~10 s) ─────────────────
            _export_onnx()

        # ── Step 3: load the ONNX model ────────────────────────────────────
        log.info("[detector] loading ONNX model")
        # task='detect' suppresses the auto-guess warning.
        # .to(device) is PyTorch-only and must NOT be called on ONNX models;
        # onnxruntime selects the execution provider (CPU) automatically.
        self._model = YOLO(str(YOLO_ONNX_PATH), task="detect")
        # Warm-up pass: avoids first-frame latency spike.
        dummy = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        log.info("[detector] model ready")

    def detect(self, frame: np.ndarray, frame_id: int = -1) -> DetectionResult:
        """
        Run YOLO inference on a BGR frame.

        Args:
            frame:    BGR uint8 ndarray from the camera.
            frame_id: monotonic frame counter for logging.

        Returns:
            DetectionResult with all navigation-relevant detections.
        """
        if self._model is None:
            raise RuntimeError("Call Detector.load() before detect()")

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        results = self._model(
            frame,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF_THRESH,
            iou=YOLO_IOU_THRESH,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000.0

        result = DetectionResult(
            frame_id=frame_id,
            timestamp=time.time(),
            inference_ms=inference_ms,
            frame_w=w,
            frame_h=h,
        )

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name: str = self._model.names[cls_id]
                conf: float = float(box.conf[0])

                # Skip classes not in our navigation map.
                if cls_name not in OBSTACLE_MAP:
                    continue

                nav_label, priority = OBSTACLE_MAP[cls_name]

                # Normalised bbox [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                nx1, ny1 = x1 / w, y1 / h
                nx2, ny2 = x2 / w, y2 / h
                bbox_h_n = ny2 - ny1
                cx_n = (nx1 + nx2) / 2.0  # normalised centre x

                zone = _classify_zone(cx_n)
                proximity = _classify_proximity(bbox_h_n)

                det = Detection(
                    class_name=cls_name,
                    nav_label=nav_label,
                    priority=priority,
                    confidence=conf,
                    zone=zone,
                    proximity=proximity,
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_norm=(nx1, ny1, nx2, ny2),
                )
                result.detections.append(det)

        log.debug(
            "[detector] frame %d: %d detections in %.1f ms",
            frame_id,
            len(result.detections),
            inference_ms,
        )
        return result


# ── Spatial helpers ───────────────────────────────────────────────────────────


def _classify_zone(cx_normalised: float) -> Zone:
    if cx_normalised < ZONE_LEFT_EDGE:
        return Zone.LEFT
    if cx_normalised > ZONE_RIGHT_EDGE:
        return Zone.RIGHT
    return Zone.CENTRE


def _classify_proximity(bbox_height_normalised: float) -> Proximity:
    if bbox_height_normalised >= DIST_CLOSE_THRESH:
        return Proximity.CLOSE
    if bbox_height_normalised >= DIST_MEDIUM_THRESH:
        return Proximity.MEDIUM
    return Proximity.FAR


# ── COCO class → index lookup (for feature vector) ───────────────────────────
# Standard YOLOv8 COCO 80-class ordering.
_COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
_COCO_INDEX: dict[str, int] = {name: i for i, name in enumerate(_COCO_CLASSES)}
