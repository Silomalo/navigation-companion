"""
camera.py — Camera capture with automatic hardware/simulation selection.

Hardware mode  (SIMULATE=0):
    Uses picamera2 (the official RPi camera stack for Bullseye/Bookworm).
    Outputs BGR NumPy arrays for OpenCV / YOLO compatibility.

Simulation mode (SIMULATE=1, or picamera2 unavailable):
    Falls back to OpenCV VideoCapture so the full pipeline can be developed
    and tested on any Linux/macOS/Windows machine with a webcam or video file.

Both modes expose the same interface:
    camera = Camera()
    camera.start()
    frame = camera.read()   # returns (H, W, 3) BGR uint8 ndarray or None
    camera.stop()
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import (
    CAMERA_FPS,
    CAMERA_FORMAT,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    SIM_CAMERA_INDEX,
    SIMULATE,
)

log = logging.getLogger(__name__)


# ── Public interface ──────────────────────────────────────────────────────────

class Camera:
    """
    Thread-safe camera wrapper.

    Capture runs in a background daemon thread and always holds the
    most recent frame.  Callers poll via read() at whatever rate they
    like without blocking on hardware I/O.
    """

    def __init__(self) -> None:
        self._frame:  Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend: str = ""
        self._error: Optional[BaseException] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the capture thread.  Returns immediately."""
        self._stop_event.clear()
        self._error = None
        if SIMULATE:
            self._thread = threading.Thread(
                target=self._sim_loop, name="camera-sim", daemon=True
            )
            self._backend = "OpenCV-simulation"
        else:
            self._thread = threading.Thread(
                target=self._hw_loop, name="camera-hw", daemon=True
            )
            self._backend = "picamera2"

        self._thread.start()
        # Wait up to 3 s for the first frame so callers don't get None immediately.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._frame is not None:
                break
            if self._error is not None:
                break
            time.sleep(0.05)

        if self._error is not None and not SIMULATE:
            raise RuntimeError("Camera failed to start in hardware mode") from self._error

        if self._frame is None:
            log.warning("[camera] no frame received within 3 s — check hardware")
        else:
            log.info("[camera] started (%s)  %dx%d", self._backend, CAMERA_WIDTH, CAMERA_HEIGHT)

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("[camera] stopped")

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame as (H, W, 3) BGR uint8, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def backend(self) -> str:
        return self._backend

    # ── hardware capture loop (picamera2) ─────────────────────────────────

    def _hw_loop(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError:
            self._error = RuntimeError("picamera2 is not installed")
            log.error("[camera] picamera2 not installed")
            return

        try:
            cam = Picamera2()
            config = cam.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": CAMERA_FORMAT},
                controls={"FrameRate": float(CAMERA_FPS)},
            )
            cam.configure(config)
            cam.start()
            log.info("[camera] picamera2 started")

            interval = 1.0 / CAMERA_FPS
            while not self._stop_event.is_set():
                # capture_array returns (H, W, 3) BGR uint8
                frame: np.ndarray = cam.capture_array()
                with self._lock:
                    self._frame = frame
                time.sleep(interval)

            cam.stop()

        except Exception as exc:  # noqa: BLE001
            self._error = exc
            log.error("[camera] hardware error: %s", exc)

    # ── simulation / webcam capture loop (OpenCV) ─────────────────────────

    def _sim_loop(self) -> None:
        cap = cv2.VideoCapture(SIM_CAMERA_INDEX)
        if not cap.isOpened():
            log.error(
                "[camera] cannot open VideoCapture(%d) — using synthetic frames",
                SIM_CAMERA_INDEX,
            )
            self._synthetic_loop()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

        log.info("[camera] OpenCV capture opened (index=%d)", SIM_CAMERA_INDEX)
        interval = 1.0 / CAMERA_FPS

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log.warning("[camera] VideoCapture.read() failed — retrying")
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(interval)

        cap.release()

    # ── last-resort: synthetic colour gradient frame ───────────────────────

    def _synthetic_loop(self) -> None:
        """Generate an animated gradient frame when no real camera is available."""
        log.info("[camera] using synthetic gradient frames for UI testing")
        t = 0
        interval = 1.0 / CAMERA_FPS

        while not self._stop_event.is_set():
            # Slowly shifting HSV gradient so the simulation is visually obvious.
            h = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), (t % 180), dtype=np.uint8)
            s = np.full_like(h, 200)
            v = np.full_like(h, 200)
            hsv = cv2.merge([h, s, v])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Overlay "SIMULATION" text.
            cv2.putText(
                bgr, "SIMULATION — no camera", (30, CAMERA_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
            )
            with self._lock:
                self._frame = bgr

            t += 2
            time.sleep(interval)
