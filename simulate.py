"""
simulate.py — Visual simulation of the Navigation Assistant.

Runs the complete detection + navigation pipeline with a rich OpenCV
visualisation window.  No Raspberry Pi hardware needed — just a webcam
or a video file.

Features:
  - Live YOLOv8n bounding boxes with colour-coded zones
  - Obstacle proximity bars on the side panel
  - Topological map node counter
  - Live TTS speech transcript overlay
  - Keyboard command injection (simulates voice input)
  - Frame-rate display and inference time graph

Usage:
    # Webcam (default camera index 0):
    SIMULATE=1 python simulate.py

    # Specific webcam index:
    SIMULATE=1 python simulate.py --camera 1

    # Video file:
    SIMULATE=1 python simulate.py --video path/to/video.mp4

    # Without a camera (pure synthetic frames):
    SIMULATE=1 python simulate.py --synthetic

Keyboard shortcuts while the window is open:
    D — "describe scene"
    W — "where am I"
    H — help
    Q — quit
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Ensure src/ is importable ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("SIMULATE", "1")

from audio_in import Microphone
from audio_out import Speaker
from config import CAMERA_HEIGHT, CAMERA_WIDTH, DETECT_FPS
from detector import Detection, DetectionResult, Detector, Proximity, Zone
from navigator import Navigator
from topo_map import TopoMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
COLOUR_URGENT = (0, 0, 255)  # red
COLOUR_NORMAL = (0, 165, 255)  # orange
COLOUR_INFO = (0, 255, 255)  # yellow
COLOUR_CLEAR = (0, 200, 80)  # green
COLOUR_LEFT = (255, 100, 0)  # blue
COLOUR_RIGHT = (0, 100, 255)  # red-orange
COLOUR_CENTRE = (0, 0, 220)  # deep red
COLOUR_TEXT = (240, 240, 240)
COLOUR_PANEL = (30, 30, 30)


def bbox_colour(det: Detection) -> tuple[int, int, int]:
    if det.is_urgent:
        return COLOUR_URGENT
    if det.priority == 2:
        return COLOUR_NORMAL
    return COLOUR_INFO


# ── Visualisation ─────────────────────────────────────────────────────────────


def draw_detections(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    h, w = frame.shape[:2]

    # Zone divider lines.
    from config import ZONE_LEFT_EDGE, ZONE_RIGHT_EDGE

    cv2.line(
        frame,
        (int(w * ZONE_LEFT_EDGE), 0),
        (int(w * ZONE_LEFT_EDGE), h),
        (80, 80, 80),
        1,
    )
    cv2.line(
        frame,
        (int(w * ZONE_RIGHT_EDGE), 0),
        (int(w * ZONE_RIGHT_EDGE), h),
        (80, 80, 80),
        1,
    )

    # Zone labels.
    cv2.putText(
        frame, "LEFT", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1
    )
    cv2.putText(
        frame,
        "AHEAD",
        (int(w * 0.44), 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 120, 120),
        1,
    )
    cv2.putText(
        frame,
        "RIGHT",
        (int(w * 0.7), 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 120, 120),
        1,
    )

    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        colour = bbox_colour(det)
        thickness = 3 if det.is_urgent else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        label = f"{det.nav_label} {det.confidence:.0%}"
        if det.proximity == Proximity.CLOSE:
            label = f"!! {label} !!"

        # Label background.
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    return frame


def draw_side_panel(
    canvas: np.ndarray,
    result: DetectionResult,
    fps: float,
    inf_ms: float,
    topo_stats: dict,
    last_speech: str,
    speech_log: deque,
) -> None:
    """Render the right-side info panel."""
    h, w = canvas.shape[:2]
    pw = 280  # panel width
    px = w - pw  # panel x start

    # Background.
    cv2.rectangle(canvas, (px, 0), (w, h), COLOUR_PANEL, -1)
    cv2.line(canvas, (px, 0), (px, h), (80, 80, 80), 1)

    y = 25
    dy = 22

    def txt(
        text: str, colour=COLOUR_TEXT, scale: float = 0.48, bold: bool = False
    ) -> None:
        nonlocal y
        cv2.putText(
            canvas,
            text,
            (px + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            2 if bold else 1,
            cv2.LINE_AA,
        )
        y += dy

    # Header.
    txt("NAV ASSISTANT SIM", COLOUR_CLEAR, scale=0.52, bold=True)
    txt(f"FPS: {fps:.1f}   Inf: {inf_ms:.0f}ms")
    txt(f"Frame: {result.frame_id}", scale=0.42)
    y += 6

    # Topo map.
    txt("─── TOPOLOGICAL MAP ───", (160, 160, 160), scale=0.43)
    txt(f"Nodes: {topo_stats['nodes']}   Edges: {topo_stats['edges']}")
    cur = topo_stats.get("current")
    txt(f"Current: {f'Node {cur}' if cur is not None else 'unknown'}", scale=0.42)
    y += 6

    # Detections.
    txt("─── DETECTIONS ───", (160, 160, 160), scale=0.43)
    if not result.detections:
        txt("Path clear", COLOUR_CLEAR)
    else:
        for det in result.by_priority[:6]:
            c = bbox_colour(det)
            txt(
                f"{det.nav_label[:14]:<14} {det.zone.value[:6]:<6} {det.proximity.value[:10]}",
                c,
                scale=0.42,
            )
    y += 6

    # Speech log.
    txt("─── SPEECH LOG ───", (160, 160, 160), scale=0.43)
    for line in list(speech_log)[-6:]:
        txt(line[:35], COLOUR_TEXT, scale=0.40)
    y += 6

    # Keyboard shortcuts.
    txt("─── KEYS ───", (160, 160, 160), scale=0.43)
    txt("[D] Describe  [W] Where am I", scale=0.40)
    txt("[H] Help      [Q] Quit", scale=0.40)


# ── Main simulation loop ──────────────────────────────────────────────────────


def run_simulation(
    video_source: int | str,
    use_synthetic: bool,
) -> None:

    # ── Subsystems ────────────────────────────────────────────────────────
    detector = Detector()
    log.info("Loading YOLOv8n …")
    detector.load()

    topo = TopoMap()
    nav = Navigator(topo_map=topo)
    speaker = Speaker()
    mic = Microphone()

    speaker.start()
    mic.start()
    speaker.say("Simulation started")

    # ── Video capture ─────────────────────────────────────────────────────
    if use_synthetic:
        cap = None
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            log.error(
                "Cannot open video source: %s — switching to synthetic", video_source
            )
            cap = None

    # ── State ─────────────────────────────────────────────────────────────
    frame_id = 0
    fps_window: deque[float] = deque(maxlen=30)
    speech_log: deque[str] = deque(maxlen=20)
    last_speech = ""
    synth_t = 0

    log.info("Simulation window open.  Press Q to quit.")
    cv2.namedWindow("Navigation Assistant — Simulation", cv2.WINDOW_NORMAL)

    while True:
        t0 = time.perf_counter()

        # ── Grab frame ────────────────────────────────────────────────
        if cap is not None:
            ret, raw_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                continue
            frame = cv2.resize(raw_frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        else:
            frame = _synthetic_frame(CAMERA_WIDTH, CAMERA_HEIGHT, synth_t)
            synth_t += 3

        # ── Detect ────────────────────────────────────────────────────
        result = detector.detect(frame.copy(), frame_id=frame_id)
        frame_id += 1

        # ── Navigate ──────────────────────────────────────────────────
        nav.update(result)
        for text, urgent in nav.pending_speech():
            speaker.say(text, urgent=urgent)
            speech_log.append(("!" if urgent else " ") + " " + text[:60])
            last_speech = text

        # ── Voice command (from keyboard simulation) ───────────────────
        cmd = mic.pending_utterance()
        if cmd:
            nav.handle_command(cmd)
            speech_log.append(f"[CMD] {cmd}")
            for text, urgent in nav.pending_speech():
                speaker.say(text, urgent=urgent)
                speech_log.append(("!" if urgent else " ") + " " + text[:60])

        # ── Draw ──────────────────────────────────────────────────────
        display = frame.copy()
        draw_detections(display, result)

        # Expand canvas to fit side panel.
        panel_w = 280
        canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH + panel_w, 3), dtype=np.uint8)
        canvas[:, :CAMERA_WIDTH] = display

        fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
        fps_window.append(fps)
        avg_fps = sum(fps_window) / len(fps_window)

        draw_side_panel(
            canvas,
            result,
            avg_fps,
            result.inference_ms,
            topo.get_stats(),
            last_speech,
            speech_log,
        )

        cv2.imshow("Navigation Assistant — Simulation", canvas)

        # ── Keyboard input ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("d"):
            mic._utterance_queue.put("describe scene")
        elif key == ord("w"):
            mic._utterance_queue.put("where am I")
        elif key == ord("h"):
            mic._utterance_queue.put("help")

        # ── Rate limit ────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        sleep_t = (1.0 / DETECT_FPS) - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Cleanup ───────────────────────────────────────────────────────────
    speaker.say("Simulation ended.")
    time.sleep(1.2)
    speaker.stop()
    mic.stop()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    log.info("Simulation stopped.")


def _synthetic_frame(w: int, h: int, t: int) -> np.ndarray:
    """Animated HSV gradient with a mock street scene overlay."""
    hue = np.full((h, w), t % 180, dtype=np.uint8)
    sat = np.full_like(hue, 80)
    val = np.full_like(hue, 190)
    bgr = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)
    cv2.putText(
        bgr,
        "NO CAMERA — SYNTHETIC FRAME",
        (30, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return bgr


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Navigation Assistant visual simulator"
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--camera", type=int, default=0, metavar="N", help="Webcam index (default: 0)"
    )
    src_group.add_argument(
        "--video", type=str, default=None, metavar="FILE", help="Video file path"
    )
    src_group.add_argument(
        "--synthetic", action="store_true", help="No camera — use synthetic frames"
    )
    args = parser.parse_args()

    source: int | str = args.video if args.video else args.camera
    run_simulation(video_source=source, use_synthetic=args.synthetic)
