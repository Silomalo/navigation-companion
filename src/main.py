"""
main.py — Navigation Assistant entry point.

Wires together all subsystems and runs the main perception-action loop:

    Camera → Detector → Navigator → Speaker
                  ↑                    ↑
             TopoMap            Microphone (voice commands)

Threading model:
  - Camera runs its own OS thread (hardware I/O).
  - Microphone runs its own OS thread (ALSA blocking reads).
  - Whisper STT runs inside the mic thread pool.
  - Speaker (TTS) runs its own OS thread.
  - The MAIN thread owns the detect → navigate loop at DETECT_FPS.

Run:
    # Hardware mode (on the Raspberry Pi 5):
    python src/main.py

    # Simulation mode (any machine with a webcam):
    SIMULATE=1 python src/main.py
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path

# ── Make src/ importable when running from project root ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from audio_in  import Microphone
from audio_out import Speaker
from camera    import Camera
from config    import DETECT_FPS, SIMULATE
from detector  import Detector
from navigator import Navigator
from topo_map  import TopoMap

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(threadName)-14s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=" * 60)
    log.info("  Visually Impaired Navigation Assistant")
    log.info("  Mode: %s", "SIMULATION" if SIMULATE else "HARDWARE")
    log.info("=" * 60)

    # ── Instantiate subsystems ────────────────────────────────────────────
    camera  = Camera()
    speaker = Speaker()
    mic     = Microphone(speech_active=lambda: speaker.is_speaking)
    topo    = TopoMap()
    detector = Detector()
    nav      = Navigator(topo_map=topo)

    # ── Graceful shutdown on SIGINT / SIGTERM ─────────────────────────────
    running = True

    def _shutdown(signum, frame):  # noqa: ANN001
        nonlocal running
        log.info("Shutdown signal received")
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    camera_started = False
    mic_started = False
    speaker_started = False

    try:
        # ── Start hardware threads ────────────────────────────────────────
        log.info("Loading YOLO model …")
        detector.load()

        log.info("Starting camera …")
        camera.start()
        camera_started = True

        log.info("Starting microphone …")
        mic.start()
        mic_started = True

        log.info("Starting speaker …")
        speaker.start()
        speaker_started = True

        speaker.say("Navigation assistant ready.")
        log.info("System ready.  Press Ctrl-C to stop.")

        # ── Main perception-action loop ───────────────────────────────────
        frame_id   = 0
        loop_interval = 1.0 / DETECT_FPS
        last_fps_log  = time.monotonic()
        frames_since_log = 0

        while running:
            t_start = time.monotonic()

            # ── 1. Grab latest frame ──────────────────────────────────
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            # ── 2. Detect obstacles ───────────────────────────────────
            result = detector.detect(frame, frame_id=frame_id)
            frame_id += 1
            frames_since_log += 1

            # ── 3. Navigation logic ───────────────────────────────────
            nav.update(result)

            # ── 4. Speak pending instructions ─────────────────────────
            for text, urgent in nav.pending_speech():
                speaker.say(text, urgent=urgent)

            # ── 5. Process any voice commands ─────────────────────────
            cmd = mic.pending_utterance()
            if cmd:
                nav.handle_command(cmd)
                for text, urgent in nav.pending_speech():
                    speaker.say(text, urgent=urgent)

                # Handle stop command in main loop.
                if any(w in cmd.lower() for w in ("stop", "quit", "exit")):
                    running = False

            # ── 6. FPS telemetry ──────────────────────────────────────
            now = time.monotonic()
            if now - last_fps_log >= 10.0:
                fps = frames_since_log / (now - last_fps_log)
                log.info(
                    "Running at %.1f fps  |  topo: %s",
                    fps, topo.get_stats(),
                )
                last_fps_log = now
                frames_since_log = 0

            # ── 7. Sleep to maintain DETECT_FPS ──────────────────────
            elapsed = time.monotonic() - t_start
            sleep_t = loop_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        # ── Clean shutdown ────────────────────────────────────────────────
        log.info("Shutting down …")
        if speaker_started:
            speaker.say("Goodbye.")
            time.sleep(1.5)   # let TTS finish

        if camera_started:
            camera.stop()
        if mic_started:
            mic.stop()
        if speaker_started:
            speaker.stop()

        log.info("Navigation Assistant stopped.")


if __name__ == "__main__":
    main()
