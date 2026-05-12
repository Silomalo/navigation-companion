"""
audio_in.py — Microphone capture and speech-to-text using Whisper.

Runs two cooperating threads:
  1. capture_thread  — sounddevice InputStream fills a raw PCM ring buffer.
  2. stt_thread      — drains the ring buffer, runs Voice Activity Detection
                       (simple energy threshold), and when speech is detected
                       transcribes with OpenAI Whisper (local, no internet).

Callers receive completed utterances from an output queue:
    mic = Microphone()
    mic.start()
    utterance = mic.get_utterance()  # blocks until speech recognised
    mic.stop()

Simulation mode:
    When sounddevice is unavailable or fails to open, the capture thread
    falls back to keyboard input so the pipeline can be tested at a terminal.
    A real USB/Bluetooth microphone is always preferred when detected.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Optional

import numpy as np

from config import (
    MIC_BLOCK_FRAMES,
    MIC_CHANNELS,
    MIC_DTYPE,
    MIC_SAMPLE_RATE,
    MIC_SILENCE_THRESH,
    WHISPER_MODEL,
)

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Collect this many seconds of audio before attempting a transcription.
_SPEECH_COLLECT_S: float = 2.5
# After _SPEECH_COLLECT_S of silence, consider the utterance finished.
_SILENCE_TIMEOUT_S: float = 1.2


class Microphone:
    """
    Microphone capture with Whisper STT.

    Thread-safe.  start() / stop() are idempotent.
    """

    def __init__(self) -> None:
        self._utterance_queue: queue.Queue[str] = queue.Queue()
        self._pcm_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._whisper_model = None  # lazy load

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start capture and STT threads.

        Always attempts the real microphone via sounddevice first — this works
        on both macOS (desktop testing) and Raspberry Pi (production).
        Keyboard fallback is only used when sounddevice is unavailable or
        fails to open a stream.
        """
        self._stop_event.clear()

        t_cap = threading.Thread(
            target=self._capture_loop, name="mic-capture", daemon=True
        )
        t_stt = threading.Thread(target=self._stt_loop, name="mic-stt", daemon=True)
        self._threads = [t_cap, t_stt]

        for t in self._threads:
            t.start()

        log.info("[mic] started")

    def stop(self) -> None:
        """Signal threads to exit and wait."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3.0)
        log.info("[mic] stopped")

    def get_utterance(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Block until a recognised utterance is available and return it.
        Returns None if timeout expires or microphone is stopped.
        """
        try:
            return self._utterance_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def pending_utterance(self) -> Optional[str]:
        """Non-blocking poll — returns utterance or None."""
        try:
            return self._utterance_queue.get_nowait()
        except queue.Empty:
            return None

    # ── hardware capture thread ───────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Read PCM blocks from the microphone into _pcm_queue."""
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            log.warning(
                "[mic] sounddevice not installed — falling back to keyboard input"
            )
            self._keyboard_loop()
            return

        try:
            with sd.InputStream(
                samplerate=MIC_SAMPLE_RATE,
                channels=MIC_CHANNELS,
                dtype=MIC_DTYPE,
                blocksize=MIC_BLOCK_FRAMES,
            ) as stream:
                log.info(
                    "[mic] sounddevice stream open  %d Hz mono",
                    MIC_SAMPLE_RATE,
                )
                while not self._stop_event.is_set():
                    block, _overflowed = stream.read(MIC_BLOCK_FRAMES)
                    # block shape: (MIC_BLOCK_FRAMES, channels) — flatten to 1D
                    mono = block[:, 0] if block.ndim > 1 else block
                    try:
                        self._pcm_queue.put_nowait(mono.copy())
                    except queue.Full:
                        pass  # drop oldest implicitly (STT is slower than capture)

        except Exception as exc:  # noqa: BLE001
            log.warning("[mic] capture failed (%s) — falling back to keyboard", exc)
            self._keyboard_loop()

    # ── STT thread ────────────────────────────────────────────────────────

    def _stt_loop(self) -> None:
        """
        Consume PCM blocks, detect speech with VAD, transcribe with
        faster-whisper (CTranslate2 + int8 — ~4× faster than openai-whisper).
        """
        from faster_whisper import WhisperModel  # type: ignore

        log.info("[mic] loading faster-whisper '%s' (int8 CPU) …", WHISPER_MODEL)
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        log.info("[mic] faster-whisper ready")

        collecting: list[np.ndarray] = []
        last_voice_time: float = 0.0
        in_speech: bool = False

        while not self._stop_event.is_set():
            try:
                block = self._pcm_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = _rms(block)

            if rms > MIC_SILENCE_THRESH:
                # Voice activity detected.
                if not in_speech:
                    log.debug("[mic] speech start (rms=%.0f)", rms)
                    in_speech = True
                last_voice_time = time.monotonic()
                collecting.append(block)
            else:
                # Silence.
                if in_speech:
                    collecting.append(block)  # include trailing silence
                    elapsed = time.monotonic() - last_voice_time
                    if elapsed >= _SILENCE_TIMEOUT_S:
                        # End of utterance detected — transcribe.
                        audio = np.concatenate(collecting).astype(np.float32)
                        audio /= 32768.0  # normalise i16 → float32 [-1, 1]
                        collecting.clear()
                        in_speech = False

                        log.debug(
                            "[mic] transcribing %.1f s of audio …",
                            len(audio) / MIC_SAMPLE_RATE,
                        )
                        # faster-whisper returns a generator of segments.
                        segments, _ = model.transcribe(
                            audio,
                            language="en",
                            beam_size=5,
                        )
                        text: str = " ".join(s.text for s in segments).strip()
                        if text:
                            log.info("[mic] recognised: '%s'", text)
                            self._utterance_queue.put(text)

    # ── simulation: keyboard input ────────────────────────────────────────

    def _keyboard_loop(self) -> None:
        """
        In simulation mode, read lines from stdin and treat them as
        recognised speech commands.
        """
        print("[mic-sim] Type a voice command and press Enter.")
        print("          Examples:  'where am I'   'describe scene'   'stop'")
        while not self._stop_event.is_set():
            try:
                line = input("> ").strip()
                if line:
                    self._utterance_queue.put(line)
            except (EOFError, KeyboardInterrupt):
                break


# ── helpers ───────────────────────────────────────────────────────────────────


def _rms(block: np.ndarray) -> float:
    """Root-mean-square amplitude of a PCM block."""
    return float(math.sqrt(np.mean(block.astype(np.float64) ** 2)))
