"""
audio_out.py — Text-to-Speech output through the earphone.

Uses pyttsx3 which drives:
  - espeak / espeak-ng on Linux (installed by default on Raspberry Pi OS)
  - NSSpeechSynthesizer on macOS (for development / simulation)
  - SAPI5 on Windows (for development)

A background daemon thread serialises speech requests so the main loop
never blocks on audio I/O.  Requests are queued; urgent messages can
interrupt the current phrase.

    tts = Speaker()
    tts.start()
    tts.say("Obstacle ahead, person on the left")
    tts.say("Warning: step ahead", urgent=True)   # pre-empts queue
    tts.stop()

In simulation mode (SIMULATE=1) the text is also printed to stdout in
addition to being spoken, so the pipeline is auditable without hardware.
"""

from __future__ import annotations

import logging
import platform
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config import SIMULATE, TTS_RATE, TTS_VOICE, TTS_VOLUME

log = logging.getLogger(__name__)

# macOS: pyttsx3 uses NSSpeechSynthesizer which requires the AppKit main run
# loop — it hangs silently in a background thread. Use the built-in `say`
# command instead, which is thread-safe on macOS.
_IS_MACOS: bool = platform.system() == "Darwin"


@dataclass
class _SpeechRequest:
    text: str
    urgent: bool = False
    # Sentinel for clean shutdown.
    shutdown: bool = False


class Speaker:
    """
    Thread-safe TTS speaker.

    Maintains a single background thread that owns the pyttsx3 engine
    (the engine must not be used from multiple threads simultaneously).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[_SpeechRequest] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_text: str = ""

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()
        log.info("[tts] speaker started")

    def stop(self) -> None:
        # Send sentinel; the worker will stop any active subprocess speech.
        self._queue.put(_SpeechRequest(text="", shutdown=True))
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("[tts] speaker stopped")

    # ── public API ────────────────────────────────────────────────────────

    def say(self, text: str, urgent: bool = False) -> None:
        """
        Queue a speech request.

        Args:
            text:   Text to speak.
            urgent: If True, clears the current queue and speaks immediately.
        """
        if not text:
            return

        if SIMULATE:
            # Always print in simulation so developer can see what would be spoken.
            tag = "[URGENT] " if urgent else ""
            print(f"[TTS] {tag}{text}")

        if urgent:
            # Flush pending non-urgent requests.
            _drain_queue(self._queue)

        self._queue.put(_SpeechRequest(text=text, urgent=urgent))
        log.debug("[tts] queued: '%s' (urgent=%s)", text, urgent)

    def say_now(self, text: str) -> None:
        """Convenience: urgent speech, always pre-empts the queue."""
        self.say(text, urgent=True)

    @property
    def is_speaking(self) -> bool:
        return bool(self._current_text) or not self._queue.empty()

    # ── background TTS thread ─────────────────────────────────────────────

    def _run(self) -> None:
        if _IS_MACOS:
            self._subprocess_loop(self._macos_cmd)
        elif shutil.which("espeak-ng"):
            self._subprocess_loop(self._espeak_cmd)
        else:
            self._pyttsx3_loop()

    def _macos_cmd(self, text: str) -> list[str]:
        """Build a macOS `say` command."""
        if TTS_VOICE:
            return ["say", "-v", TTS_VOICE, "-r", str(TTS_RATE), text]
        return ["say", "-r", str(TTS_RATE), text]

    def _espeak_cmd(self, text: str) -> list[str]:
        """Build a Linux/RPi espeak-ng command."""
        cmd = ["espeak-ng", "-s", str(TTS_RATE), "-a", str(int(TTS_VOLUME * 200))]
        if TTS_VOICE:
            cmd.extend(["-v", TTS_VOICE])
        cmd.append(text)
        return cmd

    def _subprocess_loop(self, cmd_builder) -> None:  # noqa: ANN001
        """TTS loop that can interrupt active speech for urgent requests."""
        if _IS_MACOS:
            log.info("[tts] macOS — using 'say' command  rate=%d", TTS_RATE)
        else:
            log.info("[tts] Linux — using espeak-ng  rate=%d", TTS_RATE)

        req: Optional[_SpeechRequest] = None
        proc: Optional[subprocess.Popen] = None

        while True:
            if req is None:
                req = self._queue.get()

            if req.shutdown:
                if proc and proc.poll() is None:
                    _terminate_process(proc)
                break

            log.info("[tts] speaking: '%s'", req.text)
            self._current_text = req.text
            proc = subprocess.Popen(cmd_builder(req.text))

            interrupted = False
            while proc.poll() is None:
                try:
                    next_req = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if next_req.shutdown:
                    _terminate_process(proc)
                    self._current_text = ""
                    return

                if next_req.urgent:
                    _terminate_process(proc)
                    _drain_queue(self._queue)
                    req = next_req
                    interrupted = True
                    break

                # Preserve non-urgent messages behind the active phrase.
                self._queue.put(next_req)
                time.sleep(0.05)

            if not interrupted:
                req = None
            self._current_text = ""

    def _macos_say_loop(self) -> None:
        """Compatibility wrapper for older callers."""
        log.info("[tts] macOS — using 'say' command  rate=%d", TTS_RATE)
        self._subprocess_loop(self._macos_cmd)

    def _pyttsx3_loop(self) -> None:
        """Linux / RPi TTS via pyttsx3 + espeak."""
        try:
            import pyttsx3  # type: ignore
        except ImportError:
            log.warning("[tts] pyttsx3 not installed — speech will be printed only")
            self._print_only_loop()
            return

        try:
            engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001
            log.error("[tts] pyttsx3 init failed: %s — printing only", exc)
            self._print_only_loop()
            return

        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)

        if TTS_VOICE is not None:
            for voice in engine.getProperty("voices"):
                if TTS_VOICE.lower() in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    log.info("[tts] voice set to: %s", voice.name)
                    break
            else:
                log.warning("[tts] voice '%s' not found — using default", TTS_VOICE)

        log.info("[tts] pyttsx3 ready  rate=%d  volume=%.2f", TTS_RATE, TTS_VOLUME)

        while True:
            try:
                req = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if req.shutdown:
                break
            log.info("[tts] speaking: '%s'", req.text)
            self._current_text = req.text
            engine.say(req.text)
            engine.runAndWait()
            self._current_text = ""

    def _print_only_loop(self) -> None:
        """Fallback when pyttsx3 is unavailable — just drain and log."""
        while True:
            try:
                req = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if req.shutdown:
                break
            log.info("[tts-fallback] '%s'", req.text)
            print(f"[TTS] {req.text}")


# ── helpers ───────────────────────────────────────────────────────────────────


def _drain_queue(q: queue.Queue) -> None:
    """Remove all pending items from a queue without blocking."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _terminate_process(proc: subprocess.Popen) -> None:
    """Stop an active TTS subprocess without hanging shutdown."""
    proc.terminate()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=0.5)
