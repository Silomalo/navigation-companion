"""
navigator.py — Navigation logic, command parser, and feedback generator.

Responsibilities:
  1. Receive a DetectionResult every DETECT_FPS and decide what to say.
  2. Prioritise:  urgent obstacles > normal obstacles > path-clear confirmation.
  3. Rate-limit speech to FEEDBACK_MIN_INTERVAL_S to prevent audio fatigue.
  4. Parse voice commands from the microphone and respond.
  5. Feed location observations to the topological map.

The Navigator does NOT touch hardware directly — it receives data from
detectors / microphone and emits strings to the Speaker.  This keeps it
fully unit-testable.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from config import FEEDBACK_MIN_INTERVAL_S
from detector import Detection, DetectionResult, Proximity, Zone
from topo_map import TopoMap

log = logging.getLogger(__name__)

# ── Voice command keywords ────────────────────────────────────────────────────
_CMD_DESCRIBE = {"describe", "scene", "what", "see", "around", "surroundings"}
_CMD_WHERE = {"where", "location", "am", "route", "place"}
_CMD_STOP = {"stop", "quit", "exit", "off", "silence"}
_CMD_HELP = {"help", "commands", "what can you do"}


class Navigator:
    """
    Core navigation logic.

    Usage:
        nav = Navigator(topo_map)
        nav.update(detection_result)       # called each frame
        phrase = nav.pending_speech()      # poll for text to speak
        nav.handle_command("where am I")   # inject voice command
    """

    def __init__(self, topo_map: TopoMap) -> None:
        self._topo = topo_map
        self._last_speech_t: float = 0.0
        self._last_urgent_t: float = 0.0
        self._last_command_t: float = 0.0  # suppress urgent after a command
        self._speech_queue: list[tuple[str, bool]] = []  # (text, urgent)
        self._consecutive_clear: int = 0
        self._last_result: Optional[DetectionResult] = None

    # ── Main update — called every detection frame ────────────────────────

    def update(self, result: DetectionResult) -> None:
        """Process a DetectionResult and queue navigation instructions."""
        self._last_result = result
        now = time.monotonic()

        # ── 1. Update topological map ─────────────────────────────────
        self._topo.observe(result)

        # ── 2. Urgent obstacles (close + centre) ──────────────────────
        urgent = result.urgent
        if urgent:
            # Hold off if a command response was just queued (give it 3 s to
            # be heard before resuming obstacle warnings).
            if now - self._last_command_t < 3.0:
                return
            # De-dup: only speak if not repeated within 1.5 s.
            if now - self._last_urgent_t >= 1.5:
                phrase = _build_urgent_phrase(urgent)
                self._enqueue(phrase, urgent=True)
                self._last_urgent_t = now
            return  # urgent pre-empts normal flow

        # ── 3. Normal obstacle feedback (rate-limited) ─────────────────
        if now - self._last_speech_t >= FEEDBACK_MIN_INTERVAL_S:
            if result.detections:
                phrase = _build_normal_phrase(result)
                self._enqueue(phrase, urgent=False)
            else:
                # Path clear — confirm occasionally, not every frame.
                self._consecutive_clear += 1
                if self._consecutive_clear % 30 == 0:  # ~every 3 s at 10 fps
                    self._enqueue("Path clear", urgent=False)

    # ── Voice command handler ─────────────────────────────────────────────

    def handle_command(self, text: str) -> None:
        """
        Parse a spoken command and generate an appropriate response.

        Args:
            text: Whisper-transcribed utterance, e.g. "describe the scene".
        """
        # Strip punctuation so Whisper artefacts like "help." match "help".
        clean = re.sub(r"[^\w\s]", "", text.lower())
        words = set(clean.split())
        log.info("[nav] command received: '%s'", text)

        if words & _CMD_STOP:
            self._enqueue("Stopping navigation", urgent=True)

        elif words & _CMD_WHERE:
            desc = self._topo.describe_current_location()
            self._enqueue(desc or "Location not yet learned", urgent=False)

        elif words & _CMD_DESCRIBE:
            if self._last_result and self._last_result.detections:
                phrase = _full_scene_description(self._last_result)
            else:
                phrase = "No obstacles detected. The path appears clear."
            self._enqueue(phrase, urgent=False)

        elif words & _CMD_HELP:
            self._enqueue("Say: describe scene, where am I, or stop.", urgent=False)

        else:
            # Pass through to LLM / future intent parser.
            log.info("[nav] unrecognised command: '%s'", text)
            self._enqueue(f"I heard: {text}. Say help for commands.", urgent=False)

        # Suppress urgent obstacle warnings for 3 s so the response is heard.
        self._last_command_t = time.monotonic()

    # ── Speech queue ──────────────────────────────────────────────────────

    def pending_speech(self) -> list[tuple[str, bool]]:
        """
        Drain and return all queued speech requests since last call.
        Returns list of (text, urgent) tuples.
        """
        items = self._speech_queue.copy()
        self._speech_queue.clear()
        if items:
            self._last_speech_t = time.monotonic()
        return items

    # ── private ───────────────────────────────────────────────────────────

    def _enqueue(self, text: str, urgent: bool) -> None:
        if text:
            self._speech_queue.append((text, urgent))
            log.debug("[nav] queued speech: '%s' (urgent=%s)", text, urgent)


# ── Phrase builders ───────────────────────────────────────────────────────────


def _build_urgent_phrase(detections: list[Detection]) -> str:
    """Short, urgent phrase for very-close obstacles."""
    # Lead with the highest-priority detection.
    top = detections[0]
    parts = [f"Warning! {top.nav_label} {top.zone.value}"]
    if len(detections) > 1:
        others = ", ".join(d.nav_label for d in detections[1:3])
        parts.append(f"also {others}")
    return ". ".join(parts)


def _build_normal_phrase(result: DetectionResult) -> str:
    """
    Summarise the scene in 1–2 sentences.
    Prioritises centre obstacles, then mentions flanks.
    """
    centre = [d for d in result.by_priority if d.zone == Zone.CENTRE]
    left = [d for d in result.by_priority if d.zone == Zone.LEFT]
    right = [d for d in result.by_priority if d.zone == Zone.RIGHT]

    parts: list[str] = []

    if centre:
        top = centre[0]
        parts.append(f"{top.nav_label} ahead, {top.proximity.value}")

    if left:
        parts.append(f"{left[0].nav_label} on the left")

    if right:
        parts.append(f"{right[0].nav_label} on the right")

    return ". ".join(parts) if parts else "Path clear"


def _full_scene_description(result: DetectionResult) -> str:
    """
    Verbose scene description for the 'describe scene' command.
    """
    if not result.detections:
        return "No obstacles detected. Path is clear."

    counts: dict[str, int] = {}
    for d in result.detections:
        counts[d.nav_label] = counts.get(d.nav_label, 0) + 1

    items = []
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        items.append(f"{count} {label}{'s' if count > 1 else ''}")

    total = len(result.detections)
    summary = f"I can see {total} object{'s' if total > 1 else ''}: {', '.join(items)}."

    centre_obs = [d for d in result.by_priority if d.zone == Zone.CENTRE]
    if centre_obs:
        summary += f" The path ahead has {centre_obs[0].nav_label} {centre_obs[0].proximity.value}."
    else:
        summary += " The path ahead is clear."

    return summary
