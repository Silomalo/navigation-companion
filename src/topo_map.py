"""
topo_map.py — Topological route learning and recall.

A topological map represents the user's environment as a graph of
*nodes* (recognisable places) connected by *edges* (traversals between
places).  This is how a visually impaired person mentally structures
familiar routes — not GPS coordinates, but sequences of landmarks.

Implementation:
  - Each node stores a feature vector (80-dim class-confidence histogram
    from the detector) that fingerprints what the camera sees at that spot.
  - When the current frame's feature vector is similar to a stored node
    (cosine similarity ≥ TOPO_SIMILARITY_THRESH), the user is considered
    to have "returned" to that place.
  - New nodes are appended when the similarity to ALL stored nodes is below
    the threshold.
  - Edges are created between consecutive nodes to capture transition order.
  - The map is persisted to a JSON file and reloaded on next run.

Limitations / future work:
  - Add GPS coordinates when a GPS hat is attached (merge with IMU heading).
  - Use a sequence-to-sequence model for richer place descriptions.
  - Add a compass heading so edges encode direction (north, south, etc.).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    TOPO_MAP_FILE,
    TOPO_RECORD_INTERVAL_S,
    TOPO_SIMILARITY_THRESH,
)
from detector import DetectionResult

log = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TopoNode:
    node_id:      int
    label:        str                  # auto-generated or user-named
    feature:      list[float]          # 80-dim class histogram
    first_seen:   float                # Unix timestamp
    visit_count:  int = 1
    neighbours:   list[int] = field(default_factory=list)  # connected node IDs

    @property
    def feature_array(self) -> np.ndarray:
        return np.array(self.feature, dtype=np.float32)


@dataclass
class TopoEdge:
    from_id:   int
    to_id:     int
    count:     int = 1   # traversal frequency


# ── Topological map ───────────────────────────────────────────────────────────

class TopoMap:
    """
    Learns and recalls the user's regular routes.

    Thread-safe: all public methods acquire a simple lock.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, TopoNode] = {}
        self._edges: dict[tuple[int, int], TopoEdge] = {}
        self._next_id: int = 0
        self._current_node_id: Optional[int] = None
        self._last_record_t: float = 0.0
        self._map_file: Path = TOPO_MAP_FILE
        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def observe(self, result: DetectionResult) -> None:
        """
        Update the map with a new observation.
        Called every detection frame; rate-limited internally.
        """
        now = time.monotonic()
        if now - self._last_record_t < TOPO_RECORD_INTERVAL_S:
            return
        self._last_record_t = now

        vec = result.feature_vector()
        # Only record if there is something meaningful to fingerprint.
        if np.sum(vec) < 0.01:
            return

        matched_id, sim = self._find_similar(vec)

        if matched_id is not None and sim >= TOPO_SIMILARITY_THRESH:
            # Revisiting a known place.
            node = self._nodes[matched_id]
            node.visit_count += 1
            # Update feature with exponential moving average.
            alpha = 0.1
            node.feature = (
                ((1 - alpha) * node.feature_array + alpha * vec).tolist()
            )
            prev_id, self._current_node_id = self._current_node_id, matched_id
        else:
            # New place — create node.
            node = TopoNode(
                node_id=self._next_id,
                label=f"Location {self._next_id}",
                feature=vec.tolist(),
                first_seen=time.time(),
            )
            self._nodes[self._next_id] = node
            prev_id, self._current_node_id = self._current_node_id, self._next_id
            self._next_id += 1
            log.info("[topo] new location: %s (total=%d)", node.label, len(self._nodes))

        # Record edge between previous and current node.
        if prev_id is not None and prev_id != self._current_node_id:
            self._add_edge(prev_id, self._current_node_id)

        self._save()

    def describe_current_location(self) -> Optional[str]:
        """
        Return a human-readable description of the current location.
        Includes how many times the user has been here and what neighbours exist.
        """
        if self._current_node_id is None:
            return None
        node = self._nodes.get(self._current_node_id)
        if node is None:
            return None

        desc = f"You are at {node.label}."
        desc += f" You have been here {node.visit_count} time{'s' if node.visit_count > 1 else ''}."

        if node.neighbours:
            n_labels = [
                self._nodes[nid].label
                for nid in node.neighbours
                if nid in self._nodes
            ]
            if n_labels:
                desc += f" Connected to: {', '.join(n_labels[:3])}."

        return desc

    def get_stats(self) -> dict:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "current": self._current_node_id,
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "next_id": self._next_id,
            "nodes": {str(k): asdict(v) for k, v in self._nodes.items()},
            "edges": [
                {"from_id": e.from_id, "to_id": e.to_id, "count": e.count}
                for e in self._edges.values()
            ],
        }
        try:
            self._map_file.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            log.warning("[topo] could not save map: %s", exc)

    def _load(self) -> None:
        if not self._map_file.exists():
            log.info("[topo] no existing map — starting fresh")
            return
        try:
            data = json.loads(self._map_file.read_text())
            self._next_id = data.get("next_id", 0)
            for k, v in data.get("nodes", {}).items():
                node = TopoNode(**v)
                self._nodes[node.node_id] = node
            for e in data.get("edges", []):
                key = (e["from_id"], e["to_id"])
                self._edges[key] = TopoEdge(**e)
            log.info(
                "[topo] loaded map: %d nodes, %d edges",
                len(self._nodes), len(self._edges),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[topo] map load failed: %s — starting fresh", exc)

    # ── Internals ─────────────────────────────────────────────────────────

    def _find_similar(
        self, vec: np.ndarray
    ) -> tuple[Optional[int], float]:
        """Return (node_id, similarity) of the most similar stored node."""
        best_id:  Optional[int] = None
        best_sim: float = -1.0
        for node in self._nodes.values():
            sim = _cosine_similarity(vec, node.feature_array)
            if sim > best_sim:
                best_sim = sim
                best_id  = node.node_id
        return best_id, best_sim

    def _add_edge(self, from_id: int, to_id: int) -> None:
        key = (from_id, to_id)
        if key in self._edges:
            self._edges[key].count += 1
        else:
            self._edges[key] = TopoEdge(from_id=from_id, to_id=to_id)
        # Keep neighbour lists in sync.
        node = self._nodes.get(from_id)
        if node and to_id not in node.neighbours:
            node.neighbours.append(to_id)


# ── Maths ─────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
