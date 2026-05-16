import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detector import Detection, DetectionResult, Proximity, Zone
from topo_map import TopoMap


def result_for(class_name: str, confidence: float = 0.9) -> DetectionResult:
    return DetectionResult(
        frame_id=1,
        timestamp=time.time(),
        detections=[
            Detection(
                class_name=class_name,
                nav_label=class_name,
                priority=1,
                confidence=confidence,
                zone=Zone.CENTRE,
                proximity=Proximity.MEDIUM,
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                bbox_norm=(0.0, 0.0, 0.1, 0.1),
            )
        ],
    )


class TopoMapTests(unittest.TestCase):
    def make_map(self, path: Path) -> TopoMap:
        topo = TopoMap()
        topo._nodes.clear()
        topo._edges.clear()
        topo._next_id = 0
        topo._current_node_id = None
        topo._last_record_t = 0.0
        topo._map_file = path
        return topo

    def test_creates_nodes_and_edges_for_new_places(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            topo = self.make_map(Path(tmp) / "routes.json")

            topo.observe(result_for("person"))
            topo._last_record_t = 0.0
            topo.observe(result_for("car"))

            stats = topo.get_stats()
            self.assertEqual(stats["nodes"], 2)
            self.assertEqual(stats["edges"], 1)
            self.assertEqual(stats["current"], 1)

    def test_empty_observation_does_not_consume_record_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            topo = self.make_map(Path(tmp) / "routes.json")
            empty = DetectionResult(frame_id=1, timestamp=time.time())

            topo.observe(empty)
            topo.observe(result_for("person"))

            self.assertEqual(topo.get_stats()["nodes"], 1)


if __name__ == "__main__":
    unittest.main()
