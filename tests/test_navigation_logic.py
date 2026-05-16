import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detector import (
    Detection,
    DetectionResult,
    Proximity,
    Zone,
    _classify_proximity,
    _classify_zone,
)
from navigator import Navigator


class FakeTopoMap:
    def __init__(self) -> None:
        self.observed = 0

    def observe(self, result: DetectionResult) -> None:
        self.observed += 1

    def describe_current_location(self) -> str:
        return "You are at Location 1."


def make_detection(
    label: str = "person",
    priority: int = 1,
    zone: Zone = Zone.CENTRE,
    proximity: Proximity = Proximity.CLOSE,
) -> Detection:
    return Detection(
        class_name=label,
        nav_label=label,
        priority=priority,
        confidence=0.9,
        zone=zone,
        proximity=proximity,
        bbox_xyxy=(0.0, 0.0, 100.0, 100.0),
        bbox_norm=(0.0, 0.0, 0.5, 0.5),
    )


class DetectorClassificationTests(unittest.TestCase):
    def test_classifies_zones_from_normalised_center(self) -> None:
        self.assertEqual(_classify_zone(0.1), Zone.LEFT)
        self.assertEqual(_classify_zone(0.5), Zone.CENTRE)
        self.assertEqual(_classify_zone(0.9), Zone.RIGHT)

    def test_classifies_proximity_from_box_height(self) -> None:
        self.assertEqual(_classify_proximity(0.6), Proximity.CLOSE)
        self.assertEqual(_classify_proximity(0.4), Proximity.MEDIUM)
        self.assertEqual(_classify_proximity(0.1), Proximity.FAR)


class NavigatorTests(unittest.TestCase):
    def test_urgent_center_obstacle_is_queued(self) -> None:
        nav = Navigator(FakeTopoMap())
        result = DetectionResult(
            frame_id=1,
            timestamp=time.time(),
            detections=[make_detection()],
        )

        nav.update(result)

        self.assertEqual(nav.pending_speech(), [("Warning! person ahead", True)])

    def test_where_command_uses_topological_location(self) -> None:
        nav = Navigator(FakeTopoMap())

        nav.handle_command("where am I?")

        self.assertEqual(nav.pending_speech(), [("You are at Location 1.", False)])

    def test_describe_command_summarises_scene(self) -> None:
        nav = Navigator(FakeTopoMap())
        result = DetectionResult(
            frame_id=1,
            timestamp=time.time(),
            detections=[make_detection(label="person")],
        )
        nav.update(result)
        nav.pending_speech()

        nav.handle_command("describe the scene")
        speech = nav.pending_speech()

        self.assertEqual(len(speech), 1)
        self.assertIn("I can see 1 object", speech[0][0])
        self.assertFalse(speech[0][1])


if __name__ == "__main__":
    unittest.main()
