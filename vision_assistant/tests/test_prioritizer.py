"""
Tests for Prioritizer (legacy) and BlindAssistanceEngine compatibility.

Since BlindAssistanceEngine replaces Prioritizer, these tests verify
the new engine through the same .prioritize() interface.
"""

import time
import numpy as np
import pytest

from modules.blind_assistance import BlindAssistanceEngine
from utils.datatypes import DetectedObject, FrameData


@pytest.fixture
def ba_config():
    """Config for BlindAssistanceEngine."""
    return {
        "blind_assistance": {
            "dead_ahead_zone_pct": [0.35, 0.65],
            "critical_distance_m": 1.5,
            "warning_distance_m": 3.0,
            "fast_approach_velocity_m_per_s": 1.0,
            "scene_summary_interval_s": 15,
            "max_announcements_per_frame": 2,
            "staircase_bottom_frame_pct": 0.60,
            "cooldown": {
                "critical_s": 1.5,
                "warning_s": 3.0,
                "info_s": 6.0,
                "clear_s": 10.0,
                "safety_override": True,
            },
        },
        "detection": {"max_objects": 5},
        "audio": {},
    }


def _make_fd(objects):
    return FrameData(
        raw_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=0,
        timestamp=time.time(),
        objects=objects,
    )


class TestPrioritizerCompat:
    """Test that BlindAssistanceEngine works as a Prioritizer drop-in."""

    def test_prioritize_returns_frame_data(self, ba_config):
        """prioritize() must return a FrameData instance."""
        engine = BlindAssistanceEngine(ba_config)
        fd = _make_fd([])
        result = engine.prioritize(fd)
        assert isinstance(result, FrameData)

    def test_announcements_populated(self, ba_config):
        """Detected objects should produce announcements."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        fd = _make_fd([obj])
        result = engine.prioritize(fd)
        assert len(result.announcements) >= 1

    def test_objects_get_priority_score(self, ba_config):
        """Each object should get a priority_score."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="chair", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=2.0, smoothed_distance_m=2.0,
        )
        fd = _make_fd([obj])
        engine.prioritize(fd)
        assert fd.objects[0].priority_score is not None
        assert fd.objects[0].priority_score > 0

    def test_closer_object_has_higher_score(self, ba_config):
        """Closer object should have a higher priority score."""
        engine = BlindAssistanceEngine(ba_config)
        close = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        far = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=6.0, smoothed_distance_m=6.0,
        )
        fd = _make_fd([far, close])
        engine.prioritize(fd)
        assert fd.objects[0].priority_score > fd.objects[1].priority_score

    def test_path_is_clear(self, ba_config):
        """Empty frames should eventually produce 'Path is clear'."""
        engine = BlindAssistanceEngine(ba_config)
        engine._cooldown_tracker = {"__path_clear__": 0.0}
        engine._clear_frame_count = 10

        fd = _make_fd([])
        engine.prioritize(fd)

        assert any("clear" in a.lower() for a in fd.announcements)

    def test_announcement_format(self, ba_config):
        """Announcements should be non-empty strings."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="car", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        fd = _make_fd([obj])
        engine.prioritize(fd)
        for ann in fd.announcements:
            assert isinstance(ann, str)
            assert len(ann) > 0

    def test_cooldown_prevents_spam(self, ba_config):
        """Same WARNING object should be suppressed by cooldown."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="table", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=2.5, smoothed_distance_m=2.5,
        )

        fd1 = _make_fd([obj])
        engine.prioritize(fd1)
        first_count = len(fd1.announcements)

        fd2 = _make_fd([DetectedObject(
            label="table", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=2.5, smoothed_distance_m=2.5,
        )])
        engine.prioritize(fd2)
        second_count = len(fd2.announcements)

        assert first_count >= 1
        assert second_count == 0

    def test_max_announcements_respected(self, ba_config):
        """Should never produce more than max_announcements_per_frame."""
        engine = BlindAssistanceEngine(ba_config)
        objects = [
            DetectedObject(label=f"obj{i}", bbox=(250, 200, 390, 400),
                           confidence=0.9, distance_m=1.0, smoothed_distance_m=1.0)
            for i in range(10)
        ]
        fd = _make_fd(objects)
        engine.prioritize(fd)
        assert len(fd.announcements) <= 2

    def test_structured_announcements_match(self, ba_config):
        """structured_announcements text should match announcements."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        fd = _make_fd([obj])
        engine.prioritize(fd)

        assert len(fd.structured_announcements) == len(fd.announcements)
        for sa, a in zip(fd.structured_announcements, fd.announcements):
            assert sa.text == a
