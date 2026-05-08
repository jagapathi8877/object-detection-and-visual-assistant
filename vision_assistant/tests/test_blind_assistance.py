"""
Tests for BlindAssistanceEngine -- Hazard-Aware Navigation Guidance.

Covers: 5-zone spatial classification, 4-tier urgency, staircase handling,
approach detection, cooldown with safety override, scene summary, and
max announcement limits.
"""

import time
import numpy as np
import pytest

from modules.blind_assistance import BlindAssistanceEngine
from utils.datatypes import DetectedObject, FrameData


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def ba_config():
    """Full config with blind_assistance section."""
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


def _make_frame_data(objects, frame_w=640, frame_h=480):
    """Create FrameData with objects and a dummy frame."""
    return FrameData(
        raw_frame=np.zeros((frame_h, frame_w, 3), dtype=np.uint8),
        frame_id=0,
        timestamp=time.time(),
        objects=objects,
    )


# ============================================================
# Test 1: Zone Classification
# ============================================================
class TestZoneClassification:
    """Test the 5-zone spatial classification system."""

    def test_dead_ahead(self, ba_config):
        """Object at centre of frame should be DEAD_AHEAD."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=2.0, smoothed_distance_m=2.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].zone == "DEAD_AHEAD"

    def test_left_near(self, ba_config):
        """Object on the left, lower half should be LEFT_NEAR."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="chair", bbox=(10, 250, 150, 450), confidence=0.9,
            distance_m=2.0, smoothed_distance_m=2.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].zone == "LEFT_NEAR"

    def test_right_near(self, ba_config):
        """Object on the right, lower half should be RIGHT_NEAR."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="chair", bbox=(500, 300, 630, 470), confidence=0.9,
            distance_m=2.0, smoothed_distance_m=2.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].zone == "RIGHT_NEAR"

    def test_left_far(self, ba_config):
        """Object on the left, upper half should be LEFT_FAR."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="person", bbox=(10, 10, 150, 150), confidence=0.9,
            distance_m=5.0, smoothed_distance_m=5.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].zone == "LEFT_FAR"

    def test_right_far(self, ba_config):
        """Object on the right, upper half should be RIGHT_FAR."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="person", bbox=(500, 10, 630, 150), confidence=0.9,
            distance_m=5.0, smoothed_distance_m=5.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].zone == "RIGHT_FAR"


# ============================================================
# Test 2: Urgency Tier Classification
# ============================================================
class TestUrgencyClassification:
    """Test the 4-tier urgency system."""

    def test_critical_close_ahead(self, ba_config):
        """Person at 1.0m dead ahead should be CRITICAL."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].urgency == "CRITICAL"

    def test_warning_moderate_distance(self, ba_config):
        """Chair at 2.5m should be WARNING."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="chair", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=2.5, smoothed_distance_m=2.5,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].urgency == "WARNING"

    def test_info_far_object(self, ba_config):
        """Table at 6m on the side should be INFO."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="table", bbox=(10, 10, 150, 150), confidence=0.9,
            distance_m=6.0, smoothed_distance_m=6.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].urgency == "INFO"

    def test_approaching_always_critical(self, ba_config):
        """Approaching object should be CRITICAL regardless of distance."""
        engine = BlindAssistanceEngine(ba_config)
        obj = DetectedObject(
            label="car", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=5.0, smoothed_distance_m=5.0,
            is_approaching=True, approach_velocity=0.8,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].urgency == "CRITICAL"


# ============================================================
# Test 3: Staircase Special Handling
# ============================================================
class TestStaircaseHandling:
    """Test stairs/step/escalator special urgency classification."""

    def test_stairs_going_down_critical(self, ba_config):
        """Stairs with bbox bottom at 75% of frame -> going down -> CRITICAL if close."""
        engine = BlindAssistanceEngine(ba_config)
        # bbox bottom at 360/480 = 75% > 60% threshold
        obj = DetectedObject(
            label="stairs", bbox=(200, 200, 400, 360), confidence=0.9,
            distance_m=1.5, smoothed_distance_m=1.5,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].urgency == "CRITICAL"

    def test_stairs_going_up_warning(self, ba_config):
        """Stairs with bbox in upper frame -> going up -> WARNING."""
        engine = BlindAssistanceEngine(ba_config)
        # bbox bottom at 250/480 = 52% < 60% threshold
        obj = DetectedObject(
            label="stairs", bbox=(200, 100, 400, 250), confidence=0.9,
            distance_m=2.5, smoothed_distance_m=2.5,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)
        assert fd.objects[0].urgency == "WARNING"


# ============================================================
# Test 4: Cooldown and Safety Override
# ============================================================
class TestCooldownAndSafety:
    """Test cooldown system with CRITICAL safety override."""

    def test_cooldown_suppresses_repeat(self, ba_config):
        """Same object announced twice in quick succession -> second suppressed."""
        engine = BlindAssistanceEngine(ba_config)

        obj1 = DetectedObject(
            label="chair", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=2.0, smoothed_distance_m=2.0,
        )
        fd1 = _make_frame_data([obj1])
        engine.prioritize(fd1)
        count1 = len(fd1.announcements)

        obj2 = DetectedObject(
            label="chair", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=2.0, smoothed_distance_m=2.0,
        )
        fd2 = _make_frame_data([obj2])
        engine.prioritize(fd2)
        count2 = len(fd2.announcements)

        # First should announce, second should be suppressed by cooldown
        assert count1 >= 1
        assert count2 == 0

    def test_critical_overrides_cooldown(self, ba_config):
        """CRITICAL announcements should bypass cooldown."""
        engine = BlindAssistanceEngine(ba_config)

        obj1 = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        fd1 = _make_frame_data([obj1])
        engine.prioritize(fd1)

        obj2 = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        fd2 = _make_frame_data([obj2])
        engine.prioritize(fd2)

        # Both should announce because CRITICAL overrides cooldown
        assert len(fd1.announcements) >= 1
        assert len(fd2.announcements) >= 1


# ============================================================
# Test 5: Max Announcements
# ============================================================
class TestMaxAnnouncements:
    """Test that max 2 announcements per frame is enforced."""

    def test_max_two_announcements(self, ba_config):
        """5 objects should produce at most 2 announcements."""
        engine = BlindAssistanceEngine(ba_config)

        objects = [
            DetectedObject(label="person", bbox=(250, 200, 390, 400),
                           confidence=0.9, distance_m=1.0, smoothed_distance_m=1.0),
            DetectedObject(label="car", bbox=(100, 100, 200, 200),
                           confidence=0.9, distance_m=1.5, smoothed_distance_m=1.5),
            DetectedObject(label="chair", bbox=(400, 300, 500, 450),
                           confidence=0.9, distance_m=2.0, smoothed_distance_m=2.0),
            DetectedObject(label="table", bbox=(10, 10, 100, 100),
                           confidence=0.9, distance_m=3.0, smoothed_distance_m=3.0),
            DetectedObject(label="door", bbox=(300, 100, 400, 300),
                           confidence=0.9, distance_m=4.0, smoothed_distance_m=4.0),
        ]
        fd = _make_frame_data(objects)
        engine.prioritize(fd)

        assert len(fd.announcements) <= 2


# ============================================================
# Test 6: Path is Clear
# ============================================================
class TestPathIsClear:
    """Test the 'Path is clear' fallback."""

    def test_clear_after_consecutive_empty_frames(self, ba_config):
        """3+ consecutive empty frames should trigger 'Path is clear'."""
        engine = BlindAssistanceEngine(ba_config)
        # Reset clear cooldown to far in the past so it doesn't block
        engine._cooldown_tracker = {"__path_clear__": 0.0}
        engine._clear_frame_count = 10  # Already past threshold

        fd = _make_frame_data([])
        engine.prioritize(fd)

        # After many empty frames, should have "Path is clear"
        assert any("clear" in a.lower() for a in fd.announcements)


# ============================================================
# Test 7: Structured Announcements
# ============================================================
class TestStructuredAnnouncements:
    """Test that structured_announcements are populated."""

    def test_structured_has_urgency(self, ba_config):
        """structured_announcements should have urgency field."""
        engine = BlindAssistanceEngine(ba_config)

        obj = DetectedObject(
            label="person", bbox=(250, 200, 390, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        fd = _make_frame_data([obj])
        engine.prioritize(fd)

        assert len(fd.structured_announcements) >= 1
        assert fd.structured_announcements[0].urgency == "CRITICAL"
        assert len(fd.structured_announcements[0].text) > 0
