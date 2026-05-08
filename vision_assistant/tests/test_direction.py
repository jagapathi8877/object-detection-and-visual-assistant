"""
Unit Tests for the Direction Calculation Module (Sprint 3C).

Tests cover:
  - All three direction zones (left, ahead, right)
  - Boundary values (exactly on left_boundary, exactly on right_boundary)
  - Center of frame → 'ahead'
  - Far left (0%) → 'left', far right (100%) → 'right'
  - Edge case: bbox wider than frame
  - Edge case: zero frame width raises ValueError
  - Pipeline integration via assign_directions()

Run:  pytest tests/test_direction.py -v
"""

import os
import sys
import time

import numpy as np
import pytest

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.direction import calculate_direction, assign_directions
from utils.datatypes import DetectedObject, FrameData


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def direction_config() -> dict:
    """Standard direction configuration (33%/66% boundaries)."""
    return {
        "left_boundary": 0.33,
        "right_boundary": 0.66,
    }


FRAME_WIDTH = 640  # Standard test frame width


# ============================================================
# Test 1: Basic Direction Zones
# ============================================================
class TestDirectionZones:
    """Test that objects are classified into the correct zone."""

    def test_object_on_far_left(self, direction_config) -> None:
        """Object at ~5% of frame width should be 'left'."""
        # bbox center_x = (10 + 50) / 2 = 30 → 30/640 ≈ 4.7%
        bbox = (10, 100, 50, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "left"

    def test_object_in_left_zone(self, direction_config) -> None:
        """Object at ~20% of frame width should be 'left'."""
        # bbox center_x = (80 + 180) / 2 = 130 → 130/640 ≈ 20.3%
        bbox = (80, 100, 180, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "left"

    def test_object_in_center(self, direction_config) -> None:
        """Object at exact center (50%) should be 'ahead'."""
        # bbox center_x = (270 + 370) / 2 = 320 → 320/640 = 50%
        bbox = (270, 100, 370, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "ahead"

    def test_object_in_right_zone(self, direction_config) -> None:
        """Object at ~80% of frame width should be 'right'."""
        # bbox center_x = (460 + 560) / 2 = 510 → 510/640 ≈ 79.7%
        bbox = (460, 100, 560, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "right"

    def test_object_on_far_right(self, direction_config) -> None:
        """Object at ~95% of frame width should be 'right'."""
        # bbox center_x = (580 + 630) / 2 = 605 → 605/640 ≈ 94.5%
        bbox = (580, 100, 630, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "right"


# ============================================================
# Test 2: Boundary Values
# ============================================================
class TestDirectionBoundaries:
    """Test objects exactly on zone boundaries."""

    def test_exactly_on_left_boundary(self, direction_config) -> None:
        """Object center exactly at left_boundary (33%) → 'left' (inclusive)."""
        # left_boundary_px = 640 * 0.33 = 211.2
        # Need center_x = 211.2 → bbox = (161.2, _, 261.2, _)
        # Use (161, _, 261, _) → center = 211.0 which is ≤ 211.2 → 'left'
        left_px = FRAME_WIDTH * 0.33
        half_w = 50
        x1 = int(left_px - half_w)
        x2 = int(left_px + half_w)
        bbox = (x1, 100, x2, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "left"

    def test_exactly_on_right_boundary(self, direction_config) -> None:
        """Object center at or past right_boundary (66%) → 'right' (inclusive)."""
        # right_boundary_px = 640 * 0.66 = 422.4
        # Need center_x >= 422.4 → use bbox (373, _, 473, _) → center = 423.0
        bbox = (373, 100, 473, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        # center_x = 423.0 >= 422.4 → 'right'
        assert result == "right"

    def test_just_inside_center_zone(self, direction_config) -> None:
        """Object center just past left boundary → 'ahead'."""
        # center_x = 215 (just past 211.2) → 'ahead'
        bbox = (165, 100, 265, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "ahead"


# ============================================================
# Test 3: Edge Cases
# ============================================================
class TestDirectionEdgeCases:
    """Test edge cases and error handling."""

    def test_bbox_wider_than_frame(self, direction_config) -> None:
        """A bbox spanning the entire frame should return 'ahead'."""
        bbox = (0, 0, FRAME_WIDTH, 480)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "ahead"

    def test_bbox_at_pixel_zero(self, direction_config) -> None:
        """A bbox starting and ending at x=0 should return 'left'."""
        bbox = (0, 0, 0, 100)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "left"

    def test_bbox_at_max_pixel(self, direction_config) -> None:
        """A bbox at the rightmost pixel should return 'right'."""
        bbox = (FRAME_WIDTH, 0, FRAME_WIDTH, 100)
        result = calculate_direction(bbox, FRAME_WIDTH, direction_config)
        assert result == "right"

    def test_zero_frame_width_raises_error(self, direction_config) -> None:
        """Zero frame_width should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculate_direction((100, 100, 200, 200), 0, direction_config)

    def test_negative_frame_width_raises_error(self, direction_config) -> None:
        """Negative frame_width should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculate_direction((100, 100, 200, 200), -1, direction_config)

    def test_default_boundaries_used(self) -> None:
        """If boundaries missing from config, defaults (0.33/0.66) are used."""
        empty_config: dict = {}
        # center_x = 320 → 50% → should be 'ahead' with default boundaries
        bbox = (270, 100, 370, 300)
        result = calculate_direction(bbox, FRAME_WIDTH, empty_config)
        assert result == "ahead"


# ============================================================
# Test 4: Pipeline Integration (assign_directions)
# ============================================================
class TestAssignDirections:
    """Test the assign_directions pipeline helper."""

    def test_assigns_direction_to_all_objects(self, direction_config) -> None:
        """assign_directions should set direction on every DetectedObject."""
        obj_left = DetectedObject(
            label="person", bbox=(10, 50, 50, 200), confidence=0.9
        )
        obj_center = DetectedObject(
            label="chair", bbox=(270, 100, 370, 300), confidence=0.8
        )
        obj_right = DetectedObject(
            label="car", bbox=(500, 50, 600, 200), confidence=0.85
        )
        fd = FrameData(
            raw_frame=np.zeros((480, FRAME_WIDTH, 3), dtype=np.uint8),
            frame_id=0,
            timestamp=time.time(),
            objects=[obj_left, obj_center, obj_right],
        )

        assign_directions(fd, FRAME_WIDTH, direction_config)

        assert fd.objects[0].direction == "left"
        assert fd.objects[1].direction == "ahead"
        assert fd.objects[2].direction == "right"

    def test_no_objects_does_not_crash(self, direction_config) -> None:
        """assign_directions with empty object list should not crash."""
        fd = FrameData(
            raw_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            timestamp=time.time(),
        )
        # Should not raise
        assign_directions(fd, FRAME_WIDTH, direction_config)
        assert len(fd.objects) == 0
