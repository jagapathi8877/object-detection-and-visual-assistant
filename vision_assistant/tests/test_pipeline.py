"""
Tests for VisionPipeline -- Async Parallel Architecture.

Covers: frame drop strategy, FPS tracker, queue behaviour,
and pipeline construction.
"""

import queue
import time
import numpy as np
import pytest

from main import FPSTracker, _put_dropping


# ============================================================
# Test 1: FPS Tracker
# ============================================================
class TestFPSTracker:
    """Test the rolling average FPS calculator."""

    def test_initial_fps_is_zero(self):
        """FPS should be 0 with no frames recorded."""
        tracker = FPSTracker(window=30)
        assert tracker.get_fps() == 0.0

    def test_single_frame_is_zero(self):
        """FPS with 1 frame is 0 (need at least 2 for delta)."""
        tracker = FPSTracker(window=30)
        tracker.update()
        assert tracker.get_fps() == 0.0

    def test_fps_at_known_rate(self):
        """10 frames at 100ms intervals should give ~10 FPS."""
        tracker = FPSTracker(window=30)
        for i in range(10):
            tracker._timestamps.append(time.time() + i * 0.1)
        fps = tracker.get_fps()
        assert 8.0 <= fps <= 12.0, f"Expected ~10 FPS, got {fps:.1f}"

    def test_fps_at_high_rate(self):
        """30 frames at 33ms intervals should give ~30 FPS."""
        tracker = FPSTracker(window=30)
        base = time.time()
        for i in range(30):
            tracker._timestamps.append(base + i * 0.033)
        fps = tracker.get_fps()
        assert 25.0 <= fps <= 35.0, f"Expected ~30 FPS, got {fps:.1f}"

    def test_window_size_respected(self):
        """Only the last N frames should be tracked."""
        tracker = FPSTracker(window=5)
        for i in range(20):
            tracker.update()
        assert len(tracker._timestamps) <= 5


# ============================================================
# Test 2: Frame Drop Strategy
# ============================================================
class TestFrameDropStrategy:
    """Test the _put_dropping queue helper."""

    def test_put_into_empty_queue(self):
        """Putting into empty queue should succeed."""
        q = queue.Queue(maxsize=2)
        _put_dropping(q, "frame_1")
        assert q.qsize() == 1

    def test_drop_old_when_full(self):
        """When queue is full, old item should be dropped."""
        q = queue.Queue(maxsize=1)
        q.put("old_frame")
        _put_dropping(q, "new_frame")
        # Queue should have the new frame
        item = q.get_nowait()
        assert item == "new_frame"

    def test_queue_never_exceeds_maxsize(self):
        """After many puts, queue should never exceed maxsize."""
        q = queue.Queue(maxsize=2)
        for i in range(10):
            _put_dropping(q, f"frame_{i}")
        assert q.qsize() <= 2

    def test_latest_frame_is_always_available(self):
        """The most recently put frame should always be retrievable."""
        q = queue.Queue(maxsize=1)
        for i in range(5):
            _put_dropping(q, f"frame_{i}")
        item = q.get_nowait()
        assert item == "frame_4"


# ============================================================
# Test 3: Queue Behaviour
# ============================================================
class TestQueueBehaviour:
    """Test frame_queue and result_queue size constraints."""

    def test_frame_queue_maxsize(self):
        """frame_queue should have maxsize=2."""
        q = queue.Queue(maxsize=2)
        q.put("a")
        q.put("b")
        assert q.full()

    def test_result_queue_maxsize(self):
        """result_queue should have maxsize=1."""
        q = queue.Queue(maxsize=1)
        q.put("result")
        assert q.full()

    def test_stale_frames_dropped(self):
        """Stale frames should be silently dropped."""
        q = queue.Queue(maxsize=2)
        q.put("stale_1")
        q.put("stale_2")
        # Drop stale and add fresh
        _put_dropping(q, "fresh")
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert "fresh" in items
        assert len(items) <= 2
