"""
Tests for DepthEstimator -- MiDaS DPT-Small.

Covers: multi-zone sampling, EMA smoothing, approach velocity,
distance clamping, confidence scoring, and object tracking.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from utils.datatypes import DetectedObject, FrameData


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def depth_config():
    return {
        "model": "MiDaS_small",
        "scale_factor": 1000.0,
        "min_distance": 0.3,
        "max_distance": 8.0,
        "approaching_threshold_m_per_s": 0.4,
        "small_bbox_threshold_pct": 1.0,
        "ema_alpha_critical": 0.4,
        "ema_alpha_info": 0.2,
        "skip_frames": 1,
    }


@pytest.fixture
def sample_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def _create_depth_estimator(config):
    """Create a DepthEstimator with mocked MiDaS."""
    from modules.depth_estimator import DepthEstimator

    with patch.object(DepthEstimator, "__init__", lambda self, cfg: None):
        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator._scale_factor = config.get("scale_factor", 1000.0)
        estimator._min_distance = config["min_distance"]
        estimator._max_distance = config["max_distance"]
        estimator._approach_threshold = config["approaching_threshold_m_per_s"]
        estimator._small_bbox_pct = config["small_bbox_threshold_pct"]
        estimator._ema_alpha_critical = config["ema_alpha_critical"]
        estimator._ema_alpha_info = config["ema_alpha_info"]
        estimator._prev_objects = []
        estimator._prev_timestamp = 0.0
        estimator._depth_skip_frames = config.get("skip_frames", 1)
        estimator._frame_counter = 0
        estimator._cached_depth_map = None
        
        # Mock model and transform
        estimator._model = MagicMock()
        estimator._transform = MagicMock()
        estimator._transform.return_value = MagicMock()
        estimator._device = "cpu"
    return estimator


def _make_depth_map(inverse_depth_value: float, shape=(480, 640)):
    """Create a uniform inverse depth map (higher = closer)."""
    return np.full(shape, inverse_depth_value, dtype=np.float32)


def _make_frame_data(objects, frame_id=0, timestamp=1000.0):
    return FrameData(
        raw_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=timestamp,
        objects=objects,
    )


# ============================================================
# Test 1: Multi-Zone Sampling
# ============================================================
class TestMultiZoneSampling:
    """Test depth sampling across centre, lower-centre, and inset zones."""

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_uniform_depth_map(self, mock_midas, depth_config, sample_frame):
        """Uniform depth map should yield the expected distance."""
        estimator = _create_depth_estimator(depth_config)
        # distance = scale / inverse_depth -> 1000 / 400 = 2.5m
        depth_map = _make_depth_map(400.0)
        mock_midas.return_value = depth_map

        obj = DetectedObject(label="person", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj])

        result = estimator.estimate(sample_frame, fd)
        assert 2.0 <= result.objects[0].distance_m <= 3.0

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_centre_zone_captures_nearest(self, mock_midas, depth_config, sample_frame):
        """Centre zone should detect the nearest surface (max inverse depth)."""
        estimator = _create_depth_estimator(depth_config)

        # Create depth map: background at 5m (inv=200), object centre at 1.5m (inv=666)
        depth_map = _make_depth_map(200.0)
        depth_map[200:300, 150:250] = 666.0  # centre region is close
        mock_midas.return_value = depth_map

        obj = DetectedObject(label="person", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj])

        result = estimator.estimate(sample_frame, fd)
        # Should detect the 1.5m centre
        assert result.objects[0].distance_m < 3.0

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_small_bbox_low_confidence(self, mock_midas, depth_config, sample_frame):
        """Small bbox (< 1% frame area) should get low distance_confidence."""
        estimator = _create_depth_estimator(depth_config)
        depth_map = _make_depth_map(300.0)
        mock_midas.return_value = depth_map

        # Very small bbox
        obj = DetectedObject(label="person", bbox=(100, 100, 110, 110), confidence=0.9)
        fd = _make_frame_data([obj])

        result = estimator.estimate(sample_frame, fd)
        assert result.objects[0].distance_confidence == "low"


# ============================================================
# Test 2: Distance Clamping
# ============================================================
class TestDistanceClamping:
    """Test that distances are clamped to [min, max]."""

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_clamp_to_min(self, mock_midas, depth_config, sample_frame):
        """Depth below min_distance should be clamped."""
        estimator = _create_depth_estimator(depth_config)
        depth_map = _make_depth_map(10000.0) # 1000/10000 = 0.1m -> clamps to 0.3
        mock_midas.return_value = depth_map

        obj = DetectedObject(label="person", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj])

        result = estimator.estimate(sample_frame, fd)
        assert result.objects[0].distance_m >= 0.3

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_clamp_to_max(self, mock_midas, depth_config, sample_frame):
        """Depth above max_distance should be clamped."""
        estimator = _create_depth_estimator(depth_config)
        depth_map = _make_depth_map(50.0) # 1000/50 = 20m -> clamps to 8.0
        mock_midas.return_value = depth_map

        obj = DetectedObject(label="person", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj])

        result = estimator.estimate(sample_frame, fd)
        assert result.objects[0].distance_m <= 8.0


# ============================================================
# Test 3: Approach Velocity
# ============================================================
class TestApproachVelocity:
    """Test approach detection across frames."""

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_approaching_object(self, mock_midas, depth_config, sample_frame):
        """Object moving from 3m to 2m in 0.5s should be approaching."""
        estimator = _create_depth_estimator(depth_config)
        depth_map = _make_depth_map(500.0) # 1000/500 = 2m
        mock_midas.return_value = depth_map

        # Set up previous frame state
        prev_obj = DetectedObject(
            label="person", bbox=(100, 100, 300, 400), confidence=0.9,
            distance_m=3.0, smoothed_distance_m=3.0,
        )
        estimator._prev_objects = [prev_obj]
        estimator._prev_timestamp = 999.5  # 0.5s ago

        obj = DetectedObject(label="person", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj], timestamp=1000.0)

        result = estimator.estimate(sample_frame, fd)
        assert result.objects[0].is_approaching is True
        assert result.objects[0].approach_velocity > 0

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_stationary_object(self, mock_midas, depth_config, sample_frame):
        """Object staying at same distance should NOT be approaching."""
        estimator = _create_depth_estimator(depth_config)
        depth_map = _make_depth_map(333.3) # ~3m
        mock_midas.return_value = depth_map

        prev_obj = DetectedObject(
            label="person", bbox=(100, 100, 300, 400), confidence=0.9,
            distance_m=3.0, smoothed_distance_m=3.0,
        )
        estimator._prev_objects = [prev_obj]
        estimator._prev_timestamp = 999.5

        obj = DetectedObject(label="person", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj], timestamp=1000.0)

        result = estimator.estimate(sample_frame, fd)
        assert result.objects[0].is_approaching is False


# ============================================================
# Test 4: EMA Smoothing
# ============================================================
class TestEMASmoothing:
    """Test exponential moving average distance smoothing."""

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_ema_converges(self, mock_midas, depth_config, sample_frame):
        """Smoothed distance should converge over multiple frames."""
        estimator = _create_depth_estimator(depth_config)

        prev_obj = DetectedObject(
            label="chair", bbox=(100, 100, 300, 400), confidence=0.9,
            distance_m=4.0, smoothed_distance_m=4.0,
        )
        estimator._prev_objects = [prev_obj]
        estimator._prev_timestamp = 999.0

        depth_map = _make_depth_map(333.3) # ~3m
        mock_midas.return_value = depth_map

        obj = DetectedObject(label="chair", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj], timestamp=1000.0)

        result = estimator.estimate(sample_frame, fd)
        smoothed = result.objects[0].smoothed_distance_m
        # Should be between old (4.0) and new (3.0)
        assert 3.0 <= smoothed <= 4.0

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_first_frame_no_smoothing(self, mock_midas, depth_config, sample_frame):
        """First frame should set smoothed = raw."""
        estimator = _create_depth_estimator(depth_config)
        depth_map = _make_depth_map(400.0) # 2.5m
        mock_midas.return_value = depth_map

        obj = DetectedObject(label="chair", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj])

        result = estimator.estimate(sample_frame, fd)
        assert result.objects[0].smoothed_distance_m is not None


# ============================================================
# Test 5: Edge Cases
# ============================================================
class TestDepthEdgeCases:
    """Test error handling and edge cases."""

    def test_rejects_none_frame(self, depth_config):
        """Should raise ValueError on None frame."""
        estimator = _create_depth_estimator(depth_config)
        fd = _make_frame_data([])
        with pytest.raises(ValueError):
            estimator.estimate(None, fd)

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_no_objects_skips_depth(self, mock_midas, depth_config, sample_frame):
        """No detected objects should skip depth estimation entirely."""
        estimator = _create_depth_estimator(depth_config)
        fd = _make_frame_data([])
        result = estimator.estimate(sample_frame, fd)
        assert len(result.objects) == 0
        mock_midas.assert_not_called()

    @patch("modules.depth_estimator.DepthEstimator._run_midas")
    def test_depth_failure_assigns_max(self, mock_midas, depth_config, sample_frame):
        """If depth model fails, assign max distance + low confidence."""
        estimator = _create_depth_estimator(depth_config)
        mock_midas.side_effect = Exception("model crash")

        obj = DetectedObject(label="person", bbox=(100, 100, 300, 400), confidence=0.9)
        fd = _make_frame_data([obj])

        result = estimator.estimate(sample_frame, fd)
        assert result.objects[0].distance_m == 8.0
        assert result.objects[0].distance_confidence == "low"
