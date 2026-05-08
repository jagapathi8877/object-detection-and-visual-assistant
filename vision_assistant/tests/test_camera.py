"""
Unit Tests for the Camera Input Module (Sprint 2).

Tests cover:
  - CameraStream initialisation and config validation
  - Frame capture with mocked cv2.VideoCapture
  - Preprocessed frame shape, dtype, and value range
  - Thread lifecycle (start/stop/is_running)
  - preprocess_frame and get_frame_center helpers
  - Context manager support
  - Error handling (camera not found, missing config keys)

All camera hardware is mocked — tests run without a physical webcam.

Run:  pytest tests/test_camera.py -v
"""

import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.camera import CameraStream
from utils.helpers import preprocess_frame, get_frame_center


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def camera_config() -> dict:
    """Standard camera configuration for testing."""
    return {
        "index": 0,
        "width": 320,
        "height": 240,
        "fps": 30,
    }


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Generate a synthetic BGR frame (like what a real camera returns)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def create_mock_video_capture(frame: np.ndarray, is_opened: bool = True):
    """Create a mock cv2.VideoCapture that returns the given frame.

    Args:
        frame: The BGR frame to return on every read() call.
        is_opened: Whether isOpened() returns True.

    Returns:
        Configured MagicMock mimicking cv2.VideoCapture.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = is_opened
    mock_cap.read.return_value = (True, frame.copy())
    mock_cap.get.return_value = 30.0  # Fake FPS
    mock_cap.set.return_value = True
    return mock_cap


# ============================================================
# Test 1: CameraStream Initialisation
# ============================================================
class TestCameraStreamInit:
    """Test CameraStream constructor and config validation."""

    def test_init_with_valid_config(self, camera_config) -> None:
        """CameraStream should initialise without errors given valid config."""
        cam = CameraStream(camera_config)
        assert cam is not None
        assert not cam.is_running()

    def test_init_rejects_missing_index(self, camera_config) -> None:
        """CameraStream should raise ValueError when 'index' key is missing."""
        del camera_config["index"]
        with pytest.raises(ValueError, match="index"):
            CameraStream(camera_config)

    def test_init_rejects_missing_width(self, camera_config) -> None:
        """CameraStream should raise ValueError when 'width' key is missing."""
        del camera_config["width"]
        with pytest.raises(ValueError, match="width"):
            CameraStream(camera_config)

    def test_init_rejects_missing_height(self, camera_config) -> None:
        """CameraStream should raise ValueError when 'height' key is missing."""
        del camera_config["height"]
        with pytest.raises(ValueError, match="height"):
            CameraStream(camera_config)

    def test_init_rejects_missing_fps(self, camera_config) -> None:
        """CameraStream should raise ValueError when 'fps' key is missing."""
        del camera_config["fps"]
        with pytest.raises(ValueError, match="fps"):
            CameraStream(camera_config)


# ============================================================
# Test 2: Frame Capture with Mocked Camera
# ============================================================
class TestCameraCapture:
    """Test frame capture with mocked cv2.VideoCapture."""

    @patch("modules.camera.cv2.VideoCapture")
    def test_read_returns_correctly_shaped_frame(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """read() should return a frame with shape (height, width, 3)."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()

        # Give the capture thread time to produce a frame
        time.sleep(0.5)
        frame = cam.read()

        assert frame.shape == (
            camera_config["height"],
            camera_config["width"],
            3,
        )
        cam.stop()

    @patch("modules.camera.cv2.VideoCapture")
    def test_read_returns_float32_dtype(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """read() should return a float32 numpy array."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()
        time.sleep(0.5)

        frame = cam.read()
        assert frame.dtype == np.float32
        cam.stop()

    @patch("modules.camera.cv2.VideoCapture")
    def test_read_values_in_unit_range(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """read() frame values should be normalised to [0.0, 1.0]."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()
        time.sleep(0.5)

        frame = cam.read()
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0
        cam.stop()

    @patch("modules.camera.cv2.VideoCapture")
    def test_read_raises_when_not_started(
        self, mock_vc_class, camera_config
    ) -> None:
        """read() should raise RuntimeError if start() was not called."""
        cam = CameraStream(camera_config)
        with pytest.raises(RuntimeError, match="not running"):
            cam.read()


# ============================================================
# Test 3: Thread Lifecycle
# ============================================================
class TestCameraLifecycle:
    """Test start/stop/is_running thread management."""

    @patch("modules.camera.cv2.VideoCapture")
    def test_is_running_after_start(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """is_running() should return True after start() is called."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()
        time.sleep(0.3)

        assert cam.is_running() is True
        cam.stop()

    @patch("modules.camera.cv2.VideoCapture")
    def test_is_running_false_after_stop(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """is_running() should return False after stop() is called."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()
        time.sleep(0.3)
        cam.stop()

        assert cam.is_running() is False

    @patch("modules.camera.cv2.VideoCapture")
    def test_stop_joins_thread_within_timeout(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """stop() should join the capture thread within 2 seconds."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()
        time.sleep(0.3)

        start_time = time.time()
        cam.stop()
        elapsed = time.time() - start_time

        assert elapsed < 2.0, f"stop() took {elapsed:.2f}s (>2s timeout)"

    @patch("modules.camera.cv2.VideoCapture")
    def test_stop_releases_video_capture(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """stop() should call release() on the VideoCapture device."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()
        time.sleep(0.3)
        cam.stop()

        mock_cap.release.assert_called_once()

    @patch("modules.camera.cv2.VideoCapture")
    def test_double_stop_is_safe(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """Calling stop() twice should not raise an error."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        cam.start()
        time.sleep(0.3)
        cam.stop()
        cam.stop()  # Should not raise

    @patch("modules.camera.cv2.VideoCapture")
    def test_camera_not_found_raises_runtime_error(
        self, mock_vc_class, camera_config
    ) -> None:
        """start() should raise RuntimeError if camera cannot be opened."""
        mock_cap = create_mock_video_capture(
            np.zeros((480, 640, 3), dtype=np.uint8), is_opened=False
        )
        mock_vc_class.return_value = mock_cap

        cam = CameraStream(camera_config)
        with pytest.raises(RuntimeError, match="Cannot open camera"):
            cam.start()


# ============================================================
# Test 4: Context Manager
# ============================================================
class TestCameraContextManager:
    """Test CameraStream as a context manager."""

    @patch("modules.camera.cv2.VideoCapture")
    def test_context_manager_starts_and_stops(
        self, mock_vc_class, camera_config, synthetic_frame
    ) -> None:
        """Using 'with' should auto-start and auto-stop the camera."""
        mock_cap = create_mock_video_capture(synthetic_frame)
        mock_vc_class.return_value = mock_cap

        with CameraStream(camera_config) as cam:
            time.sleep(0.3)
            assert cam.is_running() is True

        # After exiting context, camera should be stopped
        assert cam.is_running() is False
        mock_cap.release.assert_called_once()


# ============================================================
# Test 5: Helper Functions (preprocess_frame, get_frame_center)
# ============================================================
class TestPreprocessFrame:
    """Test the preprocess_frame helper function."""

    def test_output_shape(self, synthetic_frame) -> None:
        """Output shape should match target dimensions."""
        result = preprocess_frame(synthetic_frame, width=320, height=240)
        assert result.shape == (240, 320, 3)

    def test_output_dtype(self, synthetic_frame) -> None:
        """Output dtype should be float32."""
        result = preprocess_frame(synthetic_frame, width=320, height=240)
        assert result.dtype == np.float32

    def test_output_range(self, synthetic_frame) -> None:
        """Output values should be in [0.0, 1.0]."""
        result = preprocess_frame(synthetic_frame, width=320, height=240)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_pure_white_frame(self) -> None:
        """Pure white frame (255) should normalise to max 1.0."""
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = preprocess_frame(white, width=50, height=50)
        assert np.isclose(result.max(), 1.0)

    def test_pure_black_frame(self) -> None:
        """Pure black frame (0) should normalise to min 0.0."""
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_frame(black, width=50, height=50)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 0.0)


class TestGetFrameCenter:
    """Test the get_frame_center helper function."""

    def test_even_dimensions(self) -> None:
        """Center of 640x480 should be (320, 240)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cx, cy = get_frame_center(frame)
        assert cx == 320
        assert cy == 240

    def test_odd_dimensions(self) -> None:
        """Center of 201x101 should be (100, 50) via integer division."""
        frame = np.zeros((101, 201, 3), dtype=np.uint8)
        cx, cy = get_frame_center(frame)
        assert cx == 100
        assert cy == 50

    def test_square_frame(self) -> None:
        """Center of 100x100 should be (50, 50)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cx, cy = get_frame_center(frame)
        assert cx == 50
        assert cy == 50
