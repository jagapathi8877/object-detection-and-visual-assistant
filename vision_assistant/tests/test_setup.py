"""
Smoke Tests for Sprint 1 — Environment Setup & Architecture.

Validates:
  1. All critical imports succeed (no missing packages)
  2. config.yaml loads without errors and contains required sections
  3. FrameData and DetectedObject dataclasses can be instantiated
  4. Logger can be created and used
  5. Helper functions work correctly

Run:  pytest tests/test_setup.py -v
"""

import os
import sys
import time

import numpy as np
import pytest

# ── Ensure project root is on the path ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Test 1: All Critical Imports
# ============================================================
class TestImports:
    """Verify that every required package is importable."""

    def test_import_numpy(self) -> None:
        """NumPy must be available for array operations."""
        import numpy
        assert numpy is not None

    def test_import_cv2(self) -> None:
        """OpenCV must be available for camera and image processing."""
        import cv2
        assert cv2 is not None

    def test_import_torch(self) -> None:
        """PyTorch must be available for MiDaS depth estimation."""
        import torch
        assert torch is not None

    def test_import_yaml(self) -> None:
        """PyYAML must be available for config loading."""
        import yaml
        assert yaml is not None

    def test_import_pyttsx3(self) -> None:
        """pyttsx3 must be available for offline text-to-speech."""
        import pyttsx3
        assert pyttsx3 is not None

    def test_import_ultralytics(self) -> None:
        """Ultralytics must be available for YOLOv8 object detection."""
        from ultralytics import YOLO
        assert YOLO is not None

    def test_import_project_datatypes(self) -> None:
        """Project datatype module must be importable."""
        from utils.datatypes import FrameData, DetectedObject
        assert FrameData is not None
        assert DetectedObject is not None

    def test_import_project_logger(self) -> None:
        """Project logger module must be importable."""
        from utils.logger import get_logger
        assert get_logger is not None

    def test_import_project_helpers(self) -> None:
        """Project helper module must be importable."""
        from utils.helpers import load_config, preprocess_frame, get_frame_center
        assert load_config is not None
        assert preprocess_frame is not None
        assert get_frame_center is not None


# ============================================================
# Test 2: Configuration Loading
# ============================================================
class TestConfig:
    """Verify config.yaml loads correctly and contains all sections."""

    @pytest.fixture
    def config(self):
        """Load the project configuration."""
        from utils.helpers import load_config
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        return load_config(config_path)

    def test_config_loads_without_error(self, config) -> None:
        """config.yaml must parse without exceptions."""
        assert config is not None
        assert isinstance(config, dict)

    def test_config_has_camera_section(self, config) -> None:
        """Camera section must exist with required keys."""
        assert "camera" in config
        cam = config["camera"]
        assert "index" in cam
        assert "width" in cam
        assert "height" in cam
        assert "fps" in cam

    def test_config_has_detection_section(self, config) -> None:
        """Detection section must exist with required keys."""
        assert "detection" in config
        det = config["detection"]
        assert "model" in det
        assert "confidence" in det
        assert "max_detections" in det

    def test_config_has_depth_section(self, config) -> None:
        """Depth section must exist with required keys."""
        assert "depth" in config
        dep = config["depth"]
        assert "model" in dep
        assert "min_distance" in dep
        assert "max_distance" in dep

    def test_config_has_direction_section(self, config) -> None:
        """Direction section must exist with boundary values."""
        assert "direction" in config
        dirn = config["direction"]
        assert "left_boundary" in dirn
        assert "right_boundary" in dirn
        assert 0 < dirn["left_boundary"] < dirn["right_boundary"] < 1

    def test_config_has_audio_section(self, config) -> None:
        """Audio section must exist with required keys."""
        assert "audio" in config
        aud = config["audio"]
        assert "engine" in aud
        assert "rate" in aud
        assert "volume" in aud
        assert "cooldown_seconds" in aud

    def test_config_has_blind_assistance_section(self, config) -> None:
        """Blind assistance section must exist."""
        assert "blind_assistance" in config
        ba = config["blind_assistance"]
        assert "critical_distance_m" in ba
        assert "warning_distance_m" in ba
        assert "max_announcements_per_frame" in ba

    def test_config_has_pipeline_section(self, config) -> None:
        """Pipeline section must exist."""
        assert "pipeline" in config
        pl = config["pipeline"]
        assert "frame_queue_size" in pl
        assert "result_queue_size" in pl

    def test_config_has_system_section(self, config) -> None:
        """System section must exist with required keys."""
        assert "system" in config
        sys_cfg = config["system"]
        assert "log_level" in sys_cfg
        assert "headless" in sys_cfg
        assert "target_latency_ms" in sys_cfg


# ============================================================
# Test 3: Dataclass Instantiation
# ============================================================
class TestDataTypes:
    """Verify FrameData and DetectedObject dataclasses work correctly."""

    def test_detected_object_creation(self) -> None:
        """DetectedObject should be instantiable with required fields."""
        from utils.datatypes import DetectedObject

        obj = DetectedObject(
            label="person",
            bbox=(100, 50, 300, 400),
            confidence=0.87,
        )
        assert obj.label == "person"
        assert obj.bbox == (100, 50, 300, 400)
        assert obj.confidence == 0.87
        # Optional fields should default to None
        assert obj.distance_m is None
        assert obj.direction is None
        assert obj.priority_score is None

    def test_detected_object_with_all_fields(self) -> None:
        """DetectedObject should accept all optional fields."""
        from utils.datatypes import DetectedObject

        obj = DetectedObject(
            label="chair",
            bbox=(10, 20, 200, 300),
            confidence=0.72,
            distance_m=2.5,
            direction="left",
            priority_score=0.88,
        )
        assert obj.distance_m == 2.5
        assert obj.direction == "left"
        assert obj.priority_score == 0.88

    def test_frame_data_creation(self) -> None:
        """FrameData should be instantiable with a numpy frame."""
        from utils.datatypes import FrameData

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = FrameData(
            raw_frame=fake_frame,
            frame_id=0,
            timestamp=time.time(),
        )
        assert fd.raw_frame.shape == (480, 640, 3)
        assert fd.frame_id == 0
        assert fd.timestamp > 0
        # Default lists should be empty
        assert fd.objects == []
        assert fd.announcements == []

    def test_frame_data_with_objects(self) -> None:
        """FrameData should hold a list of DetectedObject instances."""
        from utils.datatypes import FrameData, DetectedObject

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        obj = DetectedObject(label="car", bbox=(0, 0, 100, 100), confidence=0.9)
        fd = FrameData(
            raw_frame=fake_frame,
            frame_id=1,
            timestamp=time.time(),
            objects=[obj],
        )
        assert len(fd.objects) == 1
        assert fd.objects[0].label == "car"

    def test_detected_object_repr(self) -> None:
        """DetectedObject repr should be concise and informative."""
        from utils.datatypes import DetectedObject

        obj = DetectedObject(
            label="person",
            bbox=(0, 0, 100, 200),
            confidence=0.95,
            distance_m=1.5,
            direction="ahead",
        )
        repr_str = repr(obj)
        assert "person" in repr_str
        assert "1.5m" in repr_str
        assert "ahead" in repr_str


# ============================================================
# Test 4: Logger
# ============================================================
class TestLogger:
    """Verify logger setup and basic functionality."""

    def test_logger_creation(self) -> None:
        """Logger should be creatable with a module name."""
        from utils.logger import get_logger

        log = get_logger("test_module")
        assert log is not None
        assert log.name == "test_module"

    def test_logger_has_handlers(self) -> None:
        """Logger should have at least console + file handlers."""
        from utils.logger import get_logger

        log = get_logger("test_handlers_module")
        assert len(log.handlers) >= 2  # console + file

    def test_logger_does_not_duplicate_handlers(self) -> None:
        """Calling get_logger twice with same name should not add handlers."""
        from utils.logger import get_logger

        log1 = get_logger("test_dedup_module")
        handler_count = len(log1.handlers)
        log2 = get_logger("test_dedup_module")
        assert len(log2.handlers) == handler_count


# ============================================================
# Test 5: Helper Functions
# ============================================================
class TestHelpers:
    """Verify helper utility functions."""

    def test_preprocess_frame_shape(self) -> None:
        """Preprocessed frame should have target dimensions."""
        from utils.helpers import preprocess_frame

        raw = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(raw, width=320, height=240)
        assert result.shape == (240, 320, 3)

    def test_preprocess_frame_dtype(self) -> None:
        """Preprocessed frame should be float32."""
        from utils.helpers import preprocess_frame

        raw = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(raw, width=640, height=480)
        assert result.dtype == np.float32

    def test_preprocess_frame_range(self) -> None:
        """Preprocessed pixel values should be in [0.0, 1.0]."""
        from utils.helpers import preprocess_frame

        raw = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_frame(raw, width=100, height=100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_frame_rejects_none(self) -> None:
        """preprocess_frame should raise ValueError on None input."""
        from utils.helpers import preprocess_frame

        with pytest.raises(ValueError, match="empty or None"):
            preprocess_frame(None, width=100, height=100)

    def test_preprocess_frame_rejects_grayscale(self) -> None:
        """preprocess_frame should reject single-channel images."""
        from utils.helpers import preprocess_frame

        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-channel"):
            preprocess_frame(gray, width=100, height=100)

    def test_get_frame_center(self) -> None:
        """Frame center should be computed correctly."""
        from utils.helpers import get_frame_center

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cx, cy = get_frame_center(frame)
        assert cx == 320
        assert cy == 240

    def test_get_frame_center_odd_dimensions(self) -> None:
        """Frame center with odd dims should use integer division."""
        from utils.helpers import get_frame_center

        frame = np.zeros((101, 201, 3), dtype=np.uint8)
        cx, cy = get_frame_center(frame)
        assert cx == 100  # 201 // 2
        assert cy == 50   # 101 // 2

    def test_load_config_missing_file(self) -> None:
        """load_config should raise FileNotFoundError for missing path."""
        from utils.helpers import load_config

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config_that_does_not_exist.yaml")
