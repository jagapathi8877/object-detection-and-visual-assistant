"""
Tests for ObjectDetector -- YOLOv8n with COCO 80-class detection.

Covers: model loading, label normalisation, confidence filtering,
deduplication, max detections, and whitelist filtering.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from utils.datatypes import DetectedObject, FrameData
from utils.label_map import get_speech_label


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def detection_config():
    """Standard detection config for tests."""
    return {
        "model": "yolov8n.pt",
        "confidence": 0.35,
        "iou_threshold": 0.45,
        "max_detections": 10,
        "label_normalisation": True,
        "max_objects": 5,
        "imgsz": 320,
    }


@pytest.fixture
def sample_frame():
    """640x480 BGR uint8 frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def empty_frame_data():
    """Empty FrameData for testing."""
    return FrameData(
        raw_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        frame_id=0,
        timestamp=0.0,
    )


def _make_mock_box(cls_id, conf, xyxy):
    """Create a mock YOLO detection box."""
    box = MagicMock()
    box.cls = MagicMock()
    box.cls.item.return_value = cls_id
    box.conf = MagicMock()
    box.conf.item.return_value = conf
    box.xyxy = [np.array(xyxy)]
    return box


def _create_detector(config):
    """Create a detector with mocked model loading."""
    from modules.detector import ObjectDetector

    with patch.object(ObjectDetector, "_load_model", return_value=MagicMock()):
        detector = ObjectDetector(config)
        
        # Mock the loaded model's class names
        detector._class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
            72: "refrigerator", 73: "book", 74: "clock"
        }
    return detector


# ============================================================
# Test 1: Label Normalisation
# ============================================================
class TestLabelNormalisation:
    """Test speech-friendly label mapping."""

    def test_couch_to_sofa(self):
        assert get_speech_label("couch") == "sofa"

    def test_person_unchanged(self):
        assert get_speech_label("person") == "person"

    def test_unknown_label_returns_lowercase(self):
        """Unknown labels should be returned lowercase."""
        assert get_speech_label("UNKNOWN_OBJECT") == "unknown_object"


# ============================================================
# Test 2: Detection
# ============================================================
class TestDetection:
    """Test the detect() method."""

    def test_detect_populates_objects(self, detection_config, sample_frame, empty_frame_data):
        """detect() should populate frame_data.objects."""
        detector = _create_detector(detection_config)

        mock_result = MagicMock()
        mock_result.boxes = [_make_mock_box(0, 0.85, [100, 50, 300, 400])]
        detector._model.return_value = [mock_result]

        result = detector.detect(sample_frame, empty_frame_data)
        assert len(result.objects) == 1
        assert result.objects[0].label == "person"

    def test_detect_correct_bbox(self, detection_config, sample_frame, empty_frame_data):
        """Bounding boxes should be integers."""
        detector = _create_detector(detection_config)

        mock_result = MagicMock()
        mock_result.boxes = [_make_mock_box(0, 0.90, [100.5, 50.2, 300.7, 400.1])]
        detector._model.return_value = [mock_result]

        result = detector.detect(sample_frame, empty_frame_data)
        bbox = result.objects[0].bbox
        assert all(isinstance(v, int) for v in bbox)

    def test_detect_confidence_filter(self, detection_config, sample_frame, empty_frame_data):
        """Objects below conf_threshold should be excluded."""
        detector = _create_detector(detection_config)

        mock_result = MagicMock()
        mock_result.boxes = [
            _make_mock_box(0, 0.30, [10, 10, 100, 100]),  # below 0.35
            _make_mock_box(2, 0.50, [200, 200, 400, 400]),  # above
        ]
        detector._model.return_value = [mock_result]

        result = detector.detect(sample_frame, empty_frame_data)
        assert len(result.objects) == 1
        assert result.objects[0].label == "car"

    def test_detect_empty_result(self, detection_config, sample_frame, empty_frame_data):
        """Empty results should return empty objects list."""
        detector = _create_detector(detection_config)

        mock_result = MagicMock()
        mock_result.boxes = []
        detector._model.return_value = [mock_result]

        result = detector.detect(sample_frame, empty_frame_data)
        assert len(result.objects) == 0

    def test_detect_rejects_none_frame(self, detection_config, empty_frame_data):
        """detect() should raise ValueError on None frame."""
        detector = _create_detector(detection_config)
        with pytest.raises(ValueError):
            detector.detect(None, empty_frame_data)

    def test_detect_returns_frame_data(self, detection_config, sample_frame, empty_frame_data):
        """detect() must return a FrameData instance."""
        detector = _create_detector(detection_config)
        detector._model.return_value = []
        result = detector.detect(sample_frame, empty_frame_data)
        assert isinstance(result, FrameData)

    def test_detect_label_normalisation(self, detection_config, sample_frame, empty_frame_data):
        """'couch' should be normalised to 'sofa'."""
        detector = _create_detector(detection_config)

        mock_result = MagicMock()
        mock_result.boxes = [_make_mock_box(57, 0.80, [10, 10, 200, 200])]
        detector._model.return_value = [mock_result]

        result = detector.detect(sample_frame, empty_frame_data)
        assert result.objects[0].label == "sofa"

    def test_detect_whitelist(self, detection_config, sample_frame, empty_frame_data):
        """Objects not in whitelist should be ignored if whitelist is provided."""
        detection_config["classes"] = ["person", "chair"]
        detector = _create_detector(detection_config)
        
        mock_result = MagicMock()
        mock_result.boxes = [
            _make_mock_box(0, 0.90, [10, 10, 100, 100]),   # person (whitelisted)
            _make_mock_box(2, 0.90, [20, 20, 100, 100]),   # car (not whitelisted)
            _make_mock_box(56, 0.90, [30, 30, 100, 100]),  # chair (whitelisted)
        ]
        detector._model.return_value = [mock_result]
        
        result = detector.detect(sample_frame, empty_frame_data)
        assert len(result.objects) == 2
        labels = {o.label for o in result.objects}
        assert "person" in labels
        assert "chair" in labels
        assert "car" not in labels


# ============================================================
# Test 3: Deduplication
# ============================================================
class TestDeduplication:
    """Test in-frame deduplication logic."""

    def test_duplicate_same_label_high_iou(self, detection_config, sample_frame, empty_frame_data):
        """Two 'person' boxes with IoU > 0.5 -> keep highest confidence."""
        detector = _create_detector(detection_config)

        mock_result = MagicMock()
        mock_result.boxes = [
            _make_mock_box(0, 0.90, [100, 100, 300, 300]),
            _make_mock_box(0, 0.70, [110, 110, 290, 290]),  # overlaps heavily
        ]
        detector._model.return_value = [mock_result]

        result = detector.detect(sample_frame, empty_frame_data)
        assert len(result.objects) == 1
        assert result.objects[0].confidence == 0.90

    def test_different_labels_not_deduplicated(self, detection_config, sample_frame, empty_frame_data):
        """Different labels should NOT be deduplicated even with overlap."""
        detector = _create_detector(detection_config)

        mock_result = MagicMock()
        mock_result.boxes = [
            _make_mock_box(0, 0.90, [100, 100, 300, 300]),  # person
            _make_mock_box(56, 0.80, [100, 100, 300, 300]), # chair
        ]
        detector._model.return_value = [mock_result]

        result = detector.detect(sample_frame, empty_frame_data)
        assert len(result.objects) == 2
