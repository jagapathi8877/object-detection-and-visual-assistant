"""
Integration Tests for the Full Pipeline (v2.0 Rectified).

Tests the complete pipeline flow with mocked hardware:
  Camera -> Detector -> Depth -> Direction -> BlindAssistance -> Audio

Also tests:
  - BenchmarkLogger statistics
  - Lite mode (skip depth, use bbox heuristic)
  - All DetectedObjects have valid distance/direction/urgency
  - No unhandled exceptions across multiple frames

Run:  pytest tests/test_integration.py -v
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.datatypes import DetectedObject, FrameData
from utils.benchmark import BenchmarkLogger
from modules.direction import calculate_direction, assign_directions
from modules.blind_assistance import BlindAssistanceEngine


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def full_config() -> dict:
    """Full config matching v2.0 config.yaml structure."""
    return {
        "camera": {"index": 0, "width": 640, "height": 480, "fps": 30},
        "detection": {
            "model": "yolov8n.pt",
            "world_model": "yolov8x-worldv2.pt",
            "vocabulary_preset": "blind_navigation",
            "conf_threshold": 0.35,
            "iou_threshold": 0.45,
            "max_detections": 10,
            "agnostic_nms": True,
            "max_objects": 5,
        },
        "depth": {
            "model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            "use_gpu": "false",
            "min_distance": 0.3,
            "max_distance": 8.0,
            "approaching_threshold_m_per_s": 0.4,
        },
        "direction": {"left_boundary": 0.33, "right_boundary": 0.66},
        "audio": {
            "engine": "edge-tts",
            "voice": "en-IN-NeerjaNeural",
            "cooldown": {
                "critical_seconds": 1.5,
                "warning_seconds": 3.0,
                "info_seconds": 6.0,
                "clear_seconds": 10.0,
            },
            "queue_max_depth": 3,
        },
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
        "pipeline": {
            "frame_queue_size": 2,
            "result_queue_size": 1,
            "inference_threads": 1,
            "warmup_frames": 5,
        },
        "system": {
            "log_level": "INFO",
            "headless": True,
            "target_latency_ms": 500,
        },
    }


def make_frame() -> np.ndarray:
    """Create a synthetic frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ============================================================
# Test 1: Full Pipeline Flow (Mocked)
# ============================================================
class TestPipelineFlow:
    """Test the complete pipeline with manual module invocations."""

    def test_full_pipeline_produces_announcements(self, full_config) -> None:
        """A detected person should produce a spoken announcement string."""
        person = DetectedObject(
            label="person",
            bbox=(250, 200, 390, 400),
            confidence=0.92,
            distance_m=1.0,
            smoothed_distance_m=1.0,
        )
        fd = FrameData(
            raw_frame=make_frame(),
            frame_id=0,
            timestamp=time.time(),
            objects=[person],
        )

        assign_directions(fd, 640, full_config["direction"])
        engine = BlindAssistanceEngine(full_config)
        fd = engine.prioritize(fd)

        assert len(fd.announcements) >= 1
        # Should contain Warning or person reference
        combined = " ".join(fd.announcements).lower()
        assert "person" in combined or "warning" in combined

    def test_multiple_objects_critical_first(self, full_config) -> None:
        """CRITICAL person should be announced before INFO chair."""
        person = DetectedObject(
            label="person", bbox=(270, 200, 370, 400), confidence=0.9,
            distance_m=1.0, smoothed_distance_m=1.0,
        )
        chair = DetectedObject(
            label="chair", bbox=(50, 200, 150, 400), confidence=0.8,
            distance_m=6.0, smoothed_distance_m=6.0,
        )
        fd = FrameData(
            raw_frame=make_frame(), frame_id=0, timestamp=time.time(),
            objects=[chair, person],
        )

        assign_directions(fd, 640, full_config["direction"])
        engine = BlindAssistanceEngine(full_config)
        fd = engine.prioritize(fd)

        # Person should be first (CRITICAL)
        assert fd.objects[0].label == "person"
        assert fd.objects[0].urgency == "CRITICAL"

    def test_all_objects_have_valid_direction(self, full_config) -> None:
        """Every object should have direction in {left, ahead, right}."""
        objects = [
            DetectedObject(label="person", bbox=(10, 10, 50, 200),
                           confidence=0.9, distance_m=2.0, smoothed_distance_m=2.0),
            DetectedObject(label="car", bbox=(300, 50, 400, 200),
                           confidence=0.85, distance_m=5.0, smoothed_distance_m=5.0),
            DetectedObject(label="dog", bbox=(550, 100, 620, 300),
                           confidence=0.7, distance_m=3.0, smoothed_distance_m=3.0),
        ]
        fd = FrameData(
            raw_frame=make_frame(), frame_id=0, timestamp=time.time(),
            objects=objects,
        )

        assign_directions(fd, 640, full_config["direction"])

        valid_directions = {"left", "ahead", "right"}
        for obj in fd.objects:
            assert obj.direction in valid_directions

    def test_all_objects_get_urgency_and_zone(self, full_config) -> None:
        """Every object should get urgency and zone after prioritize()."""
        objects = [
            DetectedObject(label="person", bbox=(250, 200, 390, 400),
                           confidence=0.9, distance_m=1.0, smoothed_distance_m=1.0),
            DetectedObject(label="chair", bbox=(50, 300, 150, 450),
                           confidence=0.8, distance_m=4.0, smoothed_distance_m=4.0),
        ]
        fd = FrameData(
            raw_frame=make_frame(), frame_id=0, timestamp=time.time(),
            objects=objects,
        )

        assign_directions(fd, 640, full_config["direction"])
        engine = BlindAssistanceEngine(full_config)
        fd = engine.prioritize(fd)

        valid_urgencies = {"CRITICAL", "WARNING", "INFO"}
        valid_zones = {"DEAD_AHEAD", "LEFT_NEAR", "RIGHT_NEAR", "LEFT_FAR", "RIGHT_FAR"}

        for obj in fd.objects:
            assert obj.urgency in valid_urgencies
            assert obj.zone in valid_zones

    def test_no_exceptions_across_100_frames(self, full_config) -> None:
        """Pipeline should process 100 simulated frames without crashing."""
        engine = BlindAssistanceEngine(full_config)

        for i in range(100):
            num_objects = np.random.randint(0, 4)
            objects = []
            for j in range(num_objects):
                x1 = np.random.randint(0, 500)
                y1 = np.random.randint(0, 300)
                dist = np.random.uniform(0.3, 8.0)
                objects.append(DetectedObject(
                    label=np.random.choice(["person", "chair", "car", "dog"]),
                    bbox=(x1, y1, x1 + 100, y1 + 150),
                    confidence=np.random.uniform(0.5, 1.0),
                    distance_m=dist,
                    smoothed_distance_m=dist,
                ))

            fd = FrameData(
                raw_frame=make_frame(), frame_id=i, timestamp=time.time(),
                objects=objects,
            )

            assign_directions(fd, 640, full_config["direction"])
            fd = engine.prioritize(fd)

            assert isinstance(fd, FrameData)


# ============================================================
# Test 2: Lite Mode (Bbox Heuristic)
# ============================================================
class TestLiteMode:
    """Test the bbox-height distance heuristic used in --lite mode."""

    def test_larger_bbox_means_closer(self) -> None:
        """A taller bounding box should estimate a shorter distance."""
        from main import estimate_distance_from_bbox

        close_obj = DetectedObject(
            label="person", bbox=(100, 50, 300, 450),
            confidence=0.9,
        )
        far_obj = DetectedObject(
            label="person", bbox=(100, 200, 300, 250),
            confidence=0.9,
        )

        d_close = estimate_distance_from_bbox(close_obj, 480)
        d_far = estimate_distance_from_bbox(far_obj, 480)

        assert d_close < d_far

    def test_distance_within_bounds(self) -> None:
        """Heuristic distance should be clamped to [min, max]."""
        from main import estimate_distance_from_bbox

        obj = DetectedObject(
            label="person", bbox=(0, 0, 640, 480),
            confidence=0.9,
        )

        dist = estimate_distance_from_bbox(obj, 480, min_dist=0.3, max_dist=8.0)
        assert 0.3 <= dist <= 8.0


# ============================================================
# Test 3: BenchmarkLogger
# ============================================================
class TestBenchmarkLogger:
    """Test the BenchmarkLogger utility."""

    def test_record_and_report(self) -> None:
        """Should produce valid stats after recording latencies."""
        bench = BenchmarkLogger()
        latencies = [30, 45, 50, 55, 60, 100, 200, 300, 400, 500]
        for lat in latencies:
            bench.record(lat)

        report = bench.report(output_path=None)

        assert report["total_frames"] == 10
        assert report["latency_ms"]["mean"] > 0
        assert report["latency_ms"]["median_p50"] > 0
        assert report["latency_ms"]["p95"] > 0
        assert report["latency_ms"]["p99"] > 0
        assert report["latency_ms"]["min"] == 30
        assert report["latency_ms"]["max"] == 500

    def test_report_saves_json(self, tmp_path) -> None:
        """Report should save a valid JSON file."""
        bench = BenchmarkLogger()
        bench.record(50.0)
        bench.record(60.0)

        output = str(tmp_path / "test_benchmark.json")
        report = bench.report(output_path=output)

        assert os.path.exists(output)
        import json
        with open(output, "r") as f:
            loaded = json.load(f)
        assert loaded["total_frames"] == 2

    def test_empty_report_returns_empty(self) -> None:
        """Report with no data should return empty dict."""
        bench = BenchmarkLogger()
        report = bench.report(output_path=None)
        assert report == {}

    def test_p95_and_p99_ordering(self) -> None:
        """p95 should be less than or equal to p99."""
        bench = BenchmarkLogger()
        for i in range(100):
            bench.record(float(i))

        report = bench.report(output_path=None)
        assert report["latency_ms"]["p95"] <= report["latency_ms"]["p99"]
