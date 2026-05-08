"""
Benchmark Logger — Per-Frame Latency Tracking and Reporting.

Records per-frame latency measurements during a pipeline session and
generates a statistical report (mean, p50, p95, p99) at session end.
Outputs to both the logger and a JSON file.

Usage:
    from utils.benchmark import BenchmarkLogger

    bench = BenchmarkLogger()
    bench.record(latency_ms=45.2)
    bench.record(latency_ms=52.1)
    # ... at session end ...
    bench.report("benchmark_report.json")
"""

import json
import os
import time
from typing import List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkLogger:
    """Records per-frame latency and generates statistical reports.

    Attributes:
        _latencies: List of per-frame latency measurements in ms.
        _start_time: Timestamp when the benchmark session started.
    """

    def __init__(self) -> None:
        """Initialise the benchmark logger."""
        self._latencies: List[float] = []
        self._start_time: float = time.time()
        logger.info("BenchmarkLogger initialised.")

    def record(self, latency_ms: float) -> None:
        """Record a single frame's end-to-end latency.

        Args:
            latency_ms: Latency in milliseconds for one pipeline cycle.
        """
        self._latencies.append(latency_ms)

    def report(self, output_path: Optional[str] = "benchmark_report.json") -> dict:
        """Generate and save a statistical benchmark report.

        Args:
            output_path: Path to save the JSON report. If None, only returns
                         the report dict without saving.

        Returns:
            Dictionary containing benchmark statistics.
        """
        if not self._latencies:
            logger.warning("No latency data recorded. Cannot generate report.")
            return {}

        arr = np.array(self._latencies)
        elapsed_s = time.time() - self._start_time

        report_data = {
            "session_duration_seconds": round(elapsed_s, 2),
            "total_frames": len(self._latencies),
            "mean_fps": round(len(self._latencies) / max(elapsed_s, 0.001), 2),
            "latency_ms": {
                "mean": round(float(np.mean(arr)), 2),
                "median_p50": round(float(np.percentile(arr, 50)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
                "std_dev": round(float(np.std(arr)), 2),
            },
        }

        # Log summary
        lat = report_data["latency_ms"]
        logger.info(
            "Benchmark Report: %d frames in %.1fs | "
            "Mean: %.1fms | P50: %.1fms | P95: %.1fms | P99: %.1fms",
            report_data["total_frames"],
            report_data["session_duration_seconds"],
            lat["mean"], lat["median_p50"], lat["p95"], lat["p99"],
        )

        # Save to JSON
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            logger.info("Benchmark report saved to '%s'", output_path)

        return report_data
