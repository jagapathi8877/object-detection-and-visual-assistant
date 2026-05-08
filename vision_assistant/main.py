"""
AI Vision Assistant -- Async Parallel Pipeline (v2.0 Rectified).

Architecture:
  Thread 1 (Camera)   : CameraStream captures frames into frame_queue(maxsize=2)
  Thread 2 (Inference): ThreadPoolExecutor runs detect+depth+direction+assistance
  Thread 3 (Audio)    : AsyncAudioFeedback runs edge-tts in asyncio loop

Frame-drop strategy: old frames are silently dropped -- blind users need
the LATEST scene, not a queue of stale frames from 2 seconds ago.

Run:
  python main.py                    # Normal mode with display
  python main.py --no-display       # Headless mode
  python main.py --lite             # Lite mode (skip depth model)
  python main.py --cam-index 1      # Use camera at index 1
"""

import argparse
import collections
import concurrent.futures
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np

from modules.audio import AudioFeedback
from modules.blind_assistance import BlindAssistanceEngine
from modules.camera import CameraStream
from modules.depth_estimator import DepthEstimator
from modules.detector import ObjectDetector
from modules.direction import assign_directions
from utils.datatypes import FrameData
from utils.helpers import load_config
from utils.logger import get_logger

logger = get_logger("main")


# ── FPS Tracker ─────────────────────────────────────────────────


class FPSTracker:
    """Tracks rolling average FPS over the last N frames.

    Attributes:
        _timestamps: Deque of frame completion timestamps.
        _window: Number of frames to average over.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: collections.deque = collections.deque(maxlen=window)
        self._window = window

    def update(self) -> None:
        """Record a frame completion timestamp."""
        self._timestamps.append(time.time())

    def get_fps(self) -> float:
        """Calculate the rolling average FPS.

        Returns:
            Current FPS as float. Returns 0.0 if insufficient data.
        """
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


# ── Frame Drop Helper ───────────────────────────────────────────


def _put_dropping(q: queue.Queue, item) -> None:
    """Put an item into a queue, dropping the oldest if full.

    Ensures we always process the LATEST frame, never a stale one.

    Args:
        q: The target queue (must have maxsize set).
        item: The item to put.
    """
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# ── Display Overlay ─────────────────────────────────────────────


_LABEL_COLORS = {
    "person": (0, 0, 255),
    "car": (0, 140, 255),
    "bus": (0, 140, 255),
    "truck": (0, 140, 255),
    "bicycle": (0, 140, 255),
    "motorcycle": (0, 140, 255),
    "chair": (255, 140, 0),
    "table": (255, 140, 0),
    "door": (255, 200, 0),
    "stairs": (0, 0, 255),
    "staircase": (0, 0, 255),
    "dog": (0, 200, 0),
}
_DEFAULT_COLOR = (180, 180, 180)


def draw_overlay(
    frame: np.ndarray,
    frame_data: FrameData,
    fps: float,
    inference_ms: float,
    target_fps: float = 15.0,
) -> np.ndarray:
    """Draw bounding boxes, FPS, and latest announcement on the frame.

    Args:
        frame: Raw BGR uint8 frame.
        frame_data: FrameData with detected objects.
        fps: Current rolling FPS.
        inference_ms: Last inference duration in ms.
        target_fps: Target FPS for colour-coding.

    Returns:
        Frame with overlays drawn.
    """
    display = frame.copy()

    for obj in frame_data.objects:
        x1, y1, x2, y2 = obj.bbox
        color = _LABEL_COLORS.get(obj.label, _DEFAULT_COLOR)

        # Urgency-based border thickness
        thickness = 3 if obj.urgency == "CRITICAL" else 2
        
        # Draw rounded rectangle (premium look)
        radius = 8
        # Top-left
        cv2.ellipse(display, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        # Top-right
        cv2.ellipse(display, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        # Bottom-right
        cv2.ellipse(display, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        # Bottom-left
        cv2.ellipse(display, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        
        # Draw straight lines connecting the corners
        cv2.line(display, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(display, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.line(display, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(display, (x1, y1 + radius), (x1, y2 - radius), color, thickness)

        # Label text
        dist = obj.smoothed_distance_m or obj.distance_m
        dist_str = f"{dist:.1f}m" if dist else "?m"
        urg_str = f"[{obj.urgency or '?'}]" if obj.urgency else ""
        label_text = f"{obj.label} {dist_str} {urg_str}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            display, label_text, (x1 + 2, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

    # FPS overlay -- top-left, colour-coded
    if fps >= target_fps:
        fps_color = (0, 255, 0)   # Green
    elif fps >= target_fps * 0.5:
        fps_color = (0, 255, 255)  # Yellow
    else:
        fps_color = (0, 0, 255)   # Red

    cv2.putText(
        display, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2,
    )

    # Inference time -- top-right
    cv2.putText(
        display, f"Inference: {inference_ms:.0f}ms",
        (display.shape[1] - 220, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    # Latest announcement -- bottom with semi-transparent background
    if frame_data.announcements:
        ann_text = frame_data.announcements[0]
        (tw, th), _ = cv2.getTextSize(ann_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_pos = display.shape[0] - 15
        # Semi-transparent background
        overlay = display.copy()
        cv2.rectangle(
            overlay, (5, y_pos - th - 8),
            (tw + 15, y_pos + 5), (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(
            display, ann_text, (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

    return display


# ── Lite Mode Distance Heuristic ────────────────────────────────


def estimate_distance_from_bbox(
    obj,
    frame_height: int,
    min_dist: float = 0.3,
    max_dist: float = 8.0,
) -> float:
    """Estimate distance from bounding box height (no depth model).

    Args:
        obj: DetectedObject with bbox set.
        frame_height: Total frame height in pixels.
        min_dist: Minimum distance estimate.
        max_dist: Maximum distance estimate.

    Returns:
        Estimated distance in metres.
    """
    _, y1, _, y2 = obj.bbox
    bbox_height = max(y2 - y1, 1)
    height_ratio = bbox_height / frame_height
    distance = max_dist - (height_ratio * (max_dist - min_dist))
    return float(np.clip(distance, min_dist, max_dist))


# ── CLI Arguments ───────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Vision Assistant v2.0 -- Real-time assistive vision",
    )
    parser.add_argument("--no-display", action="store_true", default=False,
                        help="Headless mode (no GUI window).")
    parser.add_argument("--lite", action="store_true", default=False,
                        help="Lite mode: skip depth estimation.")
    parser.add_argument("--cam-index", type=int, default=None,
                        help="Override camera device index.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file.")
    return parser.parse_args()


# ── Vision Pipeline ─────────────────────────────────────────────


class VisionPipeline:
    """Async parallel vision pipeline with frame-drop strategy.

    Thread 1: Camera -> frame_queue (maxsize=2)
    Thread 2: Inference worker (ThreadPoolExecutor)
    Thread 3: Audio worker (asyncio, daemon)

    Attributes:
        frame_queue: Camera frames waiting for inference.
        result_queue: Inference results waiting for audio.
        executor: ThreadPoolExecutor for inference.
        fps_tracker: Rolling FPS measurement.
    """

    def __init__(self, config: dict, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

        pipeline_cfg = config.get("pipeline", {})
        # Thread-safe queues
        frame_q_size = pipeline_cfg.get("frame_queue_size", 2)
        result_q_size = pipeline_cfg.get("result_queue_size", 1)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=frame_q_size)
        self.result_queue: queue.Queue = queue.Queue(maxsize=result_q_size)
        self.display_queue: queue.Queue = queue.Queue(maxsize=result_q_size)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=pipeline_cfg.get("inference_threads", 1),
            thread_name_prefix="InferenceWorker",
        )
        self.warmup_frames = pipeline_cfg.get("warmup_frames", 5)
        self.fps_tracker = FPSTracker()
        self.running = False
        self.last_inference_ms = 0.0
        self.frame_counter = 0

        # Apply CLI overrides
        if args.cam_index is not None:
            config["camera"]["index"] = args.cam_index
        self.headless = args.no_display or config["system"].get("headless", False)
        self.lite_mode = args.lite

        # Instantiate modules
        self.camera = CameraStream(config["camera"])
        self.detector = ObjectDetector(config["detection"])

        self.depth_estimator = None
        if not self.lite_mode:
            self.depth_estimator = DepthEstimator(config["depth"])
            logger.info("Depth estimation: ENABLED (MiDaS DPT-Small)")
        else:
            logger.info("Depth estimation: DISABLED (lite mode)")

        self.assistance = BlindAssistanceEngine(config)
        self.audio = AudioFeedback(config["audio"])

        self.frame_width = config["camera"]["width"]
        self.frame_height = config["camera"]["height"]
        self.direction_config = config["direction"]
        self.depth_config = config.get("depth", {})
        self.target_fps = float(config.get("pipeline", {}).get("target_fps", 15))

    def _inference_worker(self) -> None:
        """Inference thread: processes frames from frame_queue.

        Tight loop: get frame -> detect -> depth -> direction ->
        assistance -> put result. Drops stale frames.
        """
        logger.info("Inference worker started.")
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            start_time = time.time()
            
            # The processed frame is passed from the feeder
            processed_frame = getattr(frame_data, "processed_frame", None)
            if processed_frame is None:
                continue

            try:
                # Step 1: Detection (YOLOv8 expects BGR image)
                frame_data = self.detector.detect(frame_data.raw_frame, frame_data)

                # Step 2: Depth
                if self.depth_estimator is not None:
                    frame_data = self.depth_estimator.estimate(processed_frame, frame_data)
                else:
                    for obj in frame_data.objects:
                        obj.distance_m = estimate_distance_from_bbox(
                            obj, self.frame_height,
                            min_dist=self.depth_config.get("min_distance", 0.3),
                            max_dist=self.depth_config.get("max_distance", 8.0),
                        )
                        obj.smoothed_distance_m = obj.distance_m

                # Step 3: Direction
                assign_directions(frame_data, self.frame_width, self.direction_config)

                # Step 4: Blind Assistance
                frame_data = self.assistance.prioritize(frame_data)

            except Exception as exc:
                logger.error("Inference error: %s", exc, exc_info=True)

            self.last_inference_ms = (time.time() - t0) * 1000
            self.fps_tracker.update()
            self.frame_counter += 1

            # Put result to both audio and display queues
            _put_dropping(self.result_queue, frame_data)
            _put_dropping(self.display_queue, frame_data)

        logger.info("Inference worker stopped.")

    def _audio_consumer(self) -> None:
        """Audio consumer: reads results and sends to TTS.

        Runs in a separate thread, consuming from result_queue.
        """
        logger.info("Audio consumer started.")
        while self.running:
            try:
                frame_data = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Skip audio during warmup
            if self.frame_counter < self.warmup_frames:
                continue

            # Send announcements to audio
            # Map urgency to priority: CRITICAL=1, WARNING=2, INFO=5, CLEAR=8
            urgency_to_priority = {
                "CRITICAL": 1, "WARNING": 2, "INFO": 5, "CLEAR": 8,
            }
            for ann in frame_data.structured_announcements:
                priority = urgency_to_priority.get(ann.urgency, 5)
                if ann.urgency == "CRITICAL":
                    self.audio.flush()  # Clear queue for critical alerts
                self.audio.speak(ann.text, priority)

        logger.info("Audio consumer stopped.")

    def _camera_feeder(self) -> None:
        """Camera feeder: reads frames and puts into frame_queue.

        Drops old frames to ensure inference always gets the latest.
        """
        logger.info("Camera feeder started.")
        while self.running:
            try:
                raw_frame, processed_frame = self.camera.read()
                
                # Create FrameData with the raw and processed frames
                fd = FrameData(
                    raw_frame=raw_frame,
                    frame_id=self.frame_counter,  # Counter will be incremented in inference thread
                    timestamp=time.time(),
                )
                
                # We store the processed frame inside the FrameData object temporarily
                # so the inference worker can use it without re-processing
                fd.processed_frame = processed_frame 
                
                _put_dropping(self.frame_queue, fd)
            except Exception as exc:
                logger.error("Camera read error: %s", exc)
                time.sleep(0.01)
        logger.info("Camera feeder stopped.")

    def run(self) -> None:
        """Start the full pipeline and run until shutdown."""
        logger.info("=" * 60)
        logger.info("  AI Vision Assistant v2.0 -- Starting Up")
        logger.info("=" * 60)
        logger.info("Mode: %s", "LITE" if self.lite_mode else "FULL")
        logger.info("Display: %s", "OFF" if self.headless else "ON")

        # Start camera
        try:
            self.camera.start()
        except RuntimeError as exc:
            logger.error("Camera failed: %s", exc)
            self.audio.stop()
            sys.exit(1)

        self.running = True

        # Start inference thread
        inference_future = self.executor.submit(self._inference_worker)

        # Start camera feeder thread
        cam_thread = threading.Thread(
            target=self._camera_feeder,
            name="CameraFeeder",
            daemon=True,
        )
        cam_thread.start()

        # Start audio consumer thread
        audio_thread = threading.Thread(
            target=self._audio_consumer,
            name="AudioConsumer",
            daemon=True,
        )
        audio_thread.start()

        logger.info("Pipeline running. Press Ctrl+C or Q to stop.")

        # Graceful shutdown handler
        def signal_handler(sig, frame_):
            self.running = False
            logger.info("Shutdown signal received.")

        signal.signal(signal.SIGINT, signal_handler)

        # Main thread: display loop or headless logging
        try:
            last_log_time = time.time()
            
            # Show initial "Loading" window before first frame
            if not self.headless:
                loading_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                cv2.putText(loading_frame, "Loading AI Models...", (self.frame_width//2 - 150, self.frame_height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("AI Vision Assistant v2.0", loading_frame)
                cv2.waitKey(1)

            while self.running:
                if not self.headless:
                    # 1. Always get the latest raw frame for 30 FPS display
                    try:
                        raw_frame, _ = self.camera.read()
                    except RuntimeError:
                        continue
                        
                    # 2. Check if we have new inference results (non-blocking)
                    try:
                        new_fd = self.display_queue.get_nowait()
                        if new_fd is not None:
                            self.latest_display_data = new_fd
                    except queue.Empty:
                        pass

                    # 3. Draw using latest frame + latest inference data
                    display_frame = raw_frame.copy()
                    
                    if hasattr(self, 'latest_display_data') and self.latest_display_data:
                        display_frame = draw_overlay(
                            display_frame, self.latest_display_data,
                            self.fps_tracker.get_fps(),
                            self.last_inference_ms,
                            self.target_fps,
                        )
                    
                    cv2.imshow("AI Vision Assistant v2.0", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        logger.info("Quit key pressed.")
                        self.running = False
                else:
                    # Headless: periodic logging
                    time.sleep(0.1)

                # Periodic log
                now = time.time()
                if now - last_log_time >= 5.0:
                    fps = self.fps_tracker.get_fps()
                    logger.info(
                        "Frame %d | FPS: %.1f | Inference: %.0fms",
                        self.frame_counter, fps, self.last_inference_ms,
                    )
                    last_log_time = now

        except Exception as exc:
            logger.error("Main loop error: %s", exc, exc_info=True)

        # Shutdown
        self.running = False
        logger.info("Shutting down pipeline...")

        self.camera.stop()
        self.audio.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if not self.headless:
            cv2.destroyAllWindows()

        fps = self.fps_tracker.get_fps()
        logger.info(
            "Session ended. Frames: %d | Avg FPS: %.1f",
            self.frame_counter, fps,
        )
        logger.info("=" * 60)
        logger.info("  AI Vision Assistant -- Goodbye")
        logger.info("=" * 60)


# ── Entry Point ─────────────────────────────────────────────────


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    pipeline = VisionPipeline(config, args)
    pipeline.run()


if __name__ == "__main__":
    main()
