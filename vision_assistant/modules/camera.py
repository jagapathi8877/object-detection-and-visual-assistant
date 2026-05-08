"""
Camera Input Module — Threaded Frame Capture for the AI Vision Assistant.

Provides a production-grade, thread-safe camera capture class that delivers
preprocessed frames to the inference pipeline at a stable FPS. The capture
runs in its own background thread so that frame I/O never blocks AI inference.

Design Decisions:
  - Uses collections.deque(maxlen=1) as the frame buffer. This ensures we
    always process the LATEST frame (no stale queue buildup). Older frames
    are automatically discarded.
  - Preprocessing (BGR→RGB, resize, normalise) happens in the capture thread
    to keep the inference loop focused on AI work only.
  - FPS is logged every 5 seconds so developers can monitor capture health
    without flooding the console.
  - Resource cleanup is guaranteed via stop() which signals the thread,
    joins with a 2-second timeout, and releases the cv2.VideoCapture.

Usage:
    from modules.camera import CameraStream
    from utils.helpers import load_config

    config = load_config()
    camera = CameraStream(config["camera"])
    camera.start()

    frame = camera.read()   # Returns latest preprocessed frame (blocks until ready)
    # ... process frame ...

    camera.stop()            # Clean shutdown
"""

import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from utils.helpers import preprocess_frame
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraStream:
    """Threaded camera capture with single-frame buffer.

    Captures frames from a webcam (or video file) in a background thread
    and stores only the latest preprocessed frame in a thread-safe buffer.

    Attributes:
        _config: Camera configuration dictionary from config.yaml.
        _cap: OpenCV VideoCapture instance.
        _buffer: deque(maxlen=1) holding the latest preprocessed frame.
        _thread: Background capture thread.
        _running: Threading event to signal start/stop.
        _frame_ready: Threading event signalling at least one frame is available.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the camera stream.

        Args:
            config: Camera section from config.yaml containing:
                    index (int), width (int), height (int), fps (int).

        Raises:
            ValueError: If required config keys are missing.
        """
        required_keys = ["index", "width", "height", "fps"]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Missing required camera config key: '{key}'. "
                    f"Check the 'camera' section in config.yaml."
                )

        self._config = config
        self._cam_index: int = config["index"]
        self._width: int = config["width"]
        self._height: int = config["height"]
        self._target_fps: int = config["fps"]

        # Thread-safe single-frame buffer — always holds the latest frame only.
        # deque(maxlen=1) automatically discards the old frame when a new one
        # is appended, eliminating stale-frame buildup without manual locking.
        self._buffer: deque = deque(maxlen=1)

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._frame_ready = threading.Event()

        # FPS tracking for health monitoring
        self._frame_count: int = 0
        self._fps_log_interval: float = 5.0  # Log FPS every 5 seconds

        logger.info(
            "CameraStream initialised: index=%d, target=%dx%d @ %d FPS",
            self._cam_index, self._width, self._height, self._target_fps,
        )

    def start(self) -> None:
        """Start the background capture thread.

        Opens the camera device and begins capturing frames. Returns
        immediately — capture runs in a daemon thread.

        Raises:
            RuntimeError: If the camera device cannot be opened.
        """
        if self._running.is_set():
            logger.warning("CameraStream.start() called but already running.")
            return

        # Open the camera device using DirectShow (CAP_DSHOW) on Windows for better FPS
        import sys
        if sys.platform == 'win32':
            self._cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._cam_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._cam_index}. "
                f"Check that a camera is connected and the index is correct."
            )

        # Attempt to set camera properties (not all cameras support this)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        # Read back actual values (camera may not honour our requests)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera opened: actual resolution=%dx%d, reported FPS=%.1f",
            actual_w, actual_h, actual_fps,
        )

        # Start the capture thread
        self._running.set()
        self._frame_count = 0
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,  # Daemon so it doesn't prevent process exit
        )
        self._thread.start()
        logger.info("Camera capture thread started.")

    def _capture_loop(self) -> None:
        """Background capture loop — runs in its own thread.

        Continuously reads frames from the camera, preprocesses them,
        and places them in the single-frame buffer. Logs FPS every
        _fps_log_interval seconds for health monitoring.
        """
        fps_timer = time.time()
        frames_since_log = 0

        while self._running.is_set():
            ret, raw_frame = self._cap.read()

            if not ret:
                logger.warning(
                    "Camera read() returned False — frame dropped. "
                    "Camera may have been disconnected."
                )
                # Brief sleep to avoid tight-looping on a dead camera
                time.sleep(0.1)
                continue

            # Preprocess: BGR → RGB, resize, normalise to [0, 1] float32
            try:
                processed = preprocess_frame(
                    raw_frame, self._width, self._height
                )
            except ValueError as exc:
                logger.error("Frame preprocessing failed: %s", exc)
                continue

            # Place in the single-frame buffer (old frame auto-discarded)
            self._buffer.append((raw_frame, processed))
            self._frame_ready.set()  # Signal that at least one frame exists

            self._frame_count += 1
            frames_since_log += 1

            # Log FPS every 5 seconds
            elapsed = time.time() - fps_timer
            if elapsed >= self._fps_log_interval:
                measured_fps = frames_since_log / elapsed
                logger.info(
                    "Camera FPS: %.1f (frames captured: %d total)",
                    measured_fps, self._frame_count,
                )
                fps_timer = time.time()
                frames_since_log = 0

        logger.info(
            "Capture loop exited. Total frames captured: %d",
            self._frame_count,
        )

    def read(self) -> np.ndarray:
        """Return the latest preprocessed frame.

        Blocks until at least one frame is available in the buffer.
        After the first frame, this always returns immediately with
        the most recent frame.

        Returns:
            Tuple of (raw_frame, processed_frame):
            - raw_frame: np.ndarray (uint8, shape HxWx3, BGR)
            - processed_frame: np.ndarray (float32, shape HxWx3, RGB, values in [0.0, 1.0])

        Raises:
            RuntimeError: If the camera stream is not running.
        """
        if not self._running.is_set():
            raise RuntimeError(
                "CameraStream is not running. Call start() first."
            )

        # Block until the first frame is ready (with 10s timeout)
        if not self._frame_ready.wait(timeout=10.0):
            raise RuntimeError(
                "Timed out waiting for first frame from camera. "
                "Check camera connection."
            )

        # Return the latest frame from the buffer.
        # deque with maxlen=1 always has exactly 0 or 1 items.
        return self._buffer[-1]

    def read_raw(self) -> Optional[np.ndarray]:
        """(Deprecated) read() now returns the raw frame directly from buffer."""
        pass

    def stop(self) -> None:
        """Stop the capture thread and release camera resources.

        Signals the background thread to exit, waits up to 2 seconds
        for it to join, then releases the OpenCV VideoCapture device.
        Safe to call multiple times.
        """
        if not self._running.is_set():
            logger.debug("CameraStream.stop() called but not running.")
            return

        logger.info("Stopping camera stream...")
        self._running.clear()  # Signal the capture loop to exit

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "Camera thread did not terminate within 2s timeout."
                )
            else:
                logger.info("Camera capture thread joined successfully.")

        # Release the camera device
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera device released.")

        self._frame_ready.clear()

    def is_running(self) -> bool:
        """Check if the capture thread is currently active.

        Returns:
            True if the background capture thread is alive and running.
        """
        return self._running.is_set() and (
            self._thread is not None and self._thread.is_alive()
        )

    @property
    def frame_count(self) -> int:
        """Total number of frames captured since start()."""
        return self._frame_count

    def __enter__(self):
        """Context manager support — auto-start on enter."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support — auto-stop on exit."""
        self.stop()
        return False
