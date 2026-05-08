"""
Helper Utilities for the AI Vision Assistant.

Contains reusable utility functions for frame preprocessing, configuration
loading, and geometric calculations. Used by multiple modules to avoid
code duplication.
"""

import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML configuration file.

    Args:
        config_path: Path to the config.yaml file. Defaults to project root.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            f"Ensure config.yaml exists in the project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Basic structural validation
    required_sections = ["camera", "detection", "depth", "direction", "audio", "system"]
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required config section: '{section}'. "
                f"Check config.yaml for the complete template."
            )

    logger.info("Configuration loaded successfully from '%s'", config_path)
    return config


def preprocess_frame(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Preprocess a raw BGR camera frame for AI inference.

    Pipeline: BGR → RGB → Resize → Float32 normalisation [0, 1].

    Args:
        frame: Raw BGR frame from OpenCV (uint8, shape HxWx3).
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        Preprocessed frame as float32 numpy array with shape (height, width, 3)
        and values normalised to [0.0, 1.0].

    Raises:
        ValueError: If input frame is None or has unexpected dimensions.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot preprocess an empty or None frame.")

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel image (HxWx3), got shape {frame.shape}."
        )

    # Convert BGR (OpenCV default) to RGB (model input standard)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to target dimensions using bilinear interpolation
    resized = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # Normalise pixel values from [0, 255] uint8 to [0.0, 1.0] float32
    normalised = resized.astype(np.float32) / 255.0

    return normalised


def get_frame_center(frame: np.ndarray) -> Tuple[int, int]:
    """Calculate the center coordinates of a frame.

    Args:
        frame: Any numpy array with at least 2 dimensions (HxW or HxWxC).

    Returns:
        Tuple of (center_x, center_y) in pixel coordinates.

    Raises:
        ValueError: If frame has fewer than 2 dimensions.
    """
    if frame is None or len(frame.shape) < 2:
        raise ValueError(
            f"Frame must have at least 2 dimensions, got shape "
            f"{frame.shape if frame is not None else 'None'}."
        )

    height, width = frame.shape[:2]
    return width // 2, height // 2
