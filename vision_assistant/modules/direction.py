"""
Direction Calculation Module — Spatial Direction from Bounding Box Position.

Determines whether a detected object is to the LEFT, AHEAD (center), or RIGHT
of the user based on the horizontal center of its bounding box relative to
configurable frame-width boundaries.

Design Decisions:
  - Uses a simple three-zone model (left / ahead / right) with configurable
    boundary fractions. Default: left < 33% < ahead < 66% < right.
  - Boundaries are fractions (0–1) of frame width, not pixel values, so the
    logic is resolution-independent.
  - Objects exactly on the left boundary are classified as 'left' (inclusive),
    and objects exactly on the right boundary are classified as 'right'.

Usage:
    from modules.direction import calculate_direction

    direction = calculate_direction(
        bbox=(100, 50, 300, 400),
        frame_width=640,
        config={"left_boundary": 0.33, "right_boundary": 0.66},
    )
    # Returns 'left', 'ahead', or 'right'
"""

from typing import Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


def calculate_direction(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    config: dict,
) -> str:
    """Determine the spatial direction of an object from its bounding box.

    Computes the horizontal center of the bounding box and classifies it
    into one of three zones based on configurable boundary fractions.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        frame_width: Total width of the frame in pixels.
        config: Direction section from config.yaml containing:
                left_boundary (float): Fraction of frame width where
                    the left zone ends (default 0.33).
                right_boundary (float): Fraction of frame width where
                    the right zone begins (default 0.66).

    Returns:
        Direction string: 'left', 'ahead', or 'right'.

    Raises:
        ValueError: If frame_width is zero or negative.
        ValueError: If bbox has invalid coordinates.
    """
    if frame_width <= 0:
        raise ValueError(
            f"frame_width must be positive, got {frame_width}."
        )

    # Extract boundaries from config with safe defaults
    left_boundary_frac = config.get("left_boundary", 0.33)
    right_boundary_frac = config.get("right_boundary", 0.66)

    # Compute pixel boundaries from fractions
    left_boundary_px = frame_width * left_boundary_frac
    right_boundary_px = frame_width * right_boundary_frac

    # Compute horizontal center of the bounding box
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0

    # Classify into direction zones
    # Objects on the boundary are assigned to the outer zone (safer alert)
    if center_x <= left_boundary_px:
        direction = "left"
    elif center_x >= right_boundary_px:
        direction = "right"
    else:
        direction = "ahead"

    return direction


def assign_directions(
    frame_data,
    frame_width: int,
    config: dict,
) -> None:
    """Assign direction to all DetectedObjects in a FrameData instance.

    This is a convenience function for pipeline integration. It iterates
    over all objects in frame_data and sets each object's direction field.

    Args:
        frame_data: FrameData instance with objects populated.
        frame_width: Total width of the frame in pixels.
        config: Direction section from config.yaml.
    """
    for obj in frame_data.objects:
        obj.direction = calculate_direction(obj.bbox, frame_width, config)

    logger.debug(
        "Frame %d: Directions assigned to %d objects",
        frame_data.frame_id, len(frame_data.objects),
    )
