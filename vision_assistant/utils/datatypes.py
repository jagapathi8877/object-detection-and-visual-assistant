"""
Shared Data Contracts for the AI Vision Assistant Pipeline.

All modules communicate via FrameData and DetectedObject dataclasses defined
here. This ensures a consistent interface across the entire pipeline:
Camera -> Detector -> Depth -> Direction -> BlindAssistance -> Audio.

Design Decision:
  We use Python dataclasses with field defaults so modules can incrementally
  populate fields without needing to know about downstream concerns. For
  example, the Detector fills label/bbox/confidence but leaves distance_m
  and direction for later modules.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DetectedObject:
    """Represents a single detected object with all pipeline-enriched metadata.

    Attributes:
        label: Human-readable class name (e.g. 'person', 'chair').
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence score in [0.0, 1.0].
        distance_m: Estimated distance in metres (filled by DepthEstimator).
        direction: Spatial direction -- 'left', 'ahead', or 'right'
                   (filled by direction module).
        priority_score: Computed priority for announcement ordering
                        (filled by BlindAssistanceEngine).
        distance_confidence: 'high' or 'low' -- based on bbox size vs frame.
        is_approaching: True if object is getting closer frame-over-frame.
        prev_distance_m: Distance from the previous frame (for velocity calc).
        smoothed_distance_m: EMA-smoothed distance (used for announcements).
        urgency: Urgency tier -- 'CRITICAL', 'WARNING', 'INFO', or 'CLEAR'.
        zone: Spatial zone -- 'DEAD_AHEAD', 'LEFT_NEAR', 'RIGHT_NEAR',
              'LEFT_FAR', 'RIGHT_FAR'.
        approach_velocity: Rate of approach in m/s (positive = getting closer).
    """

    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    distance_m: Optional[float] = None
    direction: Optional[str] = None
    priority_score: Optional[float] = None
    distance_confidence: str = "high"
    is_approaching: bool = False
    prev_distance_m: Optional[float] = None
    smoothed_distance_m: Optional[float] = None
    urgency: Optional[str] = None
    zone: Optional[str] = None
    approach_velocity: float = 0.0

    def __repr__(self) -> str:
        """Concise representation for logging."""
        dist = f"{self.distance_m:.1f}m" if self.distance_m is not None else "?m"
        dirn = self.direction or "?"
        return (
            f"DetectedObject(label={self.label!r}, conf={self.confidence:.2f}, "
            f"dist={dist}, dir={dirn}, score={self.priority_score})"
        )


@dataclass
class Announcement:
    """A single spoken announcement with urgency metadata.

    Attributes:
        text: The spoken text (e.g. "Warning! Person directly ahead, 1 metre").
        urgency: Urgency tier -- 'CRITICAL', 'WARNING', 'INFO', or 'CLEAR'.
        obj: The DetectedObject this announcement refers to (None for CLEAR).
    """

    text: str
    urgency: str = "INFO"
    obj: Optional[DetectedObject] = None


@dataclass
class FrameData:
    """Container for all data associated with a single camera frame.

    This dataclass flows through the entire pipeline. Each module reads
    what it needs and writes its results back onto the same instance.

    Attributes:
        raw_frame: Original BGR frame captured from the camera as np.ndarray.
        objects: List of DetectedObject instances found in this frame.
        announcements: Final audio strings to be spoken.
        structured_announcements: Announcement objects with urgency metadata.
        frame_id: Monotonically increasing frame counter (0-based).
        timestamp: time.time() value at the moment of frame capture.
    """

    raw_frame: np.ndarray
    frame_id: int
    timestamp: float
    objects: List[DetectedObject] = field(default_factory=list)
    announcements: List[str] = field(default_factory=list)
    structured_announcements: List[Announcement] = field(default_factory=list)

    def __repr__(self) -> str:
        """Concise representation for logging."""
        shape = self.raw_frame.shape if self.raw_frame is not None else "None"
        return (
            f"FrameData(id={self.frame_id}, ts={self.timestamp:.3f}, "
            f"shape={shape}, objects={len(self.objects)}, "
            f"announcements={len(self.announcements)})"
        )
