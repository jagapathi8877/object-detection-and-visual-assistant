"""
Object Prioritization Engine — Ranks and Selects Objects for Announcement.

Scores detected objects by danger/importance (weighted by type and proximity),
enforces a cooldown to prevent repetitive alerts, and builds human-readable
announcement strings for the audio module.

Design Decisions:
  - Priority formula: base_weight × (1 / max(distance, 0.3)). This naturally
    surfaces closer objects of dangerous types (person, car) over distant
    furniture. The 0.3 floor prevents division-by-zero and extreme scores.
  - Cooldown is keyed by (label + direction), e.g. "person_ahead". This means
    a person moving from left to ahead is re-announced immediately, but the
    same person staying ahead is suppressed for cooldown_seconds.
  - "Path is clear" is announced only after clear_path_interval seconds of
    silence, providing reassurance without being annoying.
  - Announcement format: "Person ahead at 2 metres" — natural spoken language.

Usage:
    from modules.prioritizer import Prioritizer
    prioritizer = Prioritizer(config)
    frame_data = prioritizer.prioritize(frame_data)
    # frame_data.announcements = ["Person ahead at 2 metres", ...]
"""

import time
from typing import Dict, List, Optional

from utils.datatypes import DetectedObject, FrameData
from utils.logger import get_logger

logger = get_logger(__name__)

# Base importance weights for object categories.
# Higher weight = more dangerous/important to announce.
# Base importance weights for ALL 80 COCO object classes.
# Organized by danger tier for visually impaired navigation.
# Higher weight = more dangerous / more important to announce first.
#
# Tier 1 (1.0):  Moving entities that can collide with or injure the user
# Tier 2 (0.85): Large stationary obstacles, vehicles, and ground hazards
# Tier 3 (0.7):  Medium-sized trip hazards and furniture
# Tier 4 (0.55): Small objects and items on surfaces
# Default (0.5): Any object not in this list (ensures nothing is ever silent)
DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    # ── Tier 1: Moving / Collision Hazards (1.0) ──────────
    "person": 1.0,
    "bicycle": 1.0,
    "car": 1.0,
    "motorcycle": 1.0,
    "bus": 1.0,
    "truck": 1.0,
    "train": 1.0,
    "dog": 0.95,
    "cat": 0.90,
    "horse": 0.95,
    "cow": 0.90,
    "elephant": 1.0,
    "bear": 1.0,
    "sheep": 0.85,
    "zebra": 0.90,
    "giraffe": 0.85,
    "bird": 0.60,

    # ── Tier 2: Large Obstacles & Street Furniture (0.85) ─
    "fire hydrant": 0.90,
    "stop sign": 0.85,
    "parking meter": 0.85,
    "bench": 0.85,
    "chair": 0.85,
    "couch": 0.85,
    "bed": 0.85,
    "dining table": 0.85,
    "toilet": 0.80,
    "potted plant": 0.80,
    "refrigerator": 0.80,
    "oven": 0.75,
    "sink": 0.75,
    "microwave": 0.70,
    "boat": 0.80,
    "airplane": 0.70,
    "traffic light": 0.80,

    # ── Tier 3: Medium Trip / Ground Hazards (0.70) ───────
    "backpack": 0.70,
    "suitcase": 0.70,
    "umbrella": 0.70,
    "handbag": 0.65,
    "skateboard": 0.75,
    "surfboard": 0.70,
    "snowboard": 0.70,
    "skis": 0.70,
    "sports ball": 0.65,
    "kite": 0.50,
    "frisbee": 0.50,
    "baseball bat": 0.70,
    "baseball glove": 0.55,
    "tennis racket": 0.60,
    "tv": 0.65,
    "laptop": 0.65,

    # ── Tier 4: Small Objects (0.55) ──────────────────────
    "bottle": 0.55,
    "wine glass": 0.55,
    "cup": 0.55,
    "fork": 0.50,
    "knife": 0.65,   # Sharp = higher priority
    "spoon": 0.45,
    "bowl": 0.50,
    "banana": 0.50,
    "apple": 0.50,
    "sandwich": 0.45,
    "orange": 0.50,
    "broccoli": 0.45,
    "carrot": 0.45,
    "hot dog": 0.45,
    "pizza": 0.45,
    "donut": 0.45,
    "cake": 0.45,
    "book": 0.50,
    "clock": 0.45,
    "vase": 0.55,
    "scissors": 0.65,  # Sharp = higher priority
    "teddy bear": 0.50,
    "hair drier": 0.50,
    "toothbrush": 0.40,
    "mouse": 0.45,
    "remote": 0.45,
    "keyboard": 0.50,
    "cell phone": 0.50,
    "toaster": 0.50,
}


class Prioritizer:
    """Ranks detected objects and builds spoken announcements.

    Scores each object by (base_weight × inverse_distance), enforces a
    per-label+direction cooldown to avoid repetitive alerts, and generates
    natural-language announcement strings.

    Attributes:
        _max_objects: Maximum number of objects to announce per cycle.
        _cooldown_seconds: Time in seconds before re-announcing same object.
        _clear_path_interval: Seconds of silence before "Path is clear."
        _base_weights: Category → importance weight mapping.
        _cooldown_tracker: Dict of (label_direction) → last_announced_time.
        _last_announcement_time: Timestamp of the most recent announcement.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the prioritizer from config.

        Args:
            config: Full config dict (needs 'detection' and 'audio' sections).

        Raises:
            ValueError: If required config sections/keys are missing.
        """
        if "detection" not in config:
            raise ValueError(
                "Missing 'detection' section in config for Prioritizer."
            )
        if "audio" not in config:
            raise ValueError(
                "Missing 'audio' section in config for Prioritizer."
            )

        detection_cfg = config["detection"]
        audio_cfg = config["audio"]

        self._max_objects: int = detection_cfg.get("max_objects", 3)
        self._cooldown_seconds: float = float(
            audio_cfg.get("cooldown_seconds", 3.0)
        )
        self._clear_path_interval: float = float(
            audio_cfg.get("clear_path_interval", 5.0)
        )

        self._base_weights: Dict[str, float] = DEFAULT_BASE_WEIGHTS.copy()
        self._cooldown_tracker: Dict[str, float] = {}
        self._last_announcement_time: float = 0.0

        logger.info(
            "Prioritizer initialised: max_objects=%d, cooldown=%.1fs, "
            "clear_path_interval=%.1fs",
            self._max_objects, self._cooldown_seconds,
            self._clear_path_interval,
        )

    def prioritize(self, frame_data: FrameData) -> FrameData:
        """Score, rank, filter, and build announcements for detected objects.

        Args:
            frame_data: FrameData with objects populated (must have
                        distance_m and direction set).

        Returns:
            Updated FrameData with:
              - objects sorted by priority_score (descending)
              - priority_score set on each object
              - announcements list populated with spoken strings
        """
        now = time.time()
        frame_data.announcements = []

        # Step 1: Score each object
        for obj in frame_data.objects:
            obj.priority_score = self._compute_score(obj)

        # Step 2: Sort by priority (highest first)
        frame_data.objects.sort(
            key=lambda o: o.priority_score or 0.0,
            reverse=True,
        )

        # Step 3: Take top N objects
        top_objects = frame_data.objects[:self._max_objects]

        # Step 4: Apply cooldown filter
        announcements: List[str] = []
        for obj in top_objects:
            cooldown_key = self._make_cooldown_key(obj)

            # Check if this object+direction was announced recently
            last_time = self._cooldown_tracker.get(cooldown_key, 0.0)
            if (now - last_time) < self._cooldown_seconds:
                logger.debug(
                    "Cooldown active for '%s' — skipping.",
                    cooldown_key,
                )
                continue

            # Build announcement string
            announcement = self._format_announcement(obj)
            announcements.append(announcement)

            # Update cooldown tracker
            self._cooldown_tracker[cooldown_key] = now

        # Step 5: "Path is clear" fallback
        if not announcements and not frame_data.objects:
            elapsed = now - self._last_announcement_time
            if elapsed >= self._clear_path_interval:
                announcements.append("Path is clear")

        # Update timing
        if announcements:
            self._last_announcement_time = now

        frame_data.announcements = announcements

        logger.debug(
            "Frame %d: %d announcements from %d objects",
            frame_data.frame_id, len(announcements), len(frame_data.objects),
        )

        return frame_data

    def _compute_score(self, obj: DetectedObject) -> float:
        """Compute priority score for a single detected object.

        Formula: base_weight × (1.0 / max(distance_m, 0.3))
        Closer objects of dangerous types score highest.

        Args:
            obj: DetectedObject with label and distance_m set.

        Returns:
            Priority score as float (higher = more urgent).
        """
        base_weight = self._base_weights.get(obj.label, 0.5)
        distance = max(obj.distance_m or 10.0, 0.3)
        return base_weight * (1.0 / distance)

    def _make_cooldown_key(self, obj: DetectedObject) -> str:
        """Create a cooldown dictionary key from object label + direction.

        Args:
            obj: DetectedObject with label and direction set.

        Returns:
            Key string like "person_ahead" or "chair_left".
        """
        direction = obj.direction or "unknown"
        return f"{obj.label}_{direction}"

    @staticmethod
    def _format_announcement(obj: DetectedObject) -> str:
        """Build a human-readable spoken announcement string.

        Examples:
          - "Person ahead at 2 metres"
          - "Chair on the left at 1 metre"
          - "Car on the right at 3 metres"

        Args:
            obj: DetectedObject with label, direction, distance_m set.

        Returns:
            Formatted announcement string.
        """
        label = obj.label.capitalize()
        distance = round(obj.distance_m or 0, 0)
        distance_int = int(distance)
        metre_word = "metre" if distance_int == 1 else "metres"

        direction = obj.direction or "ahead"
        if direction == "ahead":
            dir_phrase = "ahead"
        elif direction == "left":
            dir_phrase = "on the left"
        elif direction == "right":
            dir_phrase = "on the right"
        else:
            dir_phrase = direction

        return f"{label} {dir_phrase} at {distance_int} {metre_word}"
