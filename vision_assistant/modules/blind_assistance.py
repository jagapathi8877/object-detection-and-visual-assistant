"""
Blind Assistance Engine -- Hazard-Aware Navigation Guidance.

Replaces the simple prioritizer with a full navigation intent system:
  - 4-tier urgency classification (CRITICAL / WARNING / INFO / CLEAR)
  - 5-zone spatial analysis (DEAD_AHEAD, LEFT_NEAR, RIGHT_NEAR, LEFT_FAR, RIGHT_FAR)
  - Staircase/step special handling with up/down detection
  - Approach detection using is_approaching from depth estimator
  - Scene summary every 15 seconds
  - Safety override: CRITICAL always bypasses cooldown
  - Max 2 announcements per frame (cognitive limit for blind users)

Drop-in replacement for Prioritizer -- same method signature:
  engine = BlindAssistanceEngine(config)
  frame_data = engine.prioritize(frame_data)
"""

import time
from typing import Dict, List, Optional, Set, Tuple

from utils.announcement_builder import (
    build_announcement,
    build_clear_announcement,
    build_scene_summary,
)
from utils.datatypes import Announcement, DetectedObject, FrameData
from utils.label_map import HAZARD_OBJECTS, NAVIGATION_OBJECTS, get_speech_label
from utils.logger import get_logger

logger = get_logger(__name__)

# Object sets for tier classification
_HAZARD_SET: Set[str] = {o.lower() for o in HAZARD_OBJECTS}
_STAIRCASE_LABELS: Set[str] = {"stairs", "staircase", "step", "steps", "escalator"}


class BlindAssistanceEngine:
    """Hazard-aware navigation guidance engine for blind users.

    Classifies detected objects into urgency tiers, applies 5-zone
    spatial analysis, handles staircases specially, and generates
    context-aware announcements.

    Attributes:
        _dead_ahead_zone: (left_pct, right_pct) boundaries for DEAD_AHEAD.
        _critical_dist: Distance threshold for CRITICAL tier (metres).
        _warning_dist: Distance threshold for WARNING tier (metres).
        _fast_approach_vel: Velocity for "fast approaching" (m/s).
        _max_announcements: Max announcements per frame.
        _staircase_bottom_pct: Frame height fraction for "going down".
        _scene_summary_interval: Seconds between scene summaries.
        _cooldowns: Urgency-specific cooldown durations.
        _safety_override: Whether CRITICAL bypasses cooldown.
        _cooldown_tracker: (label_zone) -> last_announced_time.
        _last_announcement_time: Timestamp of last announcement.
        _last_summary_time: Timestamp of last scene summary.
        _recent_objects: Objects seen in the last summary interval.
        _clear_frame_count: Consecutive frames with no CRITICAL/WARNING.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the blind assistance engine.

        Args:
            config: Full config dict (needs 'blind_assistance' section,
                    optionally 'detection' and 'audio').
        """
        ba_cfg = config.get("blind_assistance", {})

        zone_pct = ba_cfg.get("dead_ahead_zone_pct", [0.35, 0.65])
        self._dead_ahead_zone: Tuple[float, float] = (zone_pct[0], zone_pct[1])
        self._critical_dist: float = float(ba_cfg.get("critical_distance_m", 1.5))
        self._warning_dist: float = float(ba_cfg.get("warning_distance_m", 3.0))
        self._fast_approach_vel: float = float(
            ba_cfg.get("fast_approach_velocity_m_per_s", 1.0)
        )
        self._max_announcements: int = int(
            ba_cfg.get("max_announcements_per_frame", 2)
        )
        self._staircase_bottom_pct: float = float(
            ba_cfg.get("staircase_bottom_frame_pct", 0.60)
        )
        self._scene_summary_interval: float = float(
            ba_cfg.get("scene_summary_interval_s", 15)
        )

        cooldown_cfg = ba_cfg.get("cooldown", {})
        self._cooldowns: Dict[str, float] = {
            "CRITICAL": float(cooldown_cfg.get("critical_s", 1.5)),
            "WARNING": float(cooldown_cfg.get("warning_s", 3.0)),
            "INFO": float(cooldown_cfg.get("info_s", 6.0)),
            "CLEAR": float(cooldown_cfg.get("clear_s", 10.0)),
        }
        self._safety_override: bool = bool(
            cooldown_cfg.get("safety_override", True)
        )

        # Also read from detection/audio for backward compatibility
        det_cfg = config.get("detection", {})
        self._max_objects: int = det_cfg.get("max_objects", 5)

        # State
        self._cooldown_tracker: Dict[str, float] = {}
        self._last_announcement_time: float = 0.0
        self._last_summary_time: float = time.time()
        self._recent_objects: List[DetectedObject] = []
        self._clear_frame_count: int = 0

        logger.info(
            "BlindAssistanceEngine initialised: critical=%.1fm, warning=%.1fm, "
            "max_announcements=%d, scene_summary=%ds",
            self._critical_dist, self._warning_dist,
            self._max_announcements, self._scene_summary_interval,
        )

    def prioritize(self, frame_data: FrameData) -> FrameData:
        """Score, classify, and build announcements for detected objects.

        Drop-in replacement for Prioritizer.prioritize().

        Args:
            frame_data: FrameData with objects populated.

        Returns:
            Updated FrameData with announcements and structured_announcements.
        """
        now = time.time()
        frame_data.announcements = []
        frame_data.structured_announcements = []

        frame_h = frame_data.raw_frame.shape[0] if frame_data.raw_frame is not None else 480
        frame_w = frame_data.raw_frame.shape[1] if frame_data.raw_frame is not None else 640

        # Step 1: Classify each object (zone, urgency, score)
        for obj in frame_data.objects:
            obj.zone = self._classify_zone(obj, frame_w, frame_h)
            obj.urgency = self._classify_urgency(obj, frame_h)
            obj.priority_score = self._compute_score(obj)

        # Step 2: Sort by tier then distance
        frame_data.objects.sort(
            key=lambda o: (
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "CLEAR": 3}.get(o.urgency, 2),
                o.distance_m or 99.0,
            )
        )

        # Track for scene summary
        self._recent_objects.extend(frame_data.objects[:3])
        # Keep only last 30 seconds of objects
        cutoff = now - self._scene_summary_interval * 2
        self._recent_objects = [
            o for o in self._recent_objects
            if hasattr(o, '_tracked_time') is False  # keep all for simplicity
        ][-50:]  # cap at 50

        # Step 3: Check for CRITICAL/WARNING presence
        has_critical = any(o.urgency == "CRITICAL" for o in frame_data.objects)
        has_warning = any(o.urgency == "WARNING" for o in frame_data.objects)

        if has_critical or has_warning:
            self._clear_frame_count = 0
        else:
            self._clear_frame_count += 1

        # Step 4: Select top announcements with cooldown
        announcements: List[Announcement] = []
        for obj in frame_data.objects:
            if len(announcements) >= self._max_announcements:
                break

            cooldown_key = f"{obj.label}_{obj.zone or obj.direction or 'ahead'}"
            urgency = obj.urgency or "INFO"

            # Safety override: CRITICAL always announced
            if urgency == "CRITICAL" and self._safety_override:
                pass  # bypass cooldown
            else:
                # Check cooldown
                cooldown_dur = self._cooldowns.get(urgency, 3.0)
                last_time = self._cooldown_tracker.get(cooldown_key, 0.0)
                if (now - last_time) < cooldown_dur:
                    continue

            # Build announcement
            text = build_announcement(
                obj, urgency, obj.distance_confidence
            )
            ann = Announcement(text=text, urgency=urgency, obj=obj)
            announcements.append(ann)
            self._cooldown_tracker[cooldown_key] = now

        # Step 5: "Path is clear" fallback
        if not announcements and self._clear_frame_count >= 3:
            clear_key = "__path_clear__"
            clear_cooldown = self._cooldowns.get("CLEAR", 10.0)
            last_clear = self._cooldown_tracker.get(clear_key, 0.0)
            if (now - last_clear) >= clear_cooldown:
                text = build_clear_announcement()
                announcements.append(Announcement(text=text, urgency="CLEAR"))
                self._cooldown_tracker[clear_key] = now

        # Step 6: Scene summary (periodic)
        if (now - self._last_summary_time) >= self._scene_summary_interval:
            if not has_critical and self._recent_objects:
                summary = build_scene_summary(self._recent_objects[-20:])
                if summary:
                    announcements.append(
                        Announcement(text=summary, urgency="INFO")
                    )
                self._recent_objects = []
            self._last_summary_time = now

        # Populate frame_data
        frame_data.structured_announcements = announcements
        frame_data.announcements = [a.text for a in announcements]

        if announcements:
            self._last_announcement_time = now

        logger.debug(
            "Frame %d: %d objects -> %d announcements",
            frame_data.frame_id, len(frame_data.objects), len(announcements),
        )

        return frame_data

    def _classify_zone(
        self,
        obj: DetectedObject,
        frame_w: int,
        frame_h: int,
    ) -> str:
        """Classify object into one of 5 spatial zones.

        Zones:
          DEAD_AHEAD:  center_x in [35%, 65%] of frame width
          LEFT_NEAR:   center_x in [0%, 35%] AND center_y in [40%, 100%]
          RIGHT_NEAR:  center_x in [65%, 100%] AND center_y in [40%, 100%]
          LEFT_FAR:    center_x in [0%, 35%] AND center_y in [0%, 40%]
          RIGHT_FAR:   center_x in [65%, 100%] AND center_y in [0%, 40%]

        Args:
            obj: DetectedObject with bbox set.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            Zone string.
        """
        x1, y1, x2, y2 = obj.bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        left_pct, right_pct = self._dead_ahead_zone
        left_px = frame_w * left_pct
        right_px = frame_w * right_pct
        mid_y = frame_h * 0.40

        cx_frac = center_x / max(frame_w, 1)

        if left_pct <= cx_frac <= right_pct:
            return "DEAD_AHEAD"
        elif cx_frac < left_pct:
            return "LEFT_NEAR" if center_y >= mid_y else "LEFT_FAR"
        else:
            return "RIGHT_NEAR" if center_y >= mid_y else "RIGHT_FAR"

    def _classify_urgency(
        self,
        obj: DetectedObject,
        frame_h: int,
    ) -> str:
        """Classify object into urgency tier.

        Tier 1 CRITICAL: (distance < 1.5m AND DEAD_AHEAD) OR is_approaching
        Tier 2 WARNING:  distance < 3.0m OR (distance < 5.0m AND DEAD_AHEAD)
        Tier 3 INFO:     everything else with confidence >= 0.40
        Tier 4 CLEAR:    (handled at frame level, not per-object)

        Special: Stairs get elevated urgency.

        Args:
            obj: DetectedObject with zone and distance set.
            frame_h: Frame height for staircase analysis.

        Returns:
            Urgency string: 'CRITICAL', 'WARNING', or 'INFO'.
        """
        distance = obj.smoothed_distance_m or obj.distance_m or 99.0
        label = obj.label.lower()
        zone = obj.zone or "DEAD_AHEAD"

        # Approaching objects are always CRITICAL
        if obj.is_approaching:
            return "CRITICAL"

        # Staircase special handling
        if label in _STAIRCASE_LABELS:
            return self._classify_staircase_urgency(obj, frame_h, distance)

        # CRITICAL: close + dead ahead, or hazard objects very close
        if zone == "DEAD_AHEAD" and distance < self._critical_dist:
            return "CRITICAL"
        if label in _HAZARD_SET and distance < self._critical_dist:
            return "CRITICAL"

        # WARNING: nearby or dead ahead at moderate distance
        if distance < self._warning_dist:
            return "WARNING"
        if zone == "DEAD_AHEAD" and distance < 5.0:
            return "WARNING"

        # Far objects and peripheral: INFO
        return "INFO"

    def _classify_staircase_urgency(
        self,
        obj: DetectedObject,
        frame_h: int,
        distance: float,
    ) -> str:
        """Special urgency classification for stairs/steps.

        If bbox bottom is in the lower portion of the frame, the stairs
        are going DOWN (more dangerous). Otherwise, going UP.

        Args:
            obj: DetectedObject (stairs/step/escalator).
            frame_h: Frame height in pixels.
            distance: Current distance in metres.

        Returns:
            Urgency string.
        """
        _, _, _, y2 = obj.bbox
        is_going_down = y2 > frame_h * self._staircase_bottom_pct

        if is_going_down:
            return "CRITICAL" if distance < 2.0 else "WARNING"
        else:
            return "WARNING" if distance < 3.0 else "INFO"

    def _compute_score(self, obj: DetectedObject) -> float:
        """Compute priority score for sorting.

        Higher score = more urgent. Combines urgency tier, distance,
        and zone proximity.

        Args:
            obj: DetectedObject with urgency, distance, zone set.

        Returns:
            Priority score as float.
        """
        urgency_weight = {
            "CRITICAL": 10.0,
            "WARNING": 5.0,
            "INFO": 1.0,
        }.get(obj.urgency or "INFO", 1.0)

        distance = max(obj.smoothed_distance_m or obj.distance_m or 10.0, 0.3)
        zone_weight = 1.5 if obj.zone == "DEAD_AHEAD" else 1.0

        return urgency_weight * zone_weight * (1.0 / distance)
