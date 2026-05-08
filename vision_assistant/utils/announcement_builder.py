"""
Announcement Builder -- Smart Text Generation for Blind Navigation.

Generates context-aware announcement text based on object properties,
urgency tier, distance, and zone. Makes the voice output feel like a
real navigation assistant, not a label reader.

Design:
  CRITICAL: "Warning! Person directly ahead, 1 metre"
  WARNING:  "Chair on the left, 2.5 metres"
  INFO:     "Table on the right, about 4 metres away"
  APPROACHING: "Caution! Car approaching from ahead"
  CLEAR:    "Path is clear"
"""

from typing import Optional

from utils.datatypes import DetectedObject
from utils.label_map import get_speech_label


# ── Zone to Speech Phrase Mapping ───────────────────────────────

ZONE_PHRASES = {
    "DEAD_AHEAD": "directly ahead",
    "LEFT_NEAR": "close on your left",
    "RIGHT_NEAR": "close on your right",
    "LEFT_FAR": "far left",
    "RIGHT_FAR": "far right",
}

# Fallback for legacy 3-zone direction system
DIRECTION_PHRASES = {
    "ahead": "ahead",
    "left": "on the left",
    "right": "on the right",
}


def _round_distance(distance_m: float) -> str:
    """Round distance for natural speech.

    Rules:
      < 1m:  1 decimal place  "0.8 metres"
      1-5m:  nearest 0.5     "2.5 metres", "3 metres"
      > 5m:  nearest metre   "6 metres"

    Args:
        distance_m: Raw distance in metres.

    Returns:
        Human-readable distance string.
    """
    if distance_m < 1.0:
        rounded = round(distance_m, 1)
        word = "metre" if rounded == 1.0 else "metres"
        return f"{rounded} {word}"
    elif distance_m <= 5.0:
        rounded = round(distance_m * 2) / 2  # nearest 0.5
        if rounded == int(rounded):
            rounded = int(rounded)
        word = "metre" if rounded == 1 else "metres"
        return f"{rounded} {word}"
    else:
        rounded = round(distance_m)
        word = "metre" if rounded == 1 else "metres"
        return f"{rounded} {word}"


def _get_zone_phrase(obj: DetectedObject) -> str:
    """Get the spoken direction/zone phrase for an object.

    Prefers the 5-zone system; falls back to 3-zone direction.

    Args:
        obj: DetectedObject with zone and/or direction set.

    Returns:
        Spoken direction phrase string.
    """
    if obj.zone and obj.zone in ZONE_PHRASES:
        return ZONE_PHRASES[obj.zone]
    if obj.direction and obj.direction in DIRECTION_PHRASES:
        return DIRECTION_PHRASES[obj.direction]
    return "ahead"


def build_announcement(
    obj: DetectedObject,
    urgency: str,
    distance_confidence: str = "high",
) -> str:
    """Build a context-aware spoken announcement for a detected object.

    Args:
        obj: DetectedObject with label, zone/direction, distance set.
        urgency: Urgency tier: 'CRITICAL', 'WARNING', 'INFO', 'CLEAR'.
        distance_confidence: 'high' or 'low'.

    Returns:
        Formatted announcement string ready for TTS.
    """
    label = get_speech_label(obj.label)
    zone_phrase = _get_zone_phrase(obj)
    distance = obj.smoothed_distance_m or obj.distance_m or 0.0
    dist_str = _round_distance(distance)

    # Fast-approaching object override
    if obj.is_approaching and obj.approach_velocity > 1.0:
        return f"Fast-moving {label} approaching -- stop and wait"

    if obj.is_approaching:
        return f"{label.capitalize()} approaching from {zone_phrase}"

    if urgency == "CRITICAL":
        return f"{label.capitalize()} {zone_phrase}, {dist_str}"

    if urgency == "WARNING":
        return f"{label.capitalize()} {zone_phrase}, {dist_str}"

    # INFO: use approximate language if distance confidence is low
    if distance_confidence == "low" or distance > 5.0:
        return f"{label.capitalize()} {zone_phrase}, about {dist_str} away"

    return f"{label.capitalize()} {zone_phrase}, {dist_str}"


def build_clear_announcement() -> str:
    """Build the 'path is clear' reassurance announcement.

    Returns:
        "Path is clear" string.
    """
    return "Path is clear"


def build_scene_summary(objects: list) -> Optional[str]:
    """Build a periodic scene summary from recent detections.

    Groups objects by zone and generates a natural summary sentence.
    Example: "Around you: clear path ahead. Chair to your left."

    Args:
        objects: List of DetectedObject instances from recent frames.

    Returns:
        Scene summary string, or None if no meaningful summary.
    """
    if not objects:
        return None

    # Group by zone
    zone_objects = {}
    for obj in objects:
        zone = obj.zone or obj.direction or "ahead"
        label = get_speech_label(obj.label)
        if zone not in zone_objects:
            zone_objects[zone] = set()
        zone_objects[zone].add(label)

    # Build summary phrases
    phrases = []
    zone_order = ["DEAD_AHEAD", "ahead", "LEFT_NEAR", "left",
                  "RIGHT_NEAR", "right", "LEFT_FAR", "RIGHT_FAR"]

    for zone in zone_order:
        if zone in zone_objects:
            items = list(zone_objects[zone])[:3]  # max 3 per zone
            zone_name = ZONE_PHRASES.get(zone, DIRECTION_PHRASES.get(zone, zone))
            if len(items) == 1:
                phrases.append(f"{items[0]} {zone_name}")
            else:
                items_str = " and ".join(items[:2])
                phrases.append(f"{items_str} {zone_name}")

    if not phrases:
        return None

    return "Around you: " + ". ".join(phrases[:4])  # max 4 zone groups
