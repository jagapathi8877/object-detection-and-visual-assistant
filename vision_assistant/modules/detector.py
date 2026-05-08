"""
Object Detection Module -- YOLOv8n COCO 80-Class Detection.

Uses Ultralytics YOLOv8 nano variant for real-time object detection with
80 COCO classes. The nano variant is optimised for speed on CPU while
maintaining good accuracy for assistive navigation.

Architecture:
  STAGE 1: YOLOv8n detection with calibrated NMS (conf, iou)
  STAGE 2: Label normalisation via utils/label_map.py
  STAGE 3: Confidence-based filtering
  STAGE 4: In-frame deduplication (same label + IoU > 0.5 = keep best)

Per master prompt: Detection: YOLOv8 (ultralytics) — COCO 80-class, nano variant.
"""

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from utils.datatypes import DetectedObject, FrameData
from utils.label_map import get_speech_label, EXTENDED_LABEL_MAP
from utils.logger import get_logger

logger = get_logger(__name__)


def _compute_iou(box_a: Tuple[int, ...], box_b: Tuple[int, ...]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box_a: (x1, y1, x2, y2) pixel coordinates.
        box_b: (x1, y1, x2, y2) pixel coordinates.

    Returns:
        IoU value in [0.0, 1.0].
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / max(union, 1e-6)


class ObjectDetector:
    """YOLOv8n object detector with COCO 80-class detection.

    Detects all 80 COCO classes using the YOLOv8 nano variant for
    maximum speed on CPU. Applies optional whitelist filtering and
    label normalisation for speech-friendly output.

    Attributes:
        _model: Loaded YOLOv8n model.
        _confidence: Minimum detection confidence threshold.
        _iou_threshold: NMS IoU threshold.
        _max_detections: Maximum detections per frame.
        _class_whitelist: Optional set of allowed class names.
        _class_names: Mapping of class ID -> class name.
        _max_objects: Maximum objects to announce per cycle.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the object detector.

        Args:
            config: Detection section from config.yaml.

        Raises:
            ValueError: If required config keys are missing.
            RuntimeError: If no model can be loaded.
        """
        # Support both prompt-spec keys and legacy keys
        self._confidence: float = config.get(
            "confidence", config.get("conf_threshold", 0.5)
        )
        self._iou_threshold: float = config.get("iou_threshold", 0.45)
        self._max_detections: int = config.get("max_detections", 8)
        self._max_objects: int = config.get("max_objects", 5)
        self._imgsz: int = config.get("imgsz", 320)  # Lower = faster on CPU

        # Whitelist filtering: if 'classes' is provided in config, filter to those
        # Otherwise detect all 80 COCO classes (per prompt spec)
        whitelist_raw = config.get("classes", None)
        if whitelist_raw and isinstance(whitelist_raw, list):
            self._class_whitelist: Optional[Set[str]] = {
                c.lower().strip() for c in whitelist_raw
            }
        else:
            # No whitelist = detect ALL 80 COCO classes
            self._class_whitelist = None

        self._label_normalisation: bool = config.get("label_normalisation", True)

        # Load YOLOv8n as specified in the master prompt
        model_path = config.get("model", "yolov8n.pt")
        self._model = self._load_model(model_path)
        self._class_names: Dict[int, str] = self._model.names

        logger.info(
            "Detector ready: YOLOv8n (%s), %d COCO classes, conf=%.2f, iou=%.2f",
            model_path, len(self._class_names),
            self._confidence, self._iou_threshold,
        )

    def _load_model(self, model_path: str):
        """Load standard YOLOv8n model from ultralytics.

        Args:
            model_path: YOLOv8 model weight file (default: yolov8n.pt).

        Returns:
            Loaded YOLO model.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            logger.info("YOLOv8 loaded: '%s'", model_path)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLOv8 model '{model_path}': {exc}"
            ) from exc

    def detect(
        self,
        frame: np.ndarray,
        frame_data: FrameData,
    ) -> FrameData:
        """Run object detection on a single frame.

        Pipeline: YOLO inference -> label normalisation -> whitelist
        filtering -> confidence filtering -> in-frame deduplication.

        Args:
            frame: Input frame as np.ndarray.
            frame_data: FrameData to populate with detections.

        Returns:
            Updated FrameData with DetectedObject entries.

        Raises:
            ValueError: If frame is None or empty.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Cannot run detection on an empty or None frame.")

        # Convert float32 [0,1] to uint8 [0,255] if needed
        if frame.dtype == np.float32 and frame.max() <= 1.0:
            inference_frame = (frame * 255).astype(np.uint8)
        else:
            inference_frame = frame

        # STAGE 1: YOLOv8n inference (imgsz controls speed/accuracy tradeoff)
        results = self._model(
            inference_frame,
            conf=self._confidence,
            iou=self._iou_threshold,
            max_det=self._max_detections,
            imgsz=self._imgsz,
            verbose=False,
        )

        if not results or len(results) == 0:
            return frame_data

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return frame_data

        raw_objects: List[DetectedObject] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            raw_label = self._class_names.get(class_id, "unknown").lower().strip()
            confidence = float(box.conf.item())

            # STAGE 2: Whitelist filtering (skip if no whitelist = all 80 classes)
            if self._class_whitelist and raw_label not in self._class_whitelist:
                continue

            # STAGE 3: Label normalisation for speech-friendly output
            if self._label_normalisation:
                label = get_speech_label(raw_label)
            else:
                label = raw_label

            # Confidence filtering
            if confidence < self._confidence:
                continue

            # Extract bounding box
            xyxy = box.xyxy[0].tolist()
            bbox = (
                int(round(xyxy[0])),
                int(round(xyxy[1])),
                int(round(xyxy[2])),
                int(round(xyxy[3])),
            )

            raw_objects.append(DetectedObject(
                label=label,
                bbox=bbox,
                confidence=confidence,
            ))

        # STAGE 4: In-frame deduplication
        deduplicated = self._deduplicate(raw_objects)
        frame_data.objects = deduplicated

        logger.debug(
            "Frame %d: %d raw -> %d deduplicated objects",
            frame_data.frame_id, len(raw_objects), len(deduplicated),
        )

        return frame_data

    def _deduplicate(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """Remove duplicate detections of the same object.

        If two DetectedObjects have the same label and IoU > 0.5,
        keep only the one with higher confidence.

        Args:
            objects: List of raw DetectedObject instances.

        Returns:
            Deduplicated list.
        """
        if len(objects) <= 1:
            return objects

        # Sort by confidence descending so we keep highest first
        sorted_objs = sorted(objects, key=lambda o: o.confidence, reverse=True)
        keep: List[DetectedObject] = []

        for obj in sorted_objs:
            is_duplicate = False
            for kept in keep:
                if obj.label == kept.label:
                    iou = _compute_iou(obj.bbox, kept.bbox)
                    if iou > 0.5:
                        is_duplicate = True
                        break
            if not is_duplicate:
                keep.append(obj)

        return keep

    @property
    def whitelist(self) -> Set[str]:
        """Return the current class names."""
        return set(self._class_names.values())

    @property
    def confidence_threshold(self) -> float:
        """Return the current confidence threshold."""
        return self._confidence
