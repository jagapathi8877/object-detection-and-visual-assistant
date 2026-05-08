"""
Depth Estimation Module -- MiDaS DPT-Small Monocular Depth.

Uses Intel's MiDaS DPT-Small model loaded via torch.hub for monocular
depth estimation. Converts relative inverse-depth to approximate metric
distances using a configurable scale_factor.

Per master prompt: Depth: MiDaS DPT-Small (torch.hub)

Optimisations for accuracy:
  - Multi-zone sampling (centre + lower-centre + inset) per bbox
  - EMA smoothing per tracked object
  - Approach velocity estimation
  - Frame skipping (run depth every Nth frame)
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from utils.datatypes import DetectedObject, FrameData
from utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """MiDaS DPT-Small monocular depth estimator."""

    def __init__(self, config: dict) -> None:
        required_keys = ["model", "min_distance", "max_distance"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing depth config key: '{key}'.")

        self._scale_factor = float(config.get("scale_factor", 1000.0))
        self._min_distance = float(config["min_distance"])
        self._max_distance = float(config["max_distance"])
        self._approach_threshold = float(config.get("approaching_threshold_m_per_s", 0.4))
        self._ema_alpha_critical = float(config.get("ema_alpha_critical", 0.4))
        self._ema_alpha_info = float(config.get("ema_alpha_info", 0.2))
        self._small_bbox_pct = float(config.get("small_bbox_threshold_pct", 1.0))
        self._prev_objects: List[DetectedObject] = []
        self._prev_timestamp: float = 0.0
        self._depth_skip_frames = int(config.get("skip_frames", 3))
        self._frame_counter = 0
        self._cached_depth_map: Optional[np.ndarray] = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._transform = self._load_model(config["model"])

    def _load_model(self, model_type: str):
        """Load MiDaS via torch.hub."""
        logger.info("Loading MiDaS '%s' on %s...", model_type, self._device)
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        model.to(self._device)
        model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "MiDaS_small":
            transform = transforms.small_transform
        elif model_type in ("DPT_Large", "DPT_Hybrid"):
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform
        logger.info("MiDaS '%s' loaded on %s", model_type, self._device)
        return model, transform

    def estimate(self, frame: np.ndarray, frame_data: FrameData) -> FrameData:
        """Estimate depth for all detected objects."""
        if frame is None or frame.size == 0:
            raise ValueError("Cannot estimate depth on empty frame.")
        if not frame_data.objects:
            return frame_data

        if frame.dtype == np.float32:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame

        if len(frame_uint8.shape) == 3 and frame_uint8.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame_uint8

        frame_h, frame_w = frame_uint8.shape[:2]
        self._frame_counter += 1

        if self._cached_depth_map is None or (self._frame_counter % self._depth_skip_frames == 0):
            try:
                depth_map = self._run_midas(frame_rgb, frame_h, frame_w)
                self._cached_depth_map = depth_map
            except Exception as exc:
                logger.error("Depth estimation failed: %s", exc)
                for obj in frame_data.objects:
                    obj.distance_m = self._max_distance
                    obj.distance_confidence = "low"
                return frame_data
        else:
            depth_map = self._cached_depth_map

        tracked_prev = self._track_objects(self._prev_objects, frame_data.objects)
        frame_area = frame_h * frame_w

        for i, obj in enumerate(frame_data.objects):
            inv_depth = self._sample_depth(depth_map, obj.bbox)
            raw_depth = self._scale_factor / inv_depth if inv_depth > 0 else self._max_distance
            raw_depth = float(np.clip(raw_depth, self._min_distance, self._max_distance))
            obj.distance_m = raw_depth

            bx1, by1, bx2, by2 = obj.bbox
            bbox_area = max((bx2 - bx1) * (by2 - by1), 1)
            obj.distance_confidence = "high" if (bbox_area / frame_area) * 100 >= self._small_bbox_pct else "low"

            prev_obj = tracked_prev.get(i)
            if prev_obj and prev_obj.distance_m and self._prev_timestamp > 0:
                dt = max(frame_data.timestamp - self._prev_timestamp, 0.01)
                velocity = (prev_obj.distance_m - raw_depth) / dt
                obj.approach_velocity = max(velocity, 0.0)
                obj.is_approaching = velocity > self._approach_threshold
                obj.prev_distance_m = prev_obj.distance_m
            else:
                obj.approach_velocity = 0.0
                obj.is_approaching = False

            alpha = self._ema_alpha_critical if raw_depth < 2.0 else self._ema_alpha_info
            if prev_obj and prev_obj.smoothed_distance_m:
                obj.smoothed_distance_m = alpha * raw_depth + (1 - alpha) * prev_obj.smoothed_distance_m
            else:
                obj.smoothed_distance_m = raw_depth

        self._prev_objects = list(frame_data.objects)
        self._prev_timestamp = frame_data.timestamp
        logger.debug("Frame %d: Depth estimated for %d objects", frame_data.frame_id, len(frame_data.objects))
        return frame_data

    def _run_midas(self, frame_rgb: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
        """Run MiDaS inference."""
        input_batch = self._transform(frame_rgb).to(self._device)
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(frame_h, frame_w),
                mode="bicubic", align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy().astype(np.float32)

    def _sample_depth(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Multi-zone depth sampling. Returns inverse-depth (higher=closer)."""
        x1, y1, x2, y2 = bbox
        map_h, map_w = depth_map.shape[:2]
        x1, y1 = max(0, min(x1, map_w-1)), max(0, min(y1, map_h-1))
        x2, y2 = max(x1+1, min(x2, map_w)), max(y1+1, min(y2, map_h))
        h, w = y2 - y1, x2 - x1
        if h < 2 or w < 2:
            return 0.0
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        samples = []
        # Zone 1: Centre
        cz = depth_map[max(y1, cy-h//5):min(y2, cy+h//5), max(x1, cx-w//5):min(x2, cx+w//5)]
        if cz.size > 0:
            samples.append(float(np.median(cz)))
        # Zone 2: Lower-centre
        lz = depth_map[max(y1, y2-h//4):y2, max(x1, cx-w//4):min(x2, cx+w//4)]
        if lz.size > 0:
            samples.append(float(np.median(lz)))
        # Zone 3: Inset
        iy1, iy2 = y1 + h//10, y2 - h//10
        ix1, ix2 = x1 + w//10, x2 - w//10
        if iy2 > iy1 and ix2 > ix1:
            iz = depth_map[iy1:iy2, ix1:ix2]
            if iz.size > 0:
                samples.append(float(np.percentile(iz, 85)))
        return max(samples) if samples else 0.0

    def _track_objects(self, prev: List[DetectedObject], curr: List[DetectedObject]) -> Dict[int, Optional[DetectedObject]]:
        """Match current objects to previous frame via IoU."""
        tracked: Dict[int, Optional[DetectedObject]] = {}
        if not prev:
            return tracked
        for i, c in enumerate(curr):
            best, best_iou = None, 0.4
            for p in prev:
                if c.label != p.label:
                    continue
                iou = self._iou(c.bbox, p.bbox)
                if iou > best_iou:
                    best_iou, best = iou, p
            tracked[i] = best
        return tracked

    @staticmethod
    def _iou(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        xa, ya = max(a[0], b[0]), max(a[1], b[1])
        xb, yb = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xb-xa) * max(0, yb-ya)
        if inter == 0:
            return 0.0
        aa = (a[2]-a[0]) * (a[3]-a[1])
        ab = (b[2]-b[0]) * (b[3]-b[1])
        return inter / max(aa + ab - inter, 1e-6)
