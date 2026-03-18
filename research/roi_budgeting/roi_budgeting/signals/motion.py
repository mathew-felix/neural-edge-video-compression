from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple


def _frame_boxes(roi_payload: Mapping[str, Any], frame_idx: int) -> List[Dict[str, Any]]:
    frames = roi_payload.get("frames", {}) if isinstance(roi_payload, dict) else {}
    boxes = frames.get(str(int(frame_idx)), None)
    if boxes is None:
        boxes = frames.get(int(frame_idx), None)
    return list(boxes) if isinstance(boxes, list) else []


def _union_box(boxes: List[Dict[str, Any]]) -> Tuple[float, float, float, float] | None:
    parsed: List[Tuple[float, float, float, float]] = []
    for box in boxes:
        try:
            parsed.append(
                (
                    float(box["x1"]),
                    float(box["y1"]),
                    float(box["x2"]),
                    float(box["y2"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not parsed:
        return None
    x1 = min(v[0] for v in parsed)
    y1 = min(v[1] for v in parsed)
    x2 = max(v[2] for v in parsed)
    y2 = max(v[3] for v in parsed)
    return x1, y1, x2, y2


def build_motion_features(roi_payload: Mapping[str, Any]) -> Dict[int, Dict[str, float]]:
    """
    Build lightweight motion proxies from ROI boxes.

    This is intentionally simple for the research scaffold. More expensive motion
    terms can be added later from optical flow or crop-level alignment.
    """
    frames = roi_payload.get("frames", {}) if isinstance(roi_payload, dict) else {}
    frame_ids = sorted(int(k) for k in frames.keys() if str(k).isdigit())

    out: Dict[int, Dict[str, float]] = {}
    prev_center: Tuple[float, float] | None = None
    prev_area: float | None = None
    for frame_idx in frame_ids:
        union_box = _union_box(_frame_boxes(roi_payload, frame_idx))
        if union_box is None:
            out[frame_idx] = {
                "has_roi": 0.0,
                "center_speed": 0.0,
                "area_delta_abs": 0.0,
            }
            prev_center = None
            prev_area = None
            continue

        x1, y1, x2, y2 = union_box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if prev_center is None:
            center_speed = 0.0
        else:
            center_speed = math.hypot(cx - prev_center[0], cy - prev_center[1])

        if prev_area is None:
            area_delta_abs = 0.0
        else:
            area_delta_abs = abs(area - prev_area)

        out[frame_idx] = {
            "has_roi": 1.0,
            "center_speed": float(center_speed),
            "area_delta_abs": float(area_delta_abs),
        }
        prev_center = (cx, cy)
        prev_area = area
    return out
