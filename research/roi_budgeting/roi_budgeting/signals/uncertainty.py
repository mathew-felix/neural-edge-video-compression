from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping


def _frame_boxes(roi_payload: Mapping[str, Any], frame_idx: int) -> List[Dict[str, Any]]:
    frames = roi_payload.get("frames", {}) if isinstance(roi_payload, dict) else {}
    boxes = frames.get(str(int(frame_idx)), None)
    if boxes is None:
        boxes = frames.get(int(frame_idx), None)
    return list(boxes) if isinstance(boxes, list) else []


def _quantile(values: List[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    q_clamped = max(0.0, min(1.0, float(q)))
    pos = q_clamped * float(len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    alpha = pos - float(lo)
    return (vals[lo] * (1.0 - alpha)) + (vals[hi] * alpha)


def build_uncertainty_features(roi_payload: Mapping[str, Any]) -> Dict[int, Dict[str, float]]:
    """
    Build simple detector-uncertainty proxies from per-frame ROI outputs.

    This is a starting point. Calibration, track disagreement, and occlusion
    terms should be layered on top in later experiments.
    """
    frames = roi_payload.get("frames", {}) if isinstance(roi_payload, dict) else {}
    frame_ids = sorted(int(k) for k in frames.keys() if str(k).isdigit())
    out: Dict[int, Dict[str, float]] = {}
    prev_count = 0
    prev_tracks: set[int] = set()

    for frame_idx in frame_ids:
        boxes = _frame_boxes(roi_payload, frame_idx)
        confs: List[float] = []
        track_ids: set[int] = set()
        for box in boxes:
            raw = box.get("conf", box.get("confidence", None))
            try:
                confs.append(float(raw))
            except (TypeError, ValueError):
                continue
            try:
                track_ids.add(int(box.get("track_id")))
            except (TypeError, ValueError):
                pass

        max_conf = max(confs) if confs else 0.0
        mean_conf = (sum(confs) / len(confs)) if confs else 0.0
        count = len(boxes)
        count_change_abs = abs(count - prev_count)
        track_births = len(track_ids - prev_tracks)
        track_deaths = len(prev_tracks - track_ids)
        track_turnover = track_births + track_deaths
        out[frame_idx] = {
            "roi_count": float(count),
            "mean_conf": float(mean_conf),
            "max_conf": float(max_conf),
            "uncertainty": float(1.0 - max(0.0, min(1.0, max_conf))) if confs else 1.0,
            "missing_detection": 0.0 if boxes else 1.0,
            "count_change_abs": float(count_change_abs),
            "track_births": float(track_births),
            "track_deaths": float(track_deaths),
            "track_turnover": float(track_turnover),
        }
        prev_count = count
        prev_tracks = set(track_ids)
    return out


def build_uncertainty_scores(
    uncertainty_features: Mapping[int, Mapping[str, Any]],
) -> Dict[int, Dict[str, float]]:
    def _missing_flag(rec: Mapping[str, Any]) -> float:
        raw = rec.get("missing_detection", 1.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0

    conf_unc_vals = [
        float((rec.get("uncertainty", 0.0) or 0.0))
        for rec in uncertainty_features.values()
        if _missing_flag(rec) < 1.0
    ]
    count_change_vals = [
        float((rec.get("count_change_abs", 0.0) or 0.0))
        for rec in uncertainty_features.values()
        if _missing_flag(rec) < 1.0
    ]
    turnover_vals = [
        float((rec.get("track_turnover", 0.0) or 0.0))
        for rec in uncertainty_features.values()
        if _missing_flag(rec) < 1.0
    ]

    count_change_scale = _quantile(count_change_vals, 0.90) or max(count_change_vals or [1.0])
    turnover_scale = _quantile(turnover_vals, 0.90) or max(turnover_vals or [1.0])
    count_change_scale = max(1e-6, float(count_change_scale))
    turnover_scale = max(1e-6, float(turnover_scale))

    out: Dict[int, Dict[str, float]] = {}
    for frame_idx, rec in uncertainty_features.items():
        missing = _missing_flag(rec)
        raw_unc = float((rec.get("uncertainty", 0.0) or 0.0))
        count_change = float((rec.get("count_change_abs", 0.0) or 0.0))
        turnover = float((rec.get("track_turnover", 0.0) or 0.0))

        count_change_norm = min(1.0, count_change / count_change_scale) if missing < 1.0 else 0.0
        turnover_norm = min(1.0, turnover / turnover_scale) if missing < 1.0 else 0.0
        score = (0.6 * raw_unc) + (0.2 * count_change_norm) + (0.2 * turnover_norm)
        out[int(frame_idx)] = {
            "missing_detection": float(missing),
            "uncertainty_raw": float(raw_unc),
            "count_change_abs": float(count_change),
            "track_turnover": float(turnover),
            "count_change_norm": float(count_change_norm),
            "track_turnover_norm": float(turnover_norm),
            "uncertainty_score": float(score),
        }
    return out
