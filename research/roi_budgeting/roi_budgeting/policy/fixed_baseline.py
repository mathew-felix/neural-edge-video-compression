from __future__ import annotations

from typing import Dict, Iterable, List, Mapping


def fixed_state_interval_keep(
    *,
    frame_ids: Iterable[int],
    has_detection: Mapping[int, bool],
    state_by_frame: Mapping[int, str],
    motion_interval: int = 2,
    still_interval: int = 3,
) -> List[int]:
    """
    Lightweight ROI-only reproduction of the current fixed cadence idea.

    Useful as an offline baseline inside the research workspace.
    """
    kept: List[int] = []
    last_keep = -1
    prev_has_detection = False

    for frame_idx in sorted(int(v) for v in frame_ids):
        present = bool(has_detection.get(frame_idx, False))
        if not present:
            prev_has_detection = False
            continue

        state = str(state_by_frame.get(frame_idx, "STILL")).strip().upper()
        interval = int(motion_interval) if state == "MOTION" else int(still_interval)
        roi_birth = not prev_has_detection
        gap = (frame_idx - last_keep) if last_keep >= 0 else (frame_idx + 1)
        keep = roi_birth or (gap >= interval)
        if keep:
            kept.append(frame_idx)
            last_keep = frame_idx
        prev_has_detection = True

    return kept


def has_detection_from_frame_drop(frame_drop_json: Dict[str, object]) -> Dict[int, bool]:
    per_frame = frame_drop_json.get("per_frame", {}) or {}
    out: Dict[int, bool] = {}
    if isinstance(per_frame, dict):
        for key, record in per_frame.items():
            try:
                frame_idx = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(record, dict):
                out[frame_idx] = False
                continue
            out[frame_idx] = bool((record.get("roi_count", 0) or 0) > 0) and not bool(record.get("bbox_missing", False))
    return out


def state_from_frame_drop(frame_drop_json: Dict[str, object]) -> Dict[int, str]:
    per_frame = frame_drop_json.get("per_frame", {}) or {}
    out: Dict[int, str] = {}
    if isinstance(per_frame, dict):
        for key, record in per_frame.items():
            try:
                frame_idx = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(record, dict):
                out[frame_idx] = "STILL"
                continue
            state = str(record.get("state", "STILL")).strip().upper()
            out[frame_idx] = state if state in {"STILL", "MOTION"} else "STILL"
    return out
