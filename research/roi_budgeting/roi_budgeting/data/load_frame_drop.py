from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_frame_drop(path: str | Path) -> Dict[str, Any]:
    """Load a `frame_drop.json` artifact."""
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def roi_keep_indices(frame_drop_json: Dict[str, Any]) -> List[int]:
    raw = frame_drop_json.get("roi_kept_frames", frame_drop_json.get("kept_frames", [])) or []
    return sorted(int(x) for x in raw)


def roi_drop_indices(frame_drop_json: Dict[str, Any]) -> List[int]:
    raw = frame_drop_json.get("roi_dropped_frames", frame_drop_json.get("dropped_frames", [])) or []
    return sorted(int(x) for x in raw)


def roi_segments(frame_drop_json: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Return contiguous frame ranges where ROI exists according to `per_frame`.
    """
    per_frame = frame_drop_json.get("per_frame", {}) or {}
    frame_ids = sorted(int(k) for k in per_frame.keys() if str(k).isdigit())
    if not frame_ids:
        return []

    segments: List[Tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None

    for frame_idx in frame_ids:
        record = per_frame.get(str(frame_idx), {}) or {}
        has_roi = bool((record.get("roi_count", 0) or 0) > 0) and not bool(record.get("bbox_missing", False))
        if has_roi:
            if start is None:
                start = frame_idx
            prev = frame_idx
        elif start is not None and prev is not None:
            segments.append((start, prev))
            start = None
            prev = None

    if start is not None and prev is not None:
        segments.append((start, prev))
    return segments
