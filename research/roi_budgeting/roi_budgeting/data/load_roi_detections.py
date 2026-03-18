from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


def load_roi_detections(path: str | Path) -> Dict[str, Any]:
    """Load a `roi_detections.json` artifact."""
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def frame_boxes(payload: Mapping[str, Any], frame_idx: int) -> List[Dict[str, Any]]:
    frames = payload.get("frames", {}) if isinstance(payload, dict) else {}
    boxes = frames.get(str(int(frame_idx)), None)
    if boxes is None:
        boxes = frames.get(int(frame_idx), None)
    return list(boxes) if isinstance(boxes, list) else []
