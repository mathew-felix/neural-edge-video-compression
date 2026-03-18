from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List


def archive_entries(path: str | Path) -> List[str]:
    p = Path(path).expanduser().resolve()
    with zipfile.ZipFile(p, "r") as zf:
        return sorted(zf.namelist())


def load_json_entry(path: str | Path, entry_name: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with zipfile.ZipFile(p, "r") as zf:
        return json.loads(zf.read(entry_name).decode("utf-8"))


def inspect_archive(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with zipfile.ZipFile(p, "r") as zf:
        names = sorted(zf.namelist())
        return {
            "path": str(p),
            "entries": names,
            "has_frame_drop": any(name.endswith("frame_drop.json") for name in names),
            "has_roi_detections": any(name.endswith("roi_detections.json") for name in names),
            "has_meta": any(name.endswith("meta.json") for name in names),
        }
