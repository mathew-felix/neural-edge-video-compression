from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import cv2
import numpy as np
import torch


def load_amt_probe_manifest(path: str | Path) -> Dict[str, Any]:
    """Load a saved AMT probe manifest produced by later experiments."""
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def build_amt_risk_features(manifest: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """
    Convert a probe manifest into per-frame AMT risk values.

    Expected shape is intentionally loose for now so the manifest format can
    evolve during research.
    """
    per_frame = manifest.get("per_frame", {}) or {}
    out: Dict[int, Dict[str, float]] = {}
    for key, value in per_frame.items():
        try:
            frame_idx = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        try:
            risk = float(value.get("amt_risk", value.get("probe_error", 0.0)) or 0.0)
        except (TypeError, ValueError):
            risk = 0.0
        out[frame_idx] = {"amt_risk": risk}
    return out


def _frame_boxes(roi_payload: Mapping[str, Any], frame_idx: int) -> List[Dict[str, Any]]:
    frames = roi_payload.get("frames", {}) if isinstance(roi_payload, dict) else {}
    boxes = frames.get(str(int(frame_idx)), None)
    if boxes is None:
        boxes = frames.get(int(frame_idx), None)
    return list(boxes) if isinstance(boxes, list) else []


def _union_box(boxes: List[Dict[str, Any]]) -> Tuple[int, int, int, int] | None:
    parsed: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        try:
            parsed.append(
                (
                    int(box["x1"]),
                    int(box["y1"]),
                    int(box["x2"]),
                    int(box["y2"]),
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


def _clip_box(
    box: Tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    margin_px: int,
) -> Tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(width - 1), int(x1) - int(margin_px)))
    y1 = max(0, min(int(height - 1), int(y1) - int(margin_px)))
    x2 = max(0, min(int(width), int(x2) + int(margin_px)))
    y2 = max(0, min(int(height), int(y2) + int(margin_px)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _resize_if_needed(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = float(max_side) / float(max(h, w))
    if scale >= 1.0:
        return frame
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_amt_risk_proxy_features(
    *,
    video_path: str | Path,
    roi_payload: Mapping[str, Any],
    crop_margin_px: int = 8,
    max_crop_side: int = 256,
) -> tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    """
    Build a lightweight AMT-risk proxy from local ROI crop reconstruction error.

    This is a research-only fallback for environments without CUDA AMT execution.
    For each interior frame t, it predicts the ROI crop using a simple midpoint
    blend between t-1 and t+1 and measures the normalized MAE against the true
    frame t crop. Higher error implies harder interpolation.
    """
    path = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for AMT-risk proxy: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    features: Dict[int, Dict[str, float]] = {}
    probe_errors: List[float] = []
    probe_count = 0

    try:
        ok_prev, prev_frame = cap.read()
        ok_cur, cur_frame = cap.read()
        if not ok_prev or not ok_cur:
            return features, {
                "source": "linear_crop_proxy",
                "video_path": str(path),
                "probe_count": 0,
                "notes": "Video too short for interpolation probes.",
            }

        cur_idx = 1
        while True:
            ok_next, next_frame = cap.read()
            if not ok_next:
                break

            boxes = (
                _frame_boxes(roi_payload, cur_idx - 1)
                + _frame_boxes(roi_payload, cur_idx)
                + _frame_boxes(roi_payload, cur_idx + 1)
            )
            union = _union_box(boxes)
            if union is not None:
                clipped = _clip_box(
                    union,
                    width=width,
                    height=height,
                    margin_px=int(crop_margin_px),
                )
            else:
                clipped = None

            if clipped is None:
                features[int(cur_idx)] = {
                    "amt_risk": 0.0,
                    "probe_error": 0.0,
                }
            else:
                x1, y1, x2, y2 = clipped
                prev_crop = _resize_if_needed(prev_frame[y1:y2, x1:x2], int(max_crop_side))
                cur_crop = _resize_if_needed(cur_frame[y1:y2, x1:x2], int(max_crop_side))
                next_crop = _resize_if_needed(next_frame[y1:y2, x1:x2], int(max_crop_side))
                pred = cv2.addWeighted(prev_crop, 0.5, next_crop, 0.5, 0.0)
                err = float(np.mean(np.abs(pred.astype(np.float32) - cur_crop.astype(np.float32))) / 255.0)
                features[int(cur_idx)] = {
                    "amt_risk": float(err),
                    "probe_error": float(err),
                }
                probe_errors.append(float(err))
                probe_count += 1

            prev_frame = cur_frame
            cur_frame = next_frame
            cur_idx += 1
    finally:
        cap.release()

    summary = {
        "source": "linear_crop_proxy",
        "video_path": str(path),
        "probe_count": int(probe_count),
        "crop_margin_px": int(crop_margin_px),
        "max_crop_side": int(max_crop_side),
        "probe_error_mean": float(sum(probe_errors) / len(probe_errors)) if probe_errors else 0.0,
        "probe_error_max": float(max(probe_errors)) if probe_errors else 0.0,
        "notes": "Local Mac fallback proxy for AMT reconstruction difficulty. This is not full GPU AMT inference.",
    }
    return features, summary


def _ensure_repo_src_on_path(repo_root: Path) -> None:
    src_dir = (repo_root / "src").resolve()
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _resolve_amt_inputs(
    *,
    repo_root: Path,
    amt_cfg: Mapping[str, Any],
) -> Tuple[Path, Path, str]:
    repo_dir_raw = str(amt_cfg.get("repo_dir", repo_root / "_third_party_amt"))
    weights_raw = str(amt_cfg.get("weights_path", repo_root / "models" / "amt-s.pth"))
    variant = str(amt_cfg.get("model", "amt-s")).strip().lower()
    repo_dir = Path(repo_dir_raw).expanduser()
    weights_path = Path(weights_raw).expanduser()
    if not repo_dir.is_absolute():
        repo_dir = (repo_root / repo_dir).resolve()
    else:
        repo_dir = repo_dir.resolve()
    if not weights_path.is_absolute():
        weights_path = (repo_root / weights_path).resolve()
    else:
        weights_path = weights_path.resolve()
    return repo_dir, weights_path, variant


def _load_amt_interpolator(
    *,
    repo_root: Path,
    amt_cfg: Mapping[str, Any],
):
    _ensure_repo_src_on_path(repo_root)
    from decompression.interpolation_amt import AmtInterpolator

    repo_dir, weights_path, variant = _resolve_amt_inputs(repo_root=repo_root, amt_cfg=amt_cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for AMT probing.")
    return AmtInterpolator(
        amt_repo_dir=str(repo_dir),
        variant=str(variant),
        weights_path=str(weights_path),
        device="cuda",
        fp16=True,
        pad_to=16,
    )


def _video_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    if frame_idx < 0:
        raise ValueError("frame_idx must be >= 0")
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    if not ok:
        raise RuntimeError(f"Could not seek to frame {frame_idx}")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame


def _union_box_range(
    roi_payload: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
) -> Tuple[int, int, int, int] | None:
    boxes: List[Dict[str, Any]] = []
    for frame_idx in range(int(start_frame), int(end_frame) + 1):
        boxes.extend(_frame_boxes(roi_payload, frame_idx))
    return _union_box(boxes)


def generate_amt_probe_manifest(
    *,
    video_path: str | Path,
    roi_payload: Mapping[str, Any],
    frame_drop_json: Mapping[str, Any],
    repo_root: str | Path,
    amt_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Generate a per-frame AMT probe manifest.

    On CUDA systems, this uses the repo's AMT interpolator to estimate true
    interpolation difficulty between consecutive ROI anchors.
    """
    repo_root_p = Path(repo_root).expanduser().resolve()
    video_path_p = Path(video_path).expanduser().resolve()
    probe_positions = [
        float(v)
        for v in (amt_cfg.get("probe_positions", [0.5]) or [0.5])
        if 0.0 < float(v) < 1.0
    ]
    probe_positions = sorted(set(probe_positions)) or [0.5]
    max_probe_gap_frames = int(amt_cfg.get("max_probe_gap_frames", 12) or 12)
    crop_margin_px = int(amt_cfg.get("crop_margin_px", 8) or 8)
    max_crop_side = int(amt_cfg.get("max_crop_side", 256) or 256)

    roi_kept = sorted(int(v) for v in (frame_drop_json.get("roi_kept_frames", []) or []))
    if len(roi_kept) < 2:
        return {
            "meta": {
                "source": "gpu_amt_probe",
                "video_path": str(video_path_p),
                "probe_count": 0,
                "notes": "Not enough ROI anchors to probe.",
            },
            "per_frame": {},
        }

    interpolator = _load_amt_interpolator(repo_root=repo_root_p, amt_cfg=amt_cfg)
    cap = cv2.VideoCapture(str(video_path_p))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for AMT probes: {video_path_p}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    per_frame: Dict[str, Any] = {}
    probe_errors: List[float] = []
    probe_count = 0
    gap_count = 0

    try:
        for left_idx, right_idx in zip(roi_kept[:-1], roi_kept[1:]):
            gap = int(right_idx - left_idx)
            if gap <= 1 or gap > int(max_probe_gap_frames):
                continue
            gap_count += 1
            left_frame = _video_frame_at(cap, int(left_idx))
            right_frame = _video_frame_at(cap, int(right_idx))
            union = _union_box_range(roi_payload, int(left_idx), int(right_idx))
            if union is None:
                continue
            clipped = _clip_box(
                union,
                width=width,
                height=height,
                margin_px=int(crop_margin_px),
            )
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            left_crop = _resize_if_needed(left_frame[y1:y2, x1:x2], int(max_crop_side))
            right_crop = _resize_if_needed(right_frame[y1:y2, x1:x2], int(max_crop_side))

            chosen_probe_frames: set[int] = set()
            for pos in probe_positions:
                probe_frame = int(round(float(left_idx) + (float(gap) * float(pos))))
                if probe_frame <= int(left_idx) or probe_frame >= int(right_idx):
                    continue
                if probe_frame in chosen_probe_frames:
                    continue
                chosen_probe_frames.add(probe_frame)
                alpha = float(probe_frame - int(left_idx)) / float(gap)
                target_frame = _video_frame_at(cap, int(probe_frame))
                target_crop = _resize_if_needed(target_frame[y1:y2, x1:x2], int(max_crop_side))
                pred_crop = interpolator.interpolate(left_crop, right_crop, alpha)
                err = float(np.mean(np.abs(pred_crop.astype(np.float32) - target_crop.astype(np.float32))) / 255.0)
                per_frame[str(int(probe_frame))] = {
                    "amt_risk": float(err),
                    "probe_error": float(err),
                    "left_anchor": int(left_idx),
                    "right_anchor": int(right_idx),
                    "alpha": float(alpha),
                    "source": "gpu_amt_probe",
                }
                probe_errors.append(float(err))
                probe_count += 1
    finally:
        cap.release()

    return {
        "meta": {
            "source": "gpu_amt_probe",
            "video_path": str(video_path_p),
            "probe_count": int(probe_count),
            "gap_count": int(gap_count),
            "crop_margin_px": int(crop_margin_px),
            "max_crop_side": int(max_crop_side),
            "probe_positions": list(probe_positions),
            "max_probe_gap_frames": int(max_probe_gap_frames),
            "probe_error_mean": float(sum(probe_errors) / len(probe_errors)) if probe_errors else 0.0,
            "probe_error_max": float(max(probe_errors)) if probe_errors else 0.0,
        },
        "per_frame": per_frame,
    }
