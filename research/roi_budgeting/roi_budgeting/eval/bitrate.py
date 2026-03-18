from __future__ import annotations

import math
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

from roi_budgeting.data.load_frame_drop import roi_keep_indices, roi_segments
from roi_budgeting.data.load_roi_detections import frame_boxes


def target_bytes_for_duration(*, duration_seconds: float, target_kbps: float) -> float:
    seconds = max(0.0, float(duration_seconds))
    kbps = max(0.0, float(target_kbps))
    return seconds * kbps * 1000.0 / 8.0


def bits_per_second(*, total_bytes: float, duration_seconds: float) -> float:
    seconds = max(1e-9, float(duration_seconds))
    return float(total_bytes) * 8.0 / seconds


def _quantile(values: Sequence[float], q: float) -> float:
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


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1c = max(0, min(width - 1, int(x1)))
    y1c = max(0, min(height - 1, int(y1)))
    x2c = max(0, min(width, int(x2)))
    y2c = max(0, min(height, int(y2)))
    if x2c <= x1c:
        x2c = min(width, x1c + 1)
    if y2c <= y1c:
        y2c = min(height, y1c + 1)
    return x1c, y1c, x2c, y2c


def _union_box_from_boxes(
    *,
    roi_payload: Mapping[str, Any],
    frame_idx: int,
    fallback_box: Mapping[str, Any] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    boxes = frame_boxes(roi_payload, frame_idx)
    if boxes:
        x1 = min(int(box.get("x1", 0) or 0) for box in boxes)
        y1 = min(int(box.get("y1", 0) or 0) for box in boxes)
        x2 = max(int(box.get("x2", 0) or 0) for box in boxes)
        y2 = max(int(box.get("y2", 0) or 0) for box in boxes)
        return _clip_box(x1, y1, x2, y2, width, height)

    if isinstance(fallback_box, Mapping):
        return _clip_box(
            int(fallback_box.get("x1", 0) or 0),
            int(fallback_box.get("y1", 0) or 0),
            int(fallback_box.get("x2", 0) or 0),
            int(fallback_box.get("y2", 0) or 0),
            width,
            height,
        )
    return None


def _texture_score(crop_bgr: np.ndarray, *, max_crop_side: int) -> float:
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > max(1, int(max_crop_side)):
        scale = float(max_crop_side) / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(lap)) / 255.0)


def _encode_probe_bytes(
    crop_bgr: np.ndarray,
    *,
    codec: str,
    quality: int,
    max_side: int,
) -> float:
    if crop_bgr.size == 0:
        return 0.0

    codec_name = str(codec).strip().lower()
    if codec_name not in {"jpeg", "jpg"}:
        raise ValueError(f"Unsupported probe codec: {codec}")

    h, w = crop_bgr.shape[:2]
    longest = max(h, w)
    encoded_crop = crop_bgr
    if longest > max(1, int(max_side)):
        scale = float(max_side) / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        encoded_crop = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(
        ".jpg",
        encoded_crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))],
    )
    if not ok:
        return 0.0
    return float(len(buf))


def _pad_probe_canvas(crop_bgr: np.ndarray, *, max_side: int) -> np.ndarray:
    canvas_side = max(2, int(max_side))
    if canvas_side % 2 != 0:
        canvas_side -= 1
    if crop_bgr.size == 0:
        return np.zeros((canvas_side, canvas_side, 3), dtype=np.uint8)
    h, w = crop_bgr.shape[:2]
    longest = max(h, w)
    canvas = np.zeros((canvas_side, canvas_side, 3), dtype=np.uint8)
    resized = crop_bgr
    if longest > max(1, canvas_side):
        scale = float(canvas_side) / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    y0 = max(0, (canvas_side - rh) // 2)
    x0 = max(0, (canvas_side - rw) // 2)
    canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
    return canvas


def _resize_long_side(frame_bgr: np.ndarray, *, max_side: int) -> np.ndarray:
    if frame_bgr.size == 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max(1, int(max_side)):
        return frame_bgr
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w % 2 != 0:
        new_w = max(2, new_w - 1)
    if new_h % 2 != 0:
        new_h = max(2, new_h - 1)
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _mask_frame_to_boxes(
    frame_bgr: np.ndarray,
    *,
    boxes: Sequence[Mapping[str, Any]],
    fallback_box: Mapping[str, Any] | None,
    width: int,
    height: int,
) -> np.ndarray:
    masked = np.zeros_like(frame_bgr)
    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = _clip_box(
                int(box.get("x1", 0) or 0),
                int(box.get("y1", 0) or 0),
                int(box.get("x2", 0) or 0),
                int(box.get("y2", 0) or 0),
                width,
                height,
            )
            masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
        return masked
    if isinstance(fallback_box, Mapping):
        x1, y1, x2, y2 = _clip_box(
            int(fallback_box.get("x1", 0) or 0),
            int(fallback_box.get("y1", 0) or 0),
            int(fallback_box.get("x2", 0) or 0),
            int(fallback_box.get("y2", 0) or 0),
            width,
            height,
        )
        masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
    return masked


def _encode_probe_video_bytes(
    frames_bgr: Sequence[np.ndarray],
    *,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
    max_side: int,
) -> list[float]:
    codec_name = str(codec).strip().lower()
    if codec_name not in {"h264", "x264"}:
        raise ValueError(f"Unsupported video probe codec: {codec}")
    if not frames_bgr:
        return []
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required for the ROI x264 probe model.")

    frame_h, frame_w = frames_bgr[0].shape[:2]
    with tempfile.TemporaryDirectory(prefix="roi_probe_x264_") as tmpdir:
        out_path = Path(tmpdir) / "probe.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{frame_w}x{frame_h}",
            "-r",
            f"{float(fps):.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-tune",
            "zerolatency",
            "-crf",
            str(int(crf)),
            "-g",
            "9999",
            "-keyint_min",
            "9999",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            for frame in frames_bgr:
                if frame.shape[:2] != (frame_h, frame_w):
                    raise ValueError("All probe video frames must share the same shape.")
                try:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                except BrokenPipeError as exc:
                    stderr = proc.stderr.read() if proc.stderr is not None else b""
                    proc.wait()
                    raise RuntimeError(
                        "ffmpeg ROI probe encode failed: "
                        f"{stderr.decode('utf-8', errors='ignore').strip()}"
                    ) from exc
            proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            proc.wait()
        finally:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg ROI probe encode failed: {stderr.decode('utf-8', errors='ignore').strip()}")

        probe_cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=size",
            "-of",
            "json",
            str(out_path),
        ]
        probe_proc = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        payload = json.loads(probe_proc.stdout or "{}")
        packets = payload.get("packets", []) or []
        sizes = [float(int(pkt.get("size", 0) or 0)) for pkt in packets if isinstance(pkt, dict)]
        if len(sizes) != len(frames_bgr):
            if len(sizes) < len(frames_bgr):
                sizes.extend([0.0] * (len(frames_bgr) - len(sizes)))
            else:
                sizes = sizes[: len(frames_bgr)]
        return sizes


def build_roi_bitrate_features(
    *,
    video_path: str | Path,
    roi_payload: Mapping[str, Any],
    frame_drop_json: Mapping[str, Any],
    max_crop_side: int = 160,
    probe_codec: str = "jpeg",
    probe_quality: int = 50,
    probe_max_side: int = 96,
    fps: float = 30.0,
    frame_segments: Sequence[Tuple[int, int]] | None = None,
    probe_frame_mode: str = "crop",
) -> Dict[int, Dict[str, float]]:
    video = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for bitrate features: {video}")

    per_frame = (frame_drop_json.get("per_frame", {}) or {}) if isinstance(frame_drop_json, dict) else {}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_area = float(max(1, width * height))

    out: Dict[int, Dict[str, float]] = {}
    probe_crops: Dict[int, np.ndarray] = {}
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rec = per_frame.get(str(frame_idx), {}) or {}
            roi_count = int(rec.get("roi_count", 0) or 0)
            has_roi = bool(roi_count > 0) and not bool(rec.get("bbox_missing", False))
            boxes = frame_boxes(roi_payload, frame_idx)
            union_box = _union_box_from_boxes(
                roi_payload=roi_payload,
                frame_idx=frame_idx,
                fallback_box=rec.get("roi_box", None) if isinstance(rec, dict) else None,
                width=width,
                height=height,
            )

            area_pixels = 0.0
            area_ratio = 0.0
            texture_raw = 0.0
            roi_mean_conf = float(rec.get("roi_mean_conf", 0.0) or 0.0)
            if has_roi and union_box is not None:
                x1, y1, x2, y2 = union_box
                area_pixels = float(max(0, x2 - x1) * max(0, y2 - y1))
                area_ratio = area_pixels / frame_area
                crop = frame[y1:y2, x1:x2]
                texture_raw = _texture_score(crop, max_crop_side=max_crop_side)
                codec_name = str(probe_codec).strip().lower()
                frame_mode = str(probe_frame_mode).strip().lower()
                if codec_name in {"jpeg", "jpg"}:
                    probe_encoded_bytes = _encode_probe_bytes(
                        crop,
                        codec=probe_codec,
                        quality=int(probe_quality),
                        max_side=int(probe_max_side),
                    )
                elif codec_name in {"h264", "x264"}:
                    probe_encoded_bytes = 0.0
                    if frame_mode == "masked_frame":
                        masked = _mask_frame_to_boxes(
                            frame,
                            boxes=boxes,
                            fallback_box=rec.get("roi_box", None) if isinstance(rec, dict) else None,
                            width=width,
                            height=height,
                        )
                        probe_crops[int(frame_idx)] = _resize_long_side(masked, max_side=int(probe_max_side))
                    else:
                        probe_crops[int(frame_idx)] = _pad_probe_canvas(crop, max_side=int(probe_max_side))
                else:
                    raise ValueError(f"Unsupported probe codec: {probe_codec}")
            else:
                probe_encoded_bytes = 0.0
            out[int(frame_idx)] = {
                "has_roi": 1.0 if has_roi else 0.0,
                "roi_pixel_area": float(area_pixels),
                "roi_area_ratio": float(area_ratio),
                "roi_box_count": float(max(0, roi_count)),
                "roi_mean_conf": float(roi_mean_conf),
                "texture_raw": float(texture_raw),
                "probe_encoded_bytes": float(probe_encoded_bytes),
            }
            frame_idx += 1
    finally:
        cap.release()

    codec_name = str(probe_codec).strip().lower()
    if codec_name in {"h264", "x264"} and probe_crops:
        segments = list(frame_segments or [])
        if not segments:
            all_indices = sorted(probe_crops.keys())
            if all_indices:
                segments = [(all_indices[0], all_indices[-1])]
        for start, end in segments:
            ordered_indices = [frame_idx for frame_idx in range(int(start), int(end) + 1) if frame_idx in probe_crops]
            if not ordered_indices:
                continue
            sizes = _encode_probe_video_bytes(
                [probe_crops[frame_idx] for frame_idx in ordered_indices],
                fps=float(fps),
                codec=probe_codec,
                crf=int(probe_quality),
                preset="veryfast",
                max_side=int(probe_max_side),
            )
            for frame_idx, size in zip(ordered_indices, sizes):
                out[int(frame_idx)]["probe_encoded_bytes"] = float(size)
    return out


def estimate_roi_anchor_bytes(
    *,
    bitrate_features: Mapping[int, Mapping[str, Any]],
    motion_scores: Mapping[int, Mapping[str, Any]],
    frame_drop_json: Mapping[str, Any],
    video_meta: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    budget_cfg = (cfg.get("budget", {}) or {}) if isinstance(cfg, dict) else {}
    estimator_cfg = (budget_cfg.get("bitrate_estimator", {}) or {}) if isinstance(budget_cfg, dict) else {}

    target_kbps = float(budget_cfg.get("roi_target_kbps", 0.0) or 0.0)
    duration_sec = float(video_meta.get("duration_sec", 0.0) or 0.0)
    target_bytes = target_bytes_for_duration(duration_seconds=duration_sec, target_kbps=target_kbps)
    model_name = str(estimator_cfg.get("model", "roi_proxy_v1") or "roi_proxy_v1").strip()
    calibration_name = str(
        estimator_cfg.get(
            "calibration",
            "direct_probe_bytes" if model_name.startswith("roi_probe_") else "match_fixed_baseline_at_target_kbps",
        )
        or ""
    ).strip()

    baseline_keep = roi_keep_indices(dict(frame_drop_json))

    if model_name.startswith("roi_probe_"):
        probe_codec = str(estimator_cfg.get("probe_codec", "jpeg") or "jpeg")
        probe_quality = int(estimator_cfg.get("probe_quality", 50) or 50)
        probe_max_side = int(estimator_cfg.get("probe_max_side", 96) or 96)
        probe_frame_mode = str(estimator_cfg.get("probe_frame_mode", "crop") or "crop")
        out: Dict[int, Dict[str, float]] = {}
        for frame_idx, rec in bitrate_features.items():
            probe_bytes = (
                float((rec.get("probe_encoded_bytes", 0.0) or 0.0))
                if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
                else 0.0
            )
            out[int(frame_idx)] = {
                **{k: float(v) for k, v in rec.items()},
                "motion_score": float((motion_scores.get(frame_idx, {}) or {}).get("motion_score", 0.0) or 0.0),
                "estimated_bytes": float(probe_bytes),
            }

        all_anchor_bytes = [
            float(rec.get("estimated_bytes", 0.0) or 0.0)
            for rec in out.values()
            if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
        ]
        baseline_estimated_bytes = float(
            sum(out.get(frame_idx, {}).get("estimated_bytes", 0.0) or 0.0 for frame_idx in baseline_keep)
        )
        baseline_estimated_kbps = bits_per_second(total_bytes=baseline_estimated_bytes, duration_seconds=duration_sec) / 1000.0
        meta = {
            "model": model_name,
            "calibration": calibration_name,
            "target_kbps": float(target_kbps),
            "target_bytes": float(target_bytes),
            "duration_sec": float(duration_sec),
            "baseline_keep_count": int(len(baseline_keep)),
            "baseline_estimated_bytes": float(baseline_estimated_bytes),
            "baseline_estimated_kbps": float(baseline_estimated_kbps),
            "probe_codec": probe_codec,
            "probe_quality": int(probe_quality),
            "probe_max_side": int(probe_max_side),
            "probe_frame_mode": probe_frame_mode,
            "estimated_anchor_bytes_mean": float(sum(all_anchor_bytes) / float(len(all_anchor_bytes))) if all_anchor_bytes else 0.0,
            "estimated_anchor_bytes_max": float(max(all_anchor_bytes)) if all_anchor_bytes else 0.0,
        }
        return out, meta

    base_weight = float(estimator_cfg.get("base_weight", 1.0) or 0.0)
    area_weight = float(estimator_cfg.get("area_weight", 3.5) or 0.0)
    texture_weight = float(estimator_cfg.get("texture_weight", 1.25) or 0.0)
    motion_weight = float(estimator_cfg.get("motion_weight", 0.75) or 0.0)
    count_weight = float(estimator_cfg.get("count_weight", 0.35) or 0.0)

    area_vals = [
        float((rec.get("roi_area_ratio", 0.0) or 0.0))
        for rec in bitrate_features.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]
    texture_vals = [
        float((rec.get("texture_raw", 0.0) or 0.0))
        for rec in bitrate_features.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]
    count_vals = [
        max(0.0, float((rec.get("roi_box_count", 0.0) or 0.0)) - 1.0)
        for rec in bitrate_features.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]

    area_scale = max(1e-6, _quantile(area_vals, 0.90) or max(area_vals or [1e-6]))
    texture_scale = max(1e-6, _quantile(texture_vals, 0.90) or max(texture_vals or [1e-6]))
    count_scale = max(1.0, _quantile(count_vals, 0.90) or max(count_vals or [1.0]))

    raw_units: Dict[int, float] = {}
    out: Dict[int, Dict[str, float]] = {}
    for frame_idx, rec in bitrate_features.items():
        has_roi = float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
        motion_score = float((motion_scores.get(frame_idx, {}) or {}).get("motion_score", 0.0) or 0.0)
        if not has_roi:
            raw_units[int(frame_idx)] = 0.0
            out[int(frame_idx)] = {
                **{k: float(v) for k, v in rec.items()},
                "motion_score": float(motion_score),
                "area_norm": 0.0,
                "texture_norm": 0.0,
                "count_norm": 0.0,
                "raw_byte_units": 0.0,
                "estimated_bytes": 0.0,
            }
            continue

        area_ratio = float(rec.get("roi_area_ratio", 0.0) or 0.0)
        texture_raw = float(rec.get("texture_raw", 0.0) or 0.0)
        count_extra = max(0.0, float(rec.get("roi_box_count", 0.0) or 0.0) - 1.0)

        area_norm = min(2.0, math.sqrt(max(0.0, area_ratio) / area_scale))
        texture_norm = min(2.0, max(0.0, texture_raw) / texture_scale)
        count_norm = min(2.0, count_extra / count_scale)
        motion_norm = min(1.5, max(0.0, motion_score))

        units = (
            base_weight
            + (area_weight * area_norm)
            + (texture_weight * texture_norm)
            + (motion_weight * motion_norm)
            + (count_weight * count_norm)
        )
        raw_units[int(frame_idx)] = float(units)
        out[int(frame_idx)] = {
            **{k: float(v) for k, v in rec.items()},
            "motion_score": float(motion_score),
            "area_norm": float(area_norm),
            "texture_norm": float(texture_norm),
            "count_norm": float(count_norm),
            "raw_byte_units": float(units),
            "estimated_bytes": 0.0,
        }

    baseline_raw_total = float(sum(raw_units.get(frame_idx, 0.0) for frame_idx in baseline_keep))
    if baseline_raw_total <= 1e-9:
        scale = 0.0
    else:
        scale = float(target_bytes) / baseline_raw_total

    for frame_idx, units in raw_units.items():
        out[int(frame_idx)]["estimated_bytes"] = float(units * scale)

    all_anchor_bytes = [
        float(rec.get("estimated_bytes", 0.0) or 0.0)
        for rec in out.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]
    baseline_estimated_bytes = float(sum(out.get(frame_idx, {}).get("estimated_bytes", 0.0) or 0.0 for frame_idx in baseline_keep))
    baseline_estimated_kbps = bits_per_second(total_bytes=baseline_estimated_bytes, duration_seconds=duration_sec) / 1000.0

    meta = {
        "model": model_name,
        "calibration": calibration_name,
        "target_kbps": float(target_kbps),
        "target_bytes": float(target_bytes),
        "duration_sec": float(duration_sec),
        "baseline_keep_count": int(len(baseline_keep)),
        "baseline_raw_total": float(baseline_raw_total),
        "baseline_estimated_bytes": float(baseline_estimated_bytes),
        "baseline_estimated_kbps": float(baseline_estimated_kbps),
        "byte_scale": float(scale),
        "area_scale": float(area_scale),
        "texture_scale": float(texture_scale),
        "count_scale": float(count_scale),
        "base_weight": float(base_weight),
        "area_weight": float(area_weight),
        "texture_weight": float(texture_weight),
        "motion_weight": float(motion_weight),
        "count_weight": float(count_weight),
        "estimated_anchor_bytes_mean": float(sum(all_anchor_bytes) / float(len(all_anchor_bytes))) if all_anchor_bytes else 0.0,
        "estimated_anchor_bytes_max": float(max(all_anchor_bytes)) if all_anchor_bytes else 0.0,
    }
    return out, meta


def summarize_selected_roi_bytes(
    *,
    frame_indices: Sequence[int],
    byte_estimates: Mapping[int, Mapping[str, Any]],
    duration_sec: float,
    target_bytes: float,
) -> Dict[str, float]:
    total_bytes = float(
        sum(float((byte_estimates.get(int(frame_idx), {}) or {}).get("estimated_bytes", 0.0) or 0.0) for frame_idx in frame_indices)
    )
    estimated_kbps = bits_per_second(total_bytes=total_bytes, duration_seconds=duration_sec) / 1000.0
    avg_bytes_per_anchor = (total_bytes / float(len(frame_indices))) if frame_indices else 0.0
    budget_utilization = (total_bytes / float(target_bytes)) if target_bytes > 1e-9 else 0.0
    return {
        "estimated_roi_bytes": float(total_bytes),
        "estimated_roi_kbps": float(estimated_kbps),
        "avg_bytes_per_anchor": float(avg_bytes_per_anchor),
        "budget_utilization": float(budget_utilization),
    }
