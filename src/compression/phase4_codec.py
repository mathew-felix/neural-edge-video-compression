from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np

from .video_io import VideoInfo, probe_video
try:
    from roi_masking import boxes_for_frame as _boxes_for_frame_shared, build_boxes_mask, crop_bbox_from_boxes
except ImportError:  # pragma: no cover - test import path fallback
    from src.roi_masking import boxes_for_frame as _boxes_for_frame_shared, build_boxes_mask, crop_bbox_from_boxes


def _resolve_path(raw: str | Path, root_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root_dir / p).resolve()


def _pick_indices(frame_drop_result: Dict[str, Any], key: str, fallback_key: str) -> List[int]:
    vals = frame_drop_result.get(key, None)
    if vals is None:
        vals = frame_drop_result.get(fallback_key, None)
    if vals is None:
        vals = []
    return [int(x) for x in vals]


def _resolve_frame_index_map(
    *,
    stream_name: str,
    encoded_meta: Dict[str, Any],
    requested_indices: List[int],
) -> List[int]:
    target_len = int(encoded_meta.get("frames_encoded", 0) or 0)
    if target_len <= 0:
        return []

    authoritative = encoded_meta.get("frame_index_map", None)
    if isinstance(authoritative, list) and authoritative:
        resolved = [int(x) for x in authoritative]
        source_name = "encoder metadata"
    else:
        resolved = [int(x) for x in requested_indices[:target_len]]
        source_name = "requested indices"

    if len(resolved) != target_len:
        raise RuntimeError(
            f"{stream_name} frame_index_map length mismatch: expected {target_len}, got {len(resolved)} from {source_name}."
        )
    if any(resolved[i] >= resolved[i + 1] for i in range(len(resolved) - 1)):
        raise RuntimeError(f"{stream_name} frame_index_map must be strictly increasing.")
    return resolved


def _boxes_for_frame(roi_bbox_map: Mapping[Any, Any], frame_idx: int) -> List[Any]:
    return _boxes_for_frame_shared(roi_bbox_map, frame_idx)


def _apply_roi_mask(frame: np.ndarray, boxes: List[Any], *, roi_min_conf: float = 0.0, crop_margin_px: int = 0) -> np.ndarray:
    out = frame.copy()
    if not boxes:
        out[:] = 0
        return out
    h, w = out.shape[:2]
    mask = build_boxes_mask(
        width=int(w),
        height=int(h),
        boxes=boxes,
        roi_min_conf=float(roi_min_conf),
        roi_dilate_px=0,
    )
    out[mask == 0] = 0
    return out


def _bbox_iou(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, area_a + area_b - inter)


class _RandomAccessReader:
    def __init__(self, video_path: Path) -> None:
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video for temporal BG fill: {video_path}")
        self._cache: Dict[int, np.ndarray] = {}

    def read(self, frame_idx: int) -> np.ndarray | None:
        idx = int(frame_idx)
        cached = self._cache.get(idx)
        if cached is not None:
            return cached.copy()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._cache[idx] = frame.copy()
        if len(self._cache) > 12:
            first_key = next(iter(self._cache))
            self._cache.pop(first_key, None)
        return frame

    def close(self) -> None:
        self._cap.release()


def _build_background_model(
    *,
    video_path: Path,
    bg_kept_frames: List[int],
    max_samples: int,
) -> np.ndarray | None:
    keep = [int(x) for x in bg_kept_frames]
    if not keep:
        return None
    sample_cap = max(1, int(max_samples))
    if len(keep) > sample_cap:
        idxs = np.linspace(0, len(keep) - 1, num=sample_cap, dtype=int)
        sample_frames = [keep[int(i)] for i in idxs.tolist()]
    else:
        sample_frames = keep

    reader = _RandomAccessReader(video_path)
    stack: List[np.ndarray] = []
    try:
        for frame_idx in sample_frames:
            frame = reader.read(int(frame_idx))
            if frame is not None:
                stack.append(frame)
    finally:
        reader.close()
    if not stack:
        return None
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def _blend_patch_with_feather(
    *,
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    patch: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    out = frame.copy()
    if x2 <= x1 or y2 <= y1:
        return out
    if patch.shape[:2] != (y2 - y1, x2 - x1):
        raise ValueError("BG fill patch shape does not match target bbox.")
    feather = max(0, int(feather_px))
    if feather <= 0:
        out[y1:y2, x1:x2] = patch
        return out

    h = int(y2 - y1)
    w = int(x2 - x1)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, :] = 255
    blur_k = max(3, 2 * feather + 1)
    alpha = cv2.GaussianBlur(mask, (blur_k, blur_k), float(feather))
    alpha_f = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]

    base = out[y1:y2, x1:x2].astype(np.float32)
    fill = patch.astype(np.float32)
    blended = fill * alpha_f + base * (1.0 - alpha_f)
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _fill_bg_roi_temporal(
    *,
    frame: np.ndarray,
    frame_idx: int,
    bbox: tuple[int, int, int, int],
    reader: _RandomAccessReader,
    bg_model: np.ndarray | None,
    roi_bbox_map: Mapping[Any, Any],
    roi_min_conf: float,
    crop_margin_px: int,
    search_radius: int,
    max_iou: float,
    min_donors: int,
    max_donors: int,
    fallback_inpaint_radius: int,
    feather_px: int,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    out = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    if bg_model is not None:
        return _blend_patch_with_feather(
            frame=out,
            bbox=bbox,
            patch=bg_model[y1:y2, x1:x2],
            feather_px=int(feather_px),
        )
    donors: List[np.ndarray] = []
    radius = max(1, int(search_radius))
    donor_target = max(int(min_donors), 1)
    donor_cap = max(donor_target, int(max_donors))
    for step in range(1, radius + 1):
        for cand_idx in (int(frame_idx) - step, int(frame_idx) + step):
            if cand_idx < 0:
                continue
            cand_boxes = _boxes_for_frame(roi_bbox_map, cand_idx)
            cand_bbox = crop_bbox_from_boxes(
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                boxes=cand_boxes,
                roi_min_conf=float(roi_min_conf),
                margin_px=int(crop_margin_px),
            )
            if _bbox_iou(bbox, cand_bbox) > float(max_iou):
                continue
            donor = reader.read(cand_idx)
            if donor is None:
                continue
            donors.append(donor[y1:y2, x1:x2].copy())
            if len(donors) >= donor_cap:
                break
        if len(donors) >= donor_cap:
            break
    if len(donors) >= donor_target:
        if len(donors) == 1:
            fill_patch = donors[0]
        else:
            fill_patch = np.median(np.stack(donors, axis=0), axis=0).astype(np.uint8)
        return _blend_patch_with_feather(
            frame=out,
            bbox=bbox,
            patch=fill_patch,
            feather_px=int(feather_px),
        )
    return cv2.inpaint(out, mask, float(max(1, int(fallback_inpaint_radius))), cv2.INPAINT_TELEA)


def _iter_kept_roi_frames(
    *,
    video_path: Path,
    roi_kept_frames: List[int],
    roi_bbox_map: Mapping[Any, Any],
    roi_min_conf: float,
    crop_margin_px: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    roi_keep_set = set(int(x) for x in roi_kept_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for ROI keep streaming: {video_path}")

    src_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx in roi_keep_set:
                yield int(src_idx), _apply_roi_mask(
                    frame,
                    _boxes_for_frame(roi_bbox_map, src_idx),
                    roi_min_conf=float(roi_min_conf),
                    crop_margin_px=int(crop_margin_px),
                )
            src_idx += 1
    finally:
        cap.release()


def _iter_kept_bg_frames(
    *,
    video_path: Path,
    bg_kept_frames: List[int],
    roi_bbox_map: Mapping[Any, Any],
    roi_min_conf: float,
    bg_remove_margin_px: int,
    bg_inpaint_radius: int,
    bg_fill_search_radius: int,
    bg_fill_max_iou: float,
    bg_fill_min_donors: int,
    bg_fill_max_donors: int,
    bg_model_max_samples: int,
    bg_fill_feather_px: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    bg_keep_set = set(int(x) for x in bg_kept_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for BG keep streaming: {video_path}")

    src_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx in bg_keep_set:
                yield int(src_idx), frame
            src_idx += 1
    finally:
        cap.release()


def _codec_output_spec(codec_cfg: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    backend = str(codec_cfg.get("backend", "av1")).strip().lower()
    gop = max(1, int(codec_cfg.get("gop", 32)))

    if backend == "av1":
        encoder, encoder_args = _av1_encoder_args(codec_cfg, gop=gop)
        return ".ivf", "ivf", ["-c:v", encoder, *encoder_args]
    raise ValueError("compression.codec.backend must be 'av1'")


def _stream_codec_cfg(codec_cfg: Dict[str, Any], *, stream_name: str) -> Dict[str, Any]:
    out = dict(codec_cfg)
    out["backend"] = "av1"
    return out


def stream_extension_for_backend(backend: str) -> str:
    b = str(backend).strip().lower()
    if b == "av1":
        return ".ivf"
    raise ValueError(f"Unsupported stream backend: {backend}. AV1 is the only supported codec.")


def _ffmpeg_encoders(ffmpeg_bin: str) -> str:
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to query ffmpeg encoders via {ffmpeg_bin}.") from exc
    return str(proc.stdout)


def _resolve_av1_encoder(codec_cfg: Dict[str, Any]) -> str:
    ffmpeg_bin = str(codec_cfg.get("ffmpeg_bin", "ffmpeg")).strip() or "ffmpeg"
    requested = str(codec_cfg.get("av1_encoder", "auto")).strip().lower() or "auto"
    encoder_list = _ffmpeg_encoders(ffmpeg_bin)

    svt_ok = "libsvtav1" in encoder_list
    aom_ok = "libaom-av1" in encoder_list

    if requested == "auto":
        if svt_ok:
            return "libsvtav1"
        if aom_ok:
            return "libaom-av1"
        raise RuntimeError("No supported AV1 encoder found in ffmpeg. Expected libsvtav1 or libaom-av1.")
    if requested == "libsvtav1":
        if not svt_ok:
            raise RuntimeError("compression.codec.av1_encoder=libsvtav1 requested, but ffmpeg does not provide it.")
        return "libsvtav1"
    if requested == "libaom-av1":
        if not aom_ok:
            raise RuntimeError("compression.codec.av1_encoder=libaom-av1 requested, but ffmpeg does not provide it.")
        return "libaom-av1"
    raise ValueError("compression.codec.av1_encoder must be one of: auto, libsvtav1, libaom-av1")


def _av1_encoder_args(codec_cfg: Dict[str, Any], *, gop: int) -> Tuple[str, List[str]]:
    encoder = _resolve_av1_encoder(codec_cfg)
    if encoder == "libsvtav1":
        preset = str(codec_cfg.get("preset", "8")).strip() or "8"
        return encoder, ["-preset", preset, "-g", str(gop)]

    cpu_used = str(codec_cfg.get("cpu_used", "8")).strip() or "8"
    row_mt = str(codec_cfg.get("row_mt", "1")).strip() or "1"
    tiles = str(codec_cfg.get("tiles", "2x1")).strip() or "2x1"
    return encoder, [
        "-cpu-used",
        cpu_used,
        "-row-mt",
        row_mt,
        "-tiles",
        tiles,
        "-g",
        str(gop),
    ]


def _encode_ffmpeg_stream(
    frames: Iterable[Tuple[int, np.ndarray]],
    *,
    info: VideoInfo,
    codec_cfg: Dict[str, Any],
    qp: int,
) -> Dict[str, Any]:
    ffmpeg_bin = str(codec_cfg.get("ffmpeg_bin", "ffmpeg")).strip() or "ffmpeg"
    pix_fmt = str(codec_cfg.get("pix_fmt", "yuv420p")).strip() or "yuv420p"
    ext, muxer, codec_args = _codec_output_spec(codec_cfg)
    stderr_text = ""
    av1_encoder = _resolve_av1_encoder(codec_cfg)

    frame_index_map: List[int] = []
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{int(info.width)}x{int(info.height)}",
            "-r",
            str(float(max(1e-6, info.fps))),
            "-i",
            "-",
            "-an",
            *codec_args,
            "-pix_fmt",
            pix_fmt,
        ]
        if av1_encoder == "libsvtav1":
            cmd.extend(["-qp", str(int(qp))])
        else:
            # libaom-av1 is most reliable in constant-quality mode on Jetson-style builds.
            cmd.extend(["-crf", str(int(qp)), "-b:v", "0"])
        cmd.extend([
            "-f",
            muxer,
            str(out_path),
        ])
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for src_idx, frame in frames:
                if frame.shape[:2] != (int(info.height), int(info.width)):
                    raise ValueError("Stream encoder received a frame with the wrong size")
                if frame.dtype != np.uint8:
                    raise ValueError("Stream encoder requires uint8 frames")
                proc.stdin.write(frame.tobytes())
                frame_index_map.append(int(src_idx))
        except BrokenPipeError as exc:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.stderr is not None:
                stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
            rc = proc.wait()
            tail = "\n".join(stderr_text.splitlines()[-20:]) if stderr_text else "(no ffmpeg stderr)"
            raise RuntimeError(f"ffmpeg encode failed early with exit code {rc}.\n{tail}") from exc
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stderr is not None:
            stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
        rc = proc.wait()
        if rc != 0:
            tail = "\n".join(stderr_text.splitlines()[-20:]) if stderr_text else "(no ffmpeg stderr)"
            raise RuntimeError(f"ffmpeg encode failed with exit code {rc}.\n{tail}")
        bitstream = out_path.read_bytes()
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "bitstream_bytes": bitstream,
        "meta": {
            "frames_encoded": int(len(frame_index_map)),
            "frame_index_map": frame_index_map,
            "qp": int(qp),
            "av1_encoder": str(av1_encoder),
            "compressed_bytes": int(len(bitstream)),
        },
    }


def compress_keep_streams(
    *,
    source_video_path: str | Path,
    roi_bbox_map: Mapping[Any, Any],
    frame_drop_result: Dict[str, Any],
    compression_cfg: Dict[str, Any],
    root_dir: str | Path,
) -> Dict[str, Any]:
    root = Path(root_dir).expanduser().resolve()
    source_video = _resolve_path(source_video_path, root)
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    codec_cfg = dict(compression_cfg.get("codec", {}) or {})
    quality_cfg = dict(compression_cfg.get("quality", {}) or {})
    roi_cfg = dict(compression_cfg.get("roi", {}) or {})

    roi_qp = int(quality_cfg.get("roi_qp", 18))
    bg_qp = int(quality_cfg.get("bg_qp", 35))
    min_conf = float(roi_cfg.get("min_conf", 0.25))
    requested_dilate_px = int(roi_cfg.get("dilate_px", 0))
    crop_margin_px = int(roi_cfg.get("crop_margin_px", 24))
    bg_remove_margin_px = int(roi_cfg.get("bg_remove_margin_px", crop_margin_px))
    bg_inpaint_radius = int(roi_cfg.get("bg_inpaint_radius", 5))
    bg_fill_search_radius = int(roi_cfg.get("bg_fill_search_radius", 24))
    bg_fill_max_iou = float(roi_cfg.get("bg_fill_max_iou", 0.05))
    bg_fill_min_donors = int(roi_cfg.get("bg_fill_min_donors", 4))
    bg_fill_max_donors = int(roi_cfg.get("bg_fill_max_donors", 8))
    bg_model_max_samples = int(roi_cfg.get("bg_model_max_samples", 24))
    bg_fill_feather_px = int(roi_cfg.get("bg_fill_feather_px", 10))

    src = probe_video(str(source_video))
    info = VideoInfo(width=int(src.width), height=int(src.height), fps=float(src.fps), frames=int(src.frames))
    roi_indices_requested = _pick_indices(frame_drop_result, "roi_kept_frames", "kept_frames")
    bg_indices_requested = _pick_indices(frame_drop_result, "bg_kept_frames", "kept_frames")

    roi_codec_cfg = _stream_codec_cfg(codec_cfg, stream_name="roi")
    bg_codec_cfg = _stream_codec_cfg(codec_cfg, stream_name="bg")

    roi_encoded = _encode_ffmpeg_stream(
        _iter_kept_roi_frames(
            video_path=source_video,
            roi_kept_frames=roi_indices_requested,
            roi_bbox_map=roi_bbox_map,
            roi_min_conf=float(min_conf),
            crop_margin_px=int(crop_margin_px),
        ),
        info=info,
        codec_cfg=roi_codec_cfg,
        qp=int(roi_qp),
    )
    bg_encoded = _encode_ffmpeg_stream(
        _iter_kept_bg_frames(
            video_path=source_video,
            bg_kept_frames=bg_indices_requested,
            roi_bbox_map=roi_bbox_map,
            roi_min_conf=float(min_conf),
            bg_remove_margin_px=int(bg_remove_margin_px),
            bg_inpaint_radius=int(bg_inpaint_radius),
            bg_fill_search_radius=int(bg_fill_search_radius),
            bg_fill_max_iou=float(bg_fill_max_iou),
            bg_fill_min_donors=int(bg_fill_min_donors),
            bg_fill_max_donors=int(bg_fill_max_donors),
            bg_model_max_samples=int(bg_model_max_samples),
            bg_fill_feather_px=int(bg_fill_feather_px),
        ),
        info=info,
        codec_cfg=bg_codec_cfg,
        qp=int(bg_qp),
    )

    roi_bytes = bytes(roi_encoded["bitstream_bytes"])
    bg_bytes = bytes(bg_encoded["bitstream_bytes"])
    roi_meta = dict(roi_encoded.get("meta", {}) or {})
    bg_meta = dict(bg_encoded.get("meta", {}) or {})
    roi_indices = _resolve_frame_index_map(stream_name="ROI", encoded_meta=roi_meta, requested_indices=roi_indices_requested)
    bg_indices = _resolve_frame_index_map(stream_name="BG", encoded_meta=bg_meta, requested_indices=bg_indices_requested)
    roi_backend = "av1"
    bg_backend = "av1"

    meta: Dict[str, Any] = {
        "video": {
            "path": str(source_video),
            "width": int(src.width),
            "height": int(src.height),
            "fps": float(src.fps),
            "frames_total": int(src.frames),
            "start_frame": 0,
            "end_frame": int(max(0, src.frames - 1)),
        },
        "roi": {
            "min_conf": float(min_conf),
            "dilate_px": 0,
            "visible_dilate_px": 0,
            "requested_dilate_px": int(requested_dilate_px),
            "crop_margin_px": int(crop_margin_px),
            "bg_remove_margin_px": int(bg_remove_margin_px),
        },
        "quality": {
            "roi_qp": int(roi_qp),
            "bg_qp": int(bg_qp),
        },
        "compression": {
            "backend": "av1",
            "ffmpeg_bin": str(codec_cfg.get("ffmpeg_bin", "ffmpeg")),
            "gop": int(codec_cfg.get("gop", 32)),
            "pix_fmt": str(codec_cfg.get("pix_fmt", "yuv420p")),
            "av1_encoder": str(roi_meta.get("av1_encoder", bg_meta.get("av1_encoder", "unknown"))),
            "preset": str(codec_cfg.get("preset", "8")),
            "cpu_used": str(codec_cfg.get("cpu_used", "8")),
            "row_mt": str(codec_cfg.get("row_mt", "1")),
            "tiles": str(codec_cfg.get("tiles", "2x1")),
        },
        "streams": {
            "roi": {
                "source_keep_render": "direct_frame_stream",
                "frames_encoded": int(roi_meta.get("frames_encoded", 0)),
                "frame_index_map": roi_indices,
                "qp": int(roi_qp),
                "codec": roi_backend,
                "compressed_bytes": int(len(roi_bytes)),
            },
            "bg": {
                "source_keep_render": "direct_frame_stream",
                "frames_encoded": int(bg_meta.get("frames_encoded", 0)),
                "frame_index_map": bg_indices,
                "qp": int(bg_qp),
                "codec": bg_backend,
                "compressed_bytes": int(len(bg_bytes)),
            },
        },
        "sizes": {
            "roi_bytes": int(len(roi_bytes)),
            "bg_bytes": int(len(bg_bytes)),
        },
        "kept_frames_used": {
            "roi": roi_indices,
            "bg": bg_indices,
        },
    }

    return {
        "roi_stream_bytes": roi_bytes,
        "bg_stream_bytes": bg_bytes,
        "meta": meta,
    }
