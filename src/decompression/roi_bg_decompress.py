# src/decompression/roi_bg_decompress.py
"""
ROI/BG decompression helpers (pure module).

This module decodes ROI/BG bitstreams stored in the archive into frame arrays.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _resolve_hint(meta_hint: Any, user_limit: Optional[int]) -> Optional[int]:
    hint: Optional[int] = None
    if meta_hint is not None:
        try:
            hint = int(meta_hint)
        except (TypeError, ValueError):
            hint = None
    if user_limit is None:
        return hint
    try:
        lim = int(user_limit)
    except (TypeError, ValueError):
        lim = -1
    if lim <= 0:
        return hint
    if hint is None:
        return lim
    return min(hint, lim)


def _stream_extension(codec_backend: str) -> str:
    backend = str(codec_backend).strip().lower()
    if backend == "av1":
        return ".ivf"
    raise ValueError(f"Unsupported archive compression backend: {codec_backend}. AV1 is the only supported codec.")


def _stream_compression_cfg(meta: Dict[str, Any], *, stream_name: str) -> Dict[str, Any]:
    compression_cfg = dict(meta.get("compression", {}) or {})
    compression_cfg["backend"] = "av1"
    return compression_cfg


def _decode_stream_bytes_ffmpeg(
    stream_bytes: bytes,
    compression_cfg: Dict[str, Any],
    video_info: Dict[str, Any],
    frame_count_hint: Optional[int] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    frame_consumer: Optional[Callable[[np.ndarray], None]] = None,
) -> List[np.ndarray]:
    ffmpeg_bin = str(compression_cfg.get("ffmpeg_bin", "ffmpeg")).strip() or "ffmpeg"
    backend = str(compression_cfg.get("backend", "hevc")).strip().lower()
    width = int(video_info.get("width", 0) or 0)
    height = int(video_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video width/height in archive metadata")

    frame_size = int(width) * int(height) * 3
    frames: List[np.ndarray] = []
    max_frames = int(frame_count_hint) if frame_count_hint is not None else None
    if max_frames is not None and max_frames < 0:
        max_frames = None

    with tempfile.NamedTemporaryFile(suffix=_stream_extension(backend), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(stream_bytes)
    try:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(tmp_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        frames_decoded = 0
        while True:
            if max_frames is not None and frames_decoded >= max_frames:
                break
            chunk = proc.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise RuntimeError("ffmpeg decode produced a partial frame")
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3)).copy()
            if frame_consumer is None:
                frames.append(frame)
            else:
                frame_consumer(frame)
            frames_decoded += 1
            if progress_cb is not None:
                progress_cb(1)
        if proc.stdout is not None:
            proc.stdout.close()
        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
        rc = proc.wait()
        if rc != 0:
            tail = "\n".join(stderr_text.splitlines()[-20:]) if stderr_text else "(no ffmpeg stderr)"
            raise RuntimeError(f"ffmpeg decode failed with exit code {rc}.\n{tail}")
        return frames
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def decode_stream_to_memmap(
    stream_bytes: bytes,
    meta: Dict[str, Any],
    *,
    stream_name: str,
    work_dir: Path,
    progress_cb: Optional[Callable[[int], None]] = None,
    max_frames: Optional[int] = None,
) -> Tuple[np.memmap, int]:
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    stream_meta = streams.get(str(stream_name), {}) or {}

    width = int(video_info.get("width", 0) or 0)
    height = int(video_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video width/height in archive metadata")

    count_hint = stream_meta.get("frames_encoded", None)
    if count_hint is None and isinstance(stream_meta.get("frame_index_map", None), list):
        count_hint = len(stream_meta.get("frame_index_map", []))
    resolved_hint = _resolve_hint(count_hint, max_frames)
    if resolved_hint is None:
        raise RuntimeError(f"Archive is missing {stream_name} frame count hints required for low-memory decode.")

    work_dir.mkdir(parents=True, exist_ok=True)
    stream_path = work_dir / f"{stream_name}_frames.npy"
    stream_map = np.lib.format.open_memmap(
        stream_path,
        mode="w+",
        dtype=np.uint8,
        shape=(int(resolved_hint), height, width, 3),
    )

    written = 0

    def _consume(frame: np.ndarray) -> None:
        nonlocal written
        if written >= int(stream_map.shape[0]):
            raise RuntimeError(f"Decoded more {stream_name} frames than expected from archive metadata.")
        stream_map[written] = frame
        written += 1

    _decode_stream_bytes_ffmpeg(
        stream_bytes,
        _stream_compression_cfg(meta, stream_name=str(stream_name)),
        video_info,
        frame_count_hint=resolved_hint,
        progress_cb=progress_cb,
        frame_consumer=_consume,
    )
    stream_map.flush()
    return stream_map, written


def decode_roi_bg_streams(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)
    roi_frames = _decode_stream_bytes_ffmpeg(
        roi_bin_bytes,
        _stream_compression_cfg(meta, stream_name="roi"),
        video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
    )
    bg_frames = _decode_stream_bytes_ffmpeg(
        bg_bin_bytes,
        _stream_compression_cfg(meta, stream_name="bg"),
        video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
    )
    return roi_frames, bg_frames


def decode_roi_bg_streams_to_memmap(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    work_dir: Path,
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[np.memmap, int, np.memmap, int]:
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    width = int(video_info.get("width", 0) or 0)
    height = int(video_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video width/height in archive metadata")

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)
    if roi_hint is None or bg_hint is None:
        raise RuntimeError("Archive is missing ROI/BG frame count hints required for low-memory decode.")

    work_dir.mkdir(parents=True, exist_ok=True)
    roi_path = work_dir / "roi_frames.npy"
    bg_path = work_dir / "bg_frames.npy"
    roi_map = np.lib.format.open_memmap(roi_path, mode="w+", dtype=np.uint8, shape=(int(roi_hint), height, width, 3))
    bg_map = np.lib.format.open_memmap(bg_path, mode="w+", dtype=np.uint8, shape=(int(bg_hint), height, width, 3))

    roi_written = 0
    bg_written = 0

    def _consume_roi(frame: np.ndarray) -> None:
        nonlocal roi_written
        if roi_written >= int(roi_map.shape[0]):
            raise RuntimeError("Decoded more ROI frames than expected from archive metadata.")
        roi_map[roi_written] = frame
        roi_written += 1

    def _consume_bg(frame: np.ndarray) -> None:
        nonlocal bg_written
        if bg_written >= int(bg_map.shape[0]):
            raise RuntimeError("Decoded more BG frames than expected from archive metadata.")
        bg_map[bg_written] = frame
        bg_written += 1

    _decode_stream_bytes_ffmpeg(
        roi_bin_bytes,
        _stream_compression_cfg(meta, stream_name="roi"),
        video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
        frame_consumer=_consume_roi,
    )
    _decode_stream_bytes_ffmpeg(
        bg_bin_bytes,
        _stream_compression_cfg(meta, stream_name="bg"),
        video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
        frame_consumer=_consume_bg,
    )
    roi_map.flush()
    bg_map.flush()
    return roi_map, roi_written, bg_map, bg_written


def decode_roi_bg_streams_to_cache(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    work_dir: Path,
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[List[Path], int, List[Path], int]:
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)
    if roi_hint is None or bg_hint is None:
        raise RuntimeError("Archive is missing ROI/BG frame count hints required for low-memory decode.")

    roi_dir = work_dir / "roi"
    bg_dir = work_dir / "bg"
    roi_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    roi_paths: List[Path] = []
    bg_paths: List[Path] = []
    roi_written = 0
    bg_written = 0

    def _write_frame(dst_dir: Path, idx: int, frame: np.ndarray, suffix: str, params: List[int], out: List[Path]) -> None:
        path = dst_dir / f"{idx:06d}{suffix}"
        ok = cv2.imwrite(str(path), frame, params)
        if not ok:
            raise RuntimeError(f"Failed to write decoded temp frame: {path}")
        out.append(path)

    def _consume_roi(frame: np.ndarray) -> None:
        nonlocal roi_written
        _write_frame(roi_dir, roi_written, frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 97], roi_paths)
        roi_written += 1

    def _consume_bg(frame: np.ndarray) -> None:
        nonlocal bg_written
        _write_frame(bg_dir, bg_written, frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95], bg_paths)
        bg_written += 1

    _decode_stream_bytes_ffmpeg(
        roi_bin_bytes,
        _stream_compression_cfg(meta, stream_name="roi"),
        video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
        frame_consumer=_consume_roi,
    )
    _decode_stream_bytes_ffmpeg(
        bg_bin_bytes,
        _stream_compression_cfg(meta, stream_name="bg"),
        video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
        frame_consumer=_consume_bg,
    )
    return roi_paths, roi_written, bg_paths, bg_written
