from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def _project_venv_python() -> str:
    """Prefer repo virtualenv (`venv` or `.venv`) so subprocesses match project deps."""
    if sys.platform.startswith("win"):
        candidates = [
            ROOT / "venv" / "Scripts" / "python.exe",
            ROOT / ".venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [ROOT / "venv" / "bin" / "python", ROOT / ".venv" / "bin" / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return sys.executable

from _codec_quality import (  # noqa: E402
    aggregate_metrics,
    ms_ssim_full_bgr,
    ms_ssim_roi_bgr,
    psnr_full_bgr,
    psnr_roi_bgr,
    require_skimage,
)
from _phase0_utils import read_archive_payloads  # noqa: E402
from roi_masking import build_frame_mask  # noqa: E402


DEFAULT_VIDEO_DIR = ROOT / "data" / "test_videos"
DEFAULT_COMP_CFG = ROOT / "configs" / "gpu" / "compression.yaml"
DEFAULT_DEC_CFG = ROOT / "configs" / "gpu" / "decompression.yaml"
DEFAULT_OUT_DIR = ROOT / "outputs" / "codec_benchmark"

RESULTS_CSV_FIELDS: List[str] = [
    "video",
    "roi_profile",
    "bg_profile",
    "archive_size_bytes",
    "encode_time_sec",
    "decode_time_sec",
    "throughput_fps_encode",
    "roi_psnr_mean",
    "roi_ms_ssim_mean",
    "full_psnr_mean",
    "full_ms_ssim_mean",
    "metrics_frames_used",
    "metrics_stride",
]


ROI_CANDIDATES: List[Dict[str, Any]] = [
    {"name": "roi_av1_svt_fast_qp28", "codec": "av1", "encoder": "libsvtav1", "preset": "fast", "qp": 28},
    {"name": "roi_av1_aom_fast_qp28", "codec": "av1", "encoder": "libaom-av1", "preset": "fast", "qp": 28},
    {"name": "roi_av1_rav1e_fast_qp28", "codec": "av1", "encoder": "librav1e", "preset": "fast", "qp": 28},
    {"name": "roi_hevc_x265_fast_qp28", "codec": "hevc", "encoder": "libx265", "preset": "fast", "qp": 28},
    {"name": "roi_h264_x264_fast_qp24", "codec": "h264", "encoder": "libx264", "preset": "fast", "qp": 24},
    {"name": "roi_h264_nvenc_p4_qp24", "codec": "h264", "encoder": "h264_nvenc", "preset": "p4", "qp": 24},
    {"name": "roi_hevc_nvenc_p4_qp28", "codec": "hevc", "encoder": "hevc_nvenc", "preset": "p4", "qp": 28},
]

BG_CANDIDATES: List[Dict[str, Any]] = [
    {"name": "bg_hevc_x265_fast_qp36", "codec": "hevc", "encoder": "libx265", "preset": "fast", "qp": 36},
    {"name": "bg_hevc_x265_medium_qp35", "codec": "hevc", "encoder": "libx265", "preset": "medium", "qp": 35},
    {"name": "bg_av1_svt_fast_qp40", "codec": "av1", "encoder": "libsvtav1", "preset": "fast", "qp": 40},
    {"name": "bg_av1_rav1e_fast_qp40", "codec": "av1", "encoder": "librav1e", "preset": "fast", "qp": 40},
    {"name": "bg_h264_x264_fast_qp32", "codec": "h264", "encoder": "libx264", "preset": "fast", "qp": 32},
    {"name": "bg_h264_nvenc_p4_qp32", "codec": "h264", "encoder": "h264_nvenc", "preset": "p4", "qp": 32},
    {"name": "bg_hevc_nvenc_p4_qp36", "codec": "hevc", "encoder": "hevc_nvenc", "preset": "p4", "qp": 36},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ROI/BG codec pairs on test_videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Resume (default): if codec_pair_results.csv already exists under --out-dir, completed "
            "(video, roi_profile, bg_profile) rows are skipped and new rows are appended. "
            "Pass --no-resume to truncate the CSV and re-run everything."
        ),
    )
    parser.add_argument("--video-dir", type=str, default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument(
        "--video-names",
        type=str,
        default="",
        help="Comma-separated basenames inside --video-dir (overrides sorting + --max-videos)",
    )
    parser.add_argument("--compression-config", type=str, default=str(DEFAULT_COMP_CFG))
    parser.add_argument("--decompression-config", type=str, default=str(DEFAULT_DEC_CFG))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-videos", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip PSNR / MS-SSIM (ROI + full-frame) after decompress",
    )
    parser.add_argument(
        "--metrics-stride",
        type=int,
        default=1,
        help="Evaluate every Nth frame pair (speedup for long clips)",
    )
    parser.add_argument(
        "--max-metric-frames",
        type=int,
        default=0,
        help="Maximum number of frames to evaluate (0 = unlimited, still subject to stride)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="If > 0, only first K ROI×BG combinations (enumeration order)",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Proceed even when torch CUDA is unavailable (will likely fail once compression starts)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Truncate codec_pair_results.csv and re-run all (video, codec-pair) combinations",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return data


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _run(cmd: List[str]) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout[-2000:]
            + "\nstderr:\n"
            + proc.stderr[-2000:]
        )
    return round(dt, 3)


def _ffmpeg_encoders(ffmpeg_bin: str = "ffmpeg") -> set[str]:
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return set()
    encoders: set[str] = set()
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("Encoders:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1].strip())
    return encoders


def _iter_videos(video_dir: Path, max_videos: int) -> List[Path]:
    videos = sorted(video_dir.glob("*.mp4"))
    if max_videos > 0:
        videos = videos[:max_videos]
    return videos


def _videos_for_run(video_dir: Path, video_names: str, max_videos: int) -> List[Path]:
    names = str(video_names or "").strip()
    if names:
        out: List[Path] = []
        for raw in names.replace(",", " ").split():
            part = raw.strip()
            if not part:
                continue
            p = (video_dir / part).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Video not found: {p}")
            out.append(p)
        return out
    return _iter_videos(video_dir, max_videos)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _csv_metric(metrics: Dict[str, Any], key: str) -> str:
    v = metrics.get(key, np.nan)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{v:.6f}" if np.isfinite(v) else ""


def _load_completed_benchmark_keys(csv_path: Path, fieldnames: List[str]) -> set[tuple[str, str, str]]:
    """Return (video, roi_profile, bg_profile) keys already logged in CSV."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return set()
    keys: set[tuple[str, str, str]] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fh = reader.fieldnames
        if set(fh or []) != set(fieldnames):
            raise ValueError(
                f"{csv_path} columns do not match this script; got {sorted(fh or [])!r}, "
                f"expected {sorted(fieldnames)!r}. Use --no-resume or fix the CSV."
            )
        for row in reader:
            if not row:
                continue
            v = (row.get("video") or "").strip()
            r = (row.get("roi_profile") or "").strip()
            b = (row.get("bg_profile") or "").strip()
            if v and r and b:
                keys.add((v, r, b))
    return keys


def _evaluate_pair_metrics(
    *,
    original: Path,
    recon: Path,
    archive: Path,
    stride: int,
    max_metric_frames: int,
) -> Dict[str, Any]:
    require_skimage()
    payloads = read_archive_payloads(archive)
    meta = json.loads(payloads["meta.json"].decode("utf-8"))
    roi_meta = meta.get("roi", {}) or {}
    roi_min_conf = float(roi_meta.get("min_conf", 0.0))
    roi_dilate_px = int(roi_meta.get("visible_dilate_px", roi_meta.get("dilate_px", 0)))
    roi_json = json.loads(payloads["roi_detections.json"].decode("utf-8"))
    frame_drop_json = json.loads(payloads["frame_drop.json"].decode("utf-8"))
    boxes_map = roi_json.get("frames", {}) or {}

    cap_a = cv2.VideoCapture(str(original))
    cap_b = cv2.VideoCapture(str(recon))
    if not cap_a.isOpened() or not cap_b.isOpened():
        cap_a.release()
        cap_b.release()
        raise RuntimeError(f"Failed to open video for metrics: {original} or {recon}")

    w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st = max(1, int(stride))
    samples: Dict[str, list[float]] = defaultdict(list)
    frame_idx = -1
    metric_frames = 0

    try:
        while True:
            ok_a, frm_a = cap_a.read()
            ok_b, frm_b = cap_b.read()
            if not ok_a or not ok_b or frm_a is None or frm_b is None:
                break
            frame_idx += 1
            if frame_idx % st != 0:
                continue
            if frm_a.shape[:2] != frm_b.shape[:2]:
                frm_b = cv2.resize(frm_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
            h, w = frm_a.shape[:2]
            mask = build_frame_mask(
                frame_idx=frame_idx,
                width=w,
                height=h,
                mask_source="roi_detection",
                roi_boxes_map=boxes_map if isinstance(boxes_map, dict) else {},
                frame_drop_json=frame_drop_json if isinstance(frame_drop_json, dict) else {},
                roi_min_conf=roi_min_conf,
                roi_dilate_px=roi_dilate_px,
            )
            try:
                samples["full_psnr"].append(psnr_full_bgr(frm_a, frm_b))
                samples["full_ms_ssim"].append(ms_ssim_full_bgr(frm_a, frm_b))
            except Exception:
                pass
            r_psnr = psnr_roi_bgr(frm_a, frm_b, mask)
            r_mssim = ms_ssim_roi_bgr(frm_a, frm_b, mask)
            if np.isfinite(r_psnr):
                samples["roi_psnr"].append(float(r_psnr))
            if np.isfinite(r_mssim):
                samples["roi_ms_ssim"].append(float(r_mssim))

            metric_frames += 1
            if max_metric_frames > 0 and metric_frames >= max_metric_frames:
                break
    finally:
        cap_a.release()
        cap_b.release()

    agg = aggregate_metrics(samples)
    agg["metrics_frames_used"] = int(metric_frames)
    agg["metrics_stride"] = int(st)
    return agg


def _apply_codec_pair(base_cfg: Dict[str, Any], roi: Dict[str, Any], bg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    codec_cfg = cfg.setdefault("compression", {}).setdefault("codec", {})
    roi_cfg = codec_cfg.setdefault("roi", {})
    bg_cfg = codec_cfg.setdefault("bg", {})
    for dst, src in ((roi_cfg, roi), (bg_cfg, bg)):
        dst["codec"] = src["codec"]
        dst["encoder"] = src["encoder"]
        dst["preset"] = src["preset"]
        dst["qp"] = int(src["qp"])
    return cfg


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_cfg_path = Path(args.compression_config).expanduser().resolve()
    dec_cfg_path = Path(args.decompression_config).expanduser().resolve()
    base_cfg = _read_yaml(comp_cfg_path)
    ffmpeg_bin = str((base_cfg.get("compression", {}).get("codec", {}) or {}).get("ffmpeg_bin", "ffmpeg"))
    available_encoders = _ffmpeg_encoders(ffmpeg_bin)
    if available_encoders:
        roi_candidates = [c for c in ROI_CANDIDATES if c["encoder"] in available_encoders]
        bg_candidates = [c for c in BG_CANDIDATES if c["encoder"] in available_encoders]
    else:
        roi_candidates = ROI_CANDIDATES
        bg_candidates = BG_CANDIDATES

    if not roi_candidates or not bg_candidates:
        raise RuntimeError("No benchmark codec candidates available in local ffmpeg build.")

    video_dir = Path(args.video_dir).expanduser().resolve()
    videos = _videos_for_run(video_dir, args.video_names, int(args.max_videos))
    if not videos:
        raise FileNotFoundError("No videos found for benchmark")

    if not args.no_metrics:
        try:
            require_skimage()
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

    if not args.dry_run and not args.skip_cuda_check and not _cuda_available():
        print(
            "[ERROR] CUDA is not visible to PyTorch on this machine. "
            "`run_compression.py` uses a strict GPU ROI path.\n"
            "Run codec benchmark on DGX/CUDA hardware, or pass --skip-cuda-check anyway for debugging.",
            file=sys.stderr,
        )
        sys.exit(2)

    pair_list = list(itertools.product(roi_candidates, bg_candidates))
    if int(args.max_pairs) > 0:
        pair_list = pair_list[: int(args.max_pairs)]

    results_csv = out_dir / "codec_pair_results.csv"
    resume = not args.no_resume
    completed_keys: set[tuple[str, str, str]] = set()
    append_mode = False
    if resume and results_csv.is_file() and results_csv.stat().st_size > 0:
        try:
            completed_keys = _load_completed_benchmark_keys(results_csv, RESULTS_CSV_FIELDS)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        append_mode = True
        print(f"[RESUME] {len(completed_keys)} rows already in {results_csv.name}; will skip those keys")
    elif args.no_resume and results_csv.is_file():
        print(f"[FRESH] Overwriting {results_csv.name} (--no-resume)")

    open_mode = "a" if append_mode else "w"
    with results_csv.open(open_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS)
        if not append_mode:
            writer.writeheader()

        for roi, bg in pair_list:
            pair_name = f"{roi['name']}__{bg['name']}"
            pair_dir = out_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            pair_cfg = _apply_codec_pair(base_cfg, roi, bg)
            pair_cfg_path = pair_dir / "compression.runtime.yaml"
            _write_yaml(pair_cfg_path, pair_cfg)
            py = _project_venv_python()

            for video in videos:
                key = (video.name, roi["name"], bg["name"])
                if key in completed_keys:
                    print(f"[SKIP] resume | {video.name} | {roi['name']} + {bg['name']}")
                    continue
                archive = pair_dir / f"{video.stem}.zip"
                recon = pair_dir / f"{video.stem}_recon.mp4"
                comp_cmd = [
                    py,
                    str(ROOT / "run_compression.py"),
                    str(video),
                    "--config",
                    str(pair_cfg_path),
                    "--output",
                    str(archive),
                ]
                dec_cmd = [
                    py,
                    str(ROOT / "run_decompression.py"),
                    str(archive),
                    "--config",
                    str(dec_cfg_path),
                    "--output",
                    str(recon),
                    "--no-interpolate",
                ]
                if args.dry_run:
                    print("[DRY-RUN]", " ".join(comp_cmd))
                    print("[DRY-RUN]", " ".join(dec_cmd))
                    continue

                encode_time = _run(comp_cmd)
                decode_time = _run(dec_cmd)
                size_bytes = int(archive.stat().st_size) if archive.exists() else 0
                cap = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-count_frames",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_read_frames",
                        "-of",
                        "default=nokey=1:noprint_wrappers=1",
                        str(video),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                frames = int((cap.stdout or "0").strip() or 0) if cap.returncode == 0 else 0
                fps_encode = round(frames / encode_time, 3) if frames > 0 and encode_time > 0 else ""

                row: Dict[str, Any] = {
                    "video": video.name,
                    "roi_profile": roi["name"],
                    "bg_profile": bg["name"],
                    "archive_size_bytes": size_bytes,
                    "encode_time_sec": encode_time,
                    "decode_time_sec": decode_time,
                    "throughput_fps_encode": fps_encode,
                    "roi_psnr_mean": "",
                    "roi_ms_ssim_mean": "",
                    "full_psnr_mean": "",
                    "full_ms_ssim_mean": "",
                    "metrics_frames_used": "",
                    "metrics_stride": "",
                }

                if not args.no_metrics and archive.exists() and recon.exists():
                    metrics = _evaluate_pair_metrics(
                        original=video,
                        recon=recon,
                        archive=archive,
                        stride=int(args.metrics_stride),
                        max_metric_frames=int(args.max_metric_frames),
                    )

                    row["roi_psnr_mean"] = _csv_metric(metrics, "roi_psnr_mean")
                    row["roi_ms_ssim_mean"] = _csv_metric(metrics, "roi_ms_ssim_mean")
                    row["full_psnr_mean"] = _csv_metric(metrics, "full_psnr_mean")
                    row["full_ms_ssim_mean"] = _csv_metric(metrics, "full_ms_ssim_mean")
                    row["metrics_frames_used"] = str(metrics.get("metrics_frames_used", ""))
                    row["metrics_stride"] = str(metrics.get("metrics_stride", ""))

                writer.writerow(row)
                completed_keys.add(key)
                extra = ""
                if not args.no_metrics and row["full_psnr_mean"]:
                    extra = (
                        f" full_psnr={row['full_psnr_mean']} roi_psnr={row['roi_psnr_mean']} "
                        f"(frames={row['metrics_frames_used']}, stride={row['metrics_stride']})"
                    )
                print(
                    f"[OK] {video.name} | {roi['name']} + {bg['name']} | "
                    f"size={size_bytes}B encode={encode_time}s decode={decode_time}s{extra}"
                )

    print(f"[DONE] benchmark CSV: {results_csv} (mode={'append' if append_mode else 'write+fresh header'})")


if __name__ == "__main__":
    main()
