from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "paper_runs"
DEFAULT_COMP_CFG = ROOT / "configs" / "gpu" / "compression.yaml"
DEFAULT_RESULTS_JSONL = DEFAULT_OUT_DIR / "dgx_fixed_codec_results.jsonl"
DEFAULT_RESULTS_CSV = DEFAULT_OUT_DIR / "dgx_fixed_codec_results.csv"

FIELDS: List[str] = [
    "timestamp",
    "run_label",
    "clip_id",
    "video_path",
    "archive_path",
    "archive_size_bytes",
    "video_duration_sec",
    "bitrate_bps",
    "encode_time_sec_dgx",
    "compression_config_sha256",
    "archive_sha256",
    "roi_profile",
    "bg_profile",
    "status",
    "error",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DGX encode-only matrix with fixed ROI/BG codec profiles (resume enabled by default)."
    )
    parser.add_argument("--video-dir", type=str, default=str(ROOT / "data" / "test_videos"))
    parser.add_argument("--video-names", type=str, default="", help="Comma-separated basenames under --video-dir")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all videos in --video-dir")
    parser.add_argument("--compression-config", type=str, default=str(DEFAULT_COMP_CFG))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-label", type=str, default="dgx_fixed_codec")

    parser.add_argument("--roi-codec", type=str, default="av1")
    parser.add_argument("--roi-encoder", type=str, default="libsvtav1")
    parser.add_argument("--roi-preset", type=str, default="fast")
    parser.add_argument("--roi-qp", type=int, default=28)
    parser.add_argument("--roi-pix-fmt", type=str, default="yuv420p")

    parser.add_argument("--bg-codec", type=str, default="hevc")
    parser.add_argument("--bg-encoder", type=str, default="libx265")
    parser.add_argument("--bg-preset", type=str, default="fast")
    parser.add_argument("--bg-qp", type=int, default=36)
    parser.add_argument("--bg-pix-fmt", type=str, default="yuv420p")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite CSV/JSONL and re-run all clips")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


def _project_python() -> str:
    if sys.platform.startswith("win"):
        candidates = [ROOT / "venv" / "Scripts" / "python.exe", ROOT / ".venv" / "Scripts" / "python.exe"]
    else:
        candidates = [ROOT / "venv" / "bin" / "python", ROOT / ".venv" / "bin" / "python"]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return sys.executable


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return data


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _iter_videos(video_dir: Path, names: str, max_videos: int) -> List[Path]:
    wanted = str(names or "").strip()
    if wanted:
        out: List[Path] = []
        for raw in wanted.replace(",", " ").split():
            part = raw.strip()
            if not part:
                continue
            p = (video_dir / part).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Video not found: {p}")
            out.append(p)
        return out
    vids = sorted(video_dir.glob("*.mp4"))
    if max_videos > 0:
        vids = vids[:max_videos]
    return vids


def _video_duration_sec(video: Path, ffprobe_bin: str) -> float:
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return 0.0
    try:
        return max(0.0, float((proc.stdout or "").strip() or 0.0))
    except ValueError:
        return 0.0


def _run(cmd: List[str]) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False, capture_output=True, text=True, encoding="utf-8")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout[-2000:]
            + "\nstderr:\n"
            + proc.stderr[-2000:]
        )
    return round(dt, 3)


def _load_completed_keys(csv_path: Path) -> set[tuple[str, str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    out: set[tuple[str, str, str]] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            key = (
                str(row.get("video_path", "")).strip(),
                str(row.get("roi_profile", "")).strip(),
                str(row.get("bg_profile", "")).strip(),
            )
            if all(key):
                out.add(key)
    return out


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = _parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = out_dir / "archives_fixed_codec"
    archives_dir.mkdir(parents=True, exist_ok=True)

    comp_cfg = _resolve(args.compression_config)
    base_cfg = _read_yaml(comp_cfg)
    ffprobe_bin = str((base_cfg.get("compression", {}).get("codec", {}) or {}).get("ffprobe_bin", "ffprobe"))

    cfg = json.loads(json.dumps(base_cfg))
    codec_cfg = cfg.setdefault("compression", {}).setdefault("codec", {})
    roi_cfg = codec_cfg.setdefault("roi", {})
    bg_cfg = codec_cfg.setdefault("bg", {})
    roi_cfg.update(
        {
            "codec": args.roi_codec,
            "encoder": args.roi_encoder,
            "preset": args.roi_preset,
            "qp": int(args.roi_qp),
            "pix_fmt": args.roi_pix_fmt,
        }
    )
    bg_cfg.update(
        {
            "codec": args.bg_codec,
            "encoder": args.bg_encoder,
            "preset": args.bg_preset,
            "qp": int(args.bg_qp),
            "pix_fmt": args.bg_pix_fmt,
        }
    )

    roi_profile = f"{args.roi_codec}:{args.roi_encoder}:{args.roi_preset}:qp{int(args.roi_qp)}"
    bg_profile = f"{args.bg_codec}:{args.bg_encoder}:{args.bg_preset}:qp{int(args.bg_qp)}"
    cfg_tag = f"roi_{args.roi_encoder}_qp{int(args.roi_qp)}__bg_{args.bg_encoder}_qp{int(args.bg_qp)}"
    runtime_cfg = out_dir / f"compression.fixed.{cfg_tag}.yaml"
    _write_yaml(runtime_cfg, cfg)

    video_dir = _resolve(args.video_dir)
    videos = _iter_videos(video_dir, args.video_names, int(args.max_videos))
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found under {video_dir}")

    results_csv = out_dir / DEFAULT_RESULTS_CSV.name
    results_jsonl = out_dir / DEFAULT_RESULTS_JSONL.name

    if args.no_resume:
        mode = "w"
        completed = set()
    else:
        mode = "a" if results_csv.exists() and results_csv.stat().st_size > 0 else "w"
        completed = _load_completed_keys(results_csv)
        if completed:
            print(f"[RESUME] {len(completed)} completed rows detected; skipping those keys")

    with results_csv.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()

        py = _project_python()
        for video in videos:
            archive = archives_dir / f"{video.stem}.{cfg_tag}.zip"
            key = (str(video.resolve()), roi_profile, bg_profile)
            if key in completed:
                print(f"[SKIP] resume | {video.name}")
                continue

            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            clip_id = video.stem
            row: Dict[str, Any] = {
                "timestamp": stamp,
                "run_label": args.run_label,
                "clip_id": clip_id,
                "video_path": str(video.resolve()),
                "archive_path": str(archive.resolve()),
                "archive_size_bytes": "",
                "video_duration_sec": "",
                "bitrate_bps": "",
                "encode_time_sec_dgx": "",
                "compression_config_sha256": _sha256_text(runtime_cfg),
                "roi_profile": roi_profile,
                "bg_profile": bg_profile,
                "status": "ok",
                "error": "",
            }

            cmd = [py, str(ROOT / "run_compression.py"), str(video), "--config", str(runtime_cfg), "--output", str(archive)]
            if args.dry_run:
                print("[DRY-RUN]", " ".join(cmd))
                row["status"] = "dry_run"
            else:
                try:
                    encode_time = _run(cmd)
                    size_bytes = int(archive.stat().st_size) if archive.exists() else 0
                    duration = _video_duration_sec(video, ffprobe_bin=ffprobe_bin)
                    bitrate_bps = int(round((size_bytes * 8) / duration)) if size_bytes > 0 and duration > 0 else ""
                    row["archive_size_bytes"] = size_bytes
                    row["video_duration_sec"] = f"{duration:.6f}" if duration > 0 else ""
                    row["bitrate_bps"] = bitrate_bps
                    row["encode_time_sec_dgx"] = encode_time
                    row["archive_sha256"] = _sha256_file(archive) if archive.exists() else ""
                    print(
                        f"[OK] {video.name} | size={size_bytes}B duration={duration:.2f}s "
                        f"bitrate={bitrate_bps}bps encode={encode_time}s"
                    )
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)
                    print(f"[ERROR] {video.name} | {exc}")

            writer.writerow({k: row.get(k, "") for k in FIELDS})
            _append_jsonl(results_jsonl, row)
            if row["status"] == "ok":
                completed.add(key)

    print(f"[DONE] DGX fixed-codec encode run logged at: {results_csv}")


if __name__ == "__main__":
    main()

