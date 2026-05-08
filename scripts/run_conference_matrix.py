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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "paper_runs"
DEFAULT_RESULTS_JSONL = DEFAULT_OUT_DIR / "results.jsonl"
DEFAULT_RESULTS_CSV = DEFAULT_OUT_DIR / "results.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DGX encode + laptop decode experiment matrix")
    parser.add_argument("--video", type=str, required=True, help="Input clip path")
    parser.add_argument("--compression-config", type=str, default="configs/gpu/compression.yaml")
    parser.add_argument("--decompression-config", type=str, default="configs/gpu/decompression.yaml")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--matrix", type=str, default="", help="Optional JSON file with experiment rows")
    parser.add_argument("--clip-id", type=str, default="clip_unknown")
    parser.add_argument("--host-role", type=str, choices=["dgx", "laptop"], required=True)
    parser.add_argument("--archive", type=str, default="", help="Required for laptop role")
    parser.add_argument("--run-label", type=str, default="manual")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_matrix(matrix_path: str) -> List[Dict[str, Any]]:
    if not matrix_path:
        return [
            {"method": "roi_aware_main", "variant": "default"},
            {"method": "uniform_ffmpeg", "variant": "default"},
            {"method": "roi_unaware_control", "variant": "default"},
        ]
    payload = json.loads(_resolve(matrix_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Matrix JSON must be a list of rows")
    return [row for row in payload if isinstance(row, dict)]


def _ensure_header(csv_path: Path, fieldnames: List[str]) -> None:
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_row(jsonl_path: Path, csv_path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    _ensure_header(csv_path, fieldnames)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _run_cmd(cmd: List[str]) -> float:
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
    return dt


def _compression_cmd(video: Path, config: Path, archive: Path) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "run_compression.py"),
        str(video),
        "--config",
        str(config),
        "--output",
        str(archive),
    ]


def _decompression_cmd(archive: Path, config: Path, output: Path) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "run_decompression.py"),
        str(archive),
        "--config",
        str(config),
        "--output",
        str(output),
        "--no-interpolate",
    ]


def main() -> None:
    args = _parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_matrix(args.matrix)
    comp_cfg = _resolve(args.compression_config)
    dec_cfg = _resolve(args.decompression_config)
    video = _resolve(args.video)
    archive_override = _resolve(args.archive) if args.archive else None

    fields = [
        "timestamp",
        "run_label",
        "host_role",
        "method",
        "variant",
        "clip_id",
        "video_path",
        "archive_path",
        "reconstructed_path",
        "archive_size_bytes",
        "archive_sha256",
        "compression_config_sha256",
        "decompression_config_sha256",
        "encode_time_sec_dgx",
        "decode_time_sec_laptop",
        "transfer_time_sec",
        "bitrate_bps",
        "roi_psnr",
        "roi_ms_ssim",
        "full_psnr",
        "full_ms_ssim",
        "notes",
    ]

    for row in rows:
        method = str(row.get("method", "roi_aware_main"))
        variant = str(row.get("variant", "default"))
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        archive = archive_override or (out_dir / f"{args.clip_id}_{method}_{variant}.zip")
        recon = out_dir / f"{args.clip_id}_{method}_{variant}_reconstructed.mp4"

        encode_time = ""
        decode_time = ""
        if args.host_role == "dgx":
            cmd = _compression_cmd(video, comp_cfg, archive)
            if args.dry_run:
                print("[DRY-RUN]", " ".join(cmd))
            else:
                encode_time = round(_run_cmd(cmd), 3)
        else:
            if archive_override is None:
                raise ValueError("--archive is required for host-role laptop")
            cmd = _decompression_cmd(archive, dec_cfg, recon)
            if args.dry_run:
                print("[DRY-RUN]", " ".join(cmd))
            else:
                decode_time = round(_run_cmd(cmd), 3)

        archive_size = int(archive.stat().st_size) if archive.exists() else 0
        bitrate_bps = ""
        if archive_size > 0:
            bitrate_bps = archive_size * 8

        result = {
            "timestamp": stamp,
            "run_label": args.run_label,
            "host_role": args.host_role,
            "method": method,
            "variant": variant,
            "clip_id": args.clip_id,
            "video_path": str(video),
            "archive_path": str(archive),
            "reconstructed_path": str(recon if recon.exists() else ""),
            "archive_size_bytes": archive_size,
            "archive_sha256": (_sha256_file(archive) if archive.exists() else ""),
            "compression_config_sha256": (_sha256_text(comp_cfg) if comp_cfg.exists() else ""),
            "decompression_config_sha256": (_sha256_text(dec_cfg) if dec_cfg.exists() else ""),
            "encode_time_sec_dgx": encode_time,
            "decode_time_sec_laptop": decode_time,
            "transfer_time_sec": "",
            "bitrate_bps": bitrate_bps,
            "roi_psnr": "",
            "roi_ms_ssim": "",
            "full_psnr": "",
            "full_ms_ssim": "",
            "notes": "Fill quality metrics in post-processing step.",
        }
        _append_row(DEFAULT_RESULTS_JSONL, DEFAULT_RESULTS_CSV, result, fields)
        print(f"[OK] logged row for {method}/{variant} ({args.host_role})")


if __name__ == "__main__":
    main()
