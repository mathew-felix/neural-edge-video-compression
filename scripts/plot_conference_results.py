from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = ROOT / "outputs" / "paper_runs" / "results.csv"
DEFAULT_FIG_DIR = ROOT / "outputs" / "paper_runs" / "figures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from results.csv")
    parser.add_argument("--results-csv", type=str, default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--fig-dir", type=str, default=str(DEFAULT_FIG_DIR))
    return parser.parse_args()


def _to_float(value: str) -> float | None:
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _plot_xy(rows: List[Dict[str, str]], x_key: str, y_key: str, out_path: Path, title: str) -> None:
    grouped = defaultdict(list)
    for row in rows:
        x = _to_float(row.get(x_key, ""))
        y = _to_float(row.get(y_key, ""))
        if x is None or y is None:
            continue
        method = str(row.get("method", "unknown"))
        grouped[method].append((x, y))

    if not grouped:
        return

    plt.figure(figsize=(7, 4.5))
    for method, pts in grouped.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        plt.plot(xs, ys, marker="o", label=method)

    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = _parse_args()
    results_csv = Path(args.results_csv).expanduser().resolve()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    rows = _load_rows(results_csv)

    _plot_xy(
        rows,
        x_key="bitrate_bps",
        y_key="roi_psnr",
        out_path=fig_dir / "bitrate_vs_roi_psnr.png",
        title="Bitrate vs ROI PSNR",
    )
    _plot_xy(
        rows,
        x_key="bitrate_bps",
        y_key="full_psnr",
        out_path=fig_dir / "bitrate_vs_full_psnr.png",
        title="Bitrate vs Full-Frame PSNR",
    )
    _plot_xy(
        rows,
        x_key="encode_time_sec_dgx",
        y_key="decode_time_sec_laptop",
        out_path=fig_dir / "encode_vs_decode_runtime.png",
        title="DGX Encode vs Laptop Decode Runtime",
    )
    print(f"[OK] figures written under: {fig_dir}")


if __name__ == "__main__":
    main()
