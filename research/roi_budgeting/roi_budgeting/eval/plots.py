from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence


def _ensure_matplotlib_cache_dir() -> None:
    current = os.environ.get("MPLCONFIGDIR", "").strip()
    if current:
        return
    cache_dir = (Path.cwd() / ".matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def save_series_plot(*, x: Sequence[float], y: Sequence[float], title: str, output_path: str | Path) -> str:
    """Small plotting helper with a local import so the package stays lightweight."""
    _ensure_matplotlib_cache_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Path(output_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(list(x), list(y))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def save_experiment_comparison_plot(
    *,
    rows: Sequence[Mapping[str, object]],
    title: str,
    output_path: str | Path,
) -> str:
    _ensure_matplotlib_cache_dir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Path(output_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row.get("experiment_name", "unknown")) for row in rows]
    primary = [float(row.get("primary_delta", 0.0) or 0.0) for row in rows]
    jaccard = [float(row.get("jaccard_vs_baseline", 0.0) or 0.0) for row in rows]

    colors = []
    for label in labels:
        if label == "fixed_baseline":
            colors.append("#6c757d")
        elif label == "v1_motion_only":
            colors.append("#1f77b4")
        elif label == "v2_motion_uncertainty":
            colors.append("#2ca02c")
        else:
            colors.append("#ff7f0e")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(title)

    ax0, ax1 = axes
    x = range(len(labels))

    bars0 = ax0.bar(x, primary, color=colors, alpha=0.9)
    ax0.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax0.set_xticks(list(x), labels, rotation=15, ha="right")
    ax0.set_ylabel("Delta")
    ax0.set_title("Primary Objective Delta Vs Baseline")
    ax0.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars0, primary):
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    bars1 = ax1.bar(x, jaccard, color=colors, alpha=0.9)
    ax1.set_xticks(list(x), labels, rotation=15, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Jaccard")
    ax1.set_title("Anchor Overlap With Fixed Baseline")
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars1, jaccard):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.03, bar.get_height() + 0.02),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)
