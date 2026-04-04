from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "08_github_summary.png"
PDF_PATH = OUT_DIR / "08_github_summary.pdf"

COLORS = {
    "base": "#7A8799",
    "accent": "#C94F3D",
    "accent_light": "#E7B1A8",
    "ink": "#1F2430",
    "panel": "#F7F4EF",
}

METRICS = {
    "size": {
        "original_mb": 399.16,
        "transmitted_mb": 10.36,
        "reduction_pct": 97.4042,
        "compression_ratio_x": 38.5239,
    },
    "quality": {
        "full_psnr_db": 35.2961,
        "roi_psnr_db": 36.1226,
        "full_ms_ssim": 0.9625,
        "roi_ms_ssim": 0.9758,
    },
    "roi_stage": {
        "dense_detector_calls_norm": 100.0,
        "sparse_detector_calls_norm": 6.72,
        "speedup_x": 4.41,
        "roi_recall_pct": 96.24,
        "frame_agreement_pct": 95.25,
    },
    "runtime": {
        "compression_fps": 20.88,
        "reconstruction_fps": 6.35,
        "compression_realtime_x": 0.70,
        "reconstruction_realtime_x": 0.21,
    },
}


def style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color=COLORS["ink"])
    ax.set_facecolor(COLORS["panel"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B1A7")
    ax.spines["bottom"].set_color("#B8B1A7")
    ax.tick_params(colors=COLORS["ink"], labelsize=9)
    ax.grid(axis="y", color="#DED7CC", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, fmt: str, y_pad: float) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + y_pad,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["ink"],
        )


def make_size_panel(ax) -> None:
    style_axis(ax, "Transmitted Size Impact")
    vals = [
        METRICS["size"]["original_mb"],
        METRICS["size"]["transmitted_mb"],
    ]
    labels = ["Original set", "Transmitted ZIPs"]
    bars = ax.bar(
        labels,
        vals,
        color=[COLORS["base"], COLORS["accent"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    annotate_bars(ax, bars, "{:.2f} MB", 7.0)
    ax.set_ylabel("MB", fontsize=10, color=COLORS["ink"])
    ax.set_ylim(0, 440)
    ax.text(
        0.5,
        232,
        f"-{METRICS['size']['reduction_pct']:.1f}%\n{METRICS['size']['compression_ratio_x']:.2f}x smaller",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=COLORS["accent"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLORS["accent"]),
    )


def make_quality_panel(fig, spec) -> None:
    sub = spec.subgridspec(1, 2, wspace=0.32)
    ax_psnr = fig.add_subplot(sub[0, 0])
    ax_ms = fig.add_subplot(sub[0, 1])

    style_axis(ax_psnr, "ROI vs Full-Frame PSNR")
    psnr_vals = [
        METRICS["quality"]["full_psnr_db"],
        METRICS["quality"]["roi_psnr_db"],
    ]
    bars = ax_psnr.bar(
        ["Full frame", "ROI"],
        psnr_vals,
        color=[COLORS["base"], COLORS["accent"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    annotate_bars(ax_psnr, bars, "{:.2f} dB", 0.06)
    ax_psnr.set_ylabel("Higher is better", fontsize=10, color=COLORS["ink"])
    ax_psnr.set_ylim(34.8, 36.5)
    ax_psnr.text(
        0.5,
        35.03,
        f"+{METRICS['quality']['roi_psnr_db'] - METRICS['quality']['full_psnr_db']:.2f} dB",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["accent"],
    )

    style_axis(ax_ms, "ROI vs Full-Frame MS-SSIM")
    ms_vals = [
        METRICS["quality"]["full_ms_ssim"],
        METRICS["quality"]["roi_ms_ssim"],
    ]
    bars = ax_ms.bar(
        ["Full frame", "ROI"],
        ms_vals,
        color=[COLORS["base"], COLORS["accent"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    annotate_bars(ax_ms, bars, "{:.4f}", 0.00045)
    ax_ms.set_ylabel("Higher is better", fontsize=10, color=COLORS["ink"])
    ax_ms.set_ylim(0.958, 0.9795)
    ax_ms.text(
        0.5,
        0.9588,
        f"+{METRICS['quality']['roi_ms_ssim'] - METRICS['quality']['full_ms_ssim']:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["accent"],
    )


def make_roi_panel(fig, spec) -> None:
    sub = spec.subgridspec(1, 2, wspace=0.32)
    ax_calls = fig.add_subplot(sub[0, 0])
    ax_cont = fig.add_subplot(sub[0, 1])

    style_axis(ax_calls, "Sparse ROI Stage Efficiency")
    call_vals = [
        METRICS["roi_stage"]["dense_detector_calls_norm"],
        METRICS["roi_stage"]["sparse_detector_calls_norm"],
    ]
    bars = ax_calls.bar(
        ["Dense reference", "Sparse +\npropagation"],
        call_vals,
        color=[COLORS["base"], COLORS["accent"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    annotate_bars(ax_calls, bars, "{:.2f}", 2.4)
    ax_calls.set_ylabel("Normalized detector calls", fontsize=10, color=COLORS["ink"])
    ax_calls.set_ylim(0, 110)
    ax_calls.text(
        0.5,
        84,
        f"{METRICS['roi_stage']['speedup_x']:.2f}x faster\n93.28% fewer detector calls",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["accent"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["accent"]),
    )

    style_axis(ax_cont, "ROI Timeline Continuity")
    continuity_vals = [
        METRICS["roi_stage"]["roi_recall_pct"],
        METRICS["roi_stage"]["frame_agreement_pct"],
    ]
    bars = ax_cont.bar(
        ["ROI recall", "Frame agreement"],
        continuity_vals,
        color=[COLORS["accent"], COLORS["accent_light"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    annotate_bars(ax_cont, bars, "{:.2f}%", 1.0)
    ax_cont.set_ylabel("Percent", fontsize=10, color=COLORS["ink"])
    ax_cont.set_ylim(0, 104)


def make_runtime_panel(fig, spec) -> None:
    sub = spec.subgridspec(1, 2, wspace=0.32)
    ax_fps = fig.add_subplot(sub[0, 0])
    ax_rt = fig.add_subplot(sub[0, 1])

    style_axis(ax_fps, "Runtime Throughput")
    fps_vals = [
        METRICS["runtime"]["compression_fps"],
        METRICS["runtime"]["reconstruction_fps"],
    ]
    bars = ax_fps.bar(
        ["Compression", "Reconstruction"],
        fps_vals,
        color=[COLORS["base"], COLORS["accent"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    annotate_bars(ax_fps, bars, "{:.2f} fps", 0.45)
    ax_fps.set_ylabel("Frames per second", fontsize=10, color=COLORS["ink"])
    ax_fps.set_ylim(0, 23.5)

    style_axis(ax_rt, "Realtime Factor")
    rt_vals = [
        METRICS["runtime"]["compression_realtime_x"],
        METRICS["runtime"]["reconstruction_realtime_x"],
    ]
    bars = ax_rt.bar(
        ["Compression", "Reconstruction"],
        rt_vals,
        color=[COLORS["base"], COLORS["accent"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    annotate_bars(ax_rt, bars, "{:.2f}x", 0.025)
    ax_rt.set_ylabel("Relative to source fps", fontsize=10, color=COLORS["ink"])
    ax_rt.set_ylim(0, 0.8)
    ax_rt.text(
        0.5,
        0.62,
        "Reconstruction is the\nruntime bottleneck",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["accent"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["accent"]),
    )


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 9.3), facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

    ax_size = fig.add_subplot(grid[0, 0])
    make_size_panel(ax_size)
    make_quality_panel(fig, grid[0, 1])
    make_roi_panel(fig, grid[1, 0])
    make_runtime_panel(fig, grid[1, 1])

    fig.suptitle(
        "Summary Figure: Neural ROI-Aware Video Compression for Wildlife Monitoring",
        fontsize=18,
        fontweight="bold",
        y=0.98,
        color=COLORS["ink"],
    )
    fig.text(
        0.5,
        0.945,
        "Held-out test-set metrics from the thesis. LPIPS plots are not included in this figure.",
        ha="center",
        va="center",
        fontsize=11,
        color="#4D5563",
    )

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
