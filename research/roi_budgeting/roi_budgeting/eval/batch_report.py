from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from roi_budgeting.eval.report import REPORT_COLUMNS, build_report_rows


BATCH_DETAIL_COLUMNS = [
    "clip_name",
    "experiment_name",
    "description",
    "roi_kept_frames",
    "keep_ratio",
    "estimated_roi_kbps",
    "estimated_roi_bytes",
    "budget_utilization",
    "jaccard_vs_baseline",
    "primary_delta",
    "motion_delta",
    "uncertainty_delta",
    "amt_delta",
    "combined_delta",
    "mean_motion_score_kept",
    "mean_uncertainty_score_kept",
    "mean_amt_risk_kept",
    "mean_combined_score_kept",
    "amt_source",
    "manifest_name",
]


BATCH_AGGREGATE_COLUMNS = [
    "experiment_name",
    "clips_evaluated",
    "mean_primary_delta",
    "median_primary_delta",
    "mean_estimated_roi_kbps",
    "mean_budget_utilization",
    "mean_jaccard_vs_baseline",
    "mean_keep_ratio",
    "mean_roi_kept_frames",
]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def collect_batch_rows(*, benchmark_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(benchmark_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}")

    rows: List[Dict[str, Any]] = []
    for clip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifests_dir = clip_dir / "manifests"
        baseline_manifest = manifests_dir / "offline_eval_summary.json"
        if not baseline_manifest.exists():
            continue
        clip_rows = build_report_rows(manifests_dir=manifests_dir)
        for row in clip_rows:
            rows.append(
                {
                    "clip_name": str(clip_dir.name),
                    **{key: row.get(key, "") for key in REPORT_COLUMNS},
                }
            )
    return rows


def aggregate_batch_rows(*, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_experiment: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        name = str(row.get("experiment_name", "") or "")
        if not name:
            continue
        by_experiment.setdefault(name, []).append(row)

    order = ["fixed_baseline", "v1_motion_only", "v2_motion_uncertainty", "v3_motion_uncertainty_amt", "v4_segment_dp_amt"]
    out: List[Dict[str, Any]] = []
    for experiment_name in sorted(by_experiment.keys(), key=lambda value: (order.index(value) if value in order else len(order), value)):
        exp_rows = by_experiment[experiment_name]

        def _mean(key: str) -> float | None:
            vals = [_as_float(row.get(key)) for row in exp_rows]
            filtered = [v for v in vals if v is not None]
            return float(sum(filtered) / len(filtered)) if filtered else None

        def _median(key: str) -> float | None:
            vals = [_as_float(row.get(key)) for row in exp_rows]
            filtered = [v for v in vals if v is not None]
            return float(statistics.median(filtered)) if filtered else None

        out.append(
            {
                "experiment_name": experiment_name,
                "clips_evaluated": int(len(exp_rows)),
                "mean_primary_delta": _mean("primary_delta"),
                "median_primary_delta": _median("primary_delta"),
                "mean_estimated_roi_kbps": _mean("estimated_roi_kbps"),
                "mean_budget_utilization": _mean("budget_utilization"),
                "mean_jaccard_vs_baseline": _mean("jaccard_vs_baseline"),
                "mean_keep_ratio": _mean("keep_ratio"),
                "mean_roi_kept_frames": _mean("roi_kept_frames"),
            }
        )
    return out


def write_csv_report(*, rows: Iterable[Mapping[str, Any]], columns: Sequence[str], output_path: str | Path) -> str:
    p = Path(output_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return str(p)


def write_markdown_report(*, rows: Iterable[Mapping[str, Any]], columns: Sequence[str], output_path: str | Path) -> str:
    rows_list = list(rows)
    p = Path(output_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows_list:
        values = [_display_value(row.get(key)) for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)
