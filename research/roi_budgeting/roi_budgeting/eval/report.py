from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REPORT_COLUMNS = [
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


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


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


def _baseline_row_from_manifest(payload: Mapping[str, Any], *, manifest_name: str) -> Dict[str, Any]:
    frame_drop = payload.get("frame_drop", {}) or {}
    keep_summary = frame_drop.get("keep_summary", {}) or {}
    experiment = payload.get("experiment", {}) or {}
    baseline_reference = experiment.get("baseline_reference", {}) or {}
    return {
        "experiment_name": "fixed_baseline",
        "description": "Current fixed ROI heuristic from the production pipeline.",
        "roi_kept_frames": int(frame_drop.get("roi_kept_frames", 0) or 0),
        "keep_ratio": _as_float(keep_summary.get("keep_ratio")),
        "estimated_roi_kbps": _as_float(baseline_reference.get("estimated_roi_kbps")),
        "estimated_roi_bytes": _as_float(baseline_reference.get("estimated_roi_bytes")),
        "budget_utilization": _as_float(baseline_reference.get("budget_utilization")),
        "jaccard_vs_baseline": 1.0,
        "primary_delta": 0.0,
        "motion_delta": 0.0,
        "uncertainty_delta": 0.0,
        "amt_delta": 0.0,
        "combined_delta": 0.0,
        "mean_motion_score_kept": _as_float(baseline_reference.get("mean_motion_score_kept")),
        "mean_uncertainty_score_kept": _as_float(baseline_reference.get("mean_uncertainty_score_kept")),
        "mean_amt_risk_kept": _as_float(baseline_reference.get("mean_amt_risk_kept")),
        "mean_combined_score_kept": _as_float(baseline_reference.get("mean_combined_score_kept")),
        "amt_source": "fixed_baseline",
        "manifest_name": manifest_name,
    }


def _experiment_row_from_manifest(payload: Mapping[str, Any], *, manifest_name: str) -> Dict[str, Any]:
    experiment = payload.get("experiment", {}) or {}
    proposal = experiment.get("proposal", {}) or {}
    keep_summary = proposal.get("keep_summary", {}) or {}
    comparison = experiment.get("comparison", {}) or {}
    overlap = comparison.get("overlap", {}) or {}
    amt_probe = experiment.get("amt_probe", {}) or {}
    combined_delta = _as_float(comparison.get("mean_combined_score_delta"))
    motion_delta = _as_float(comparison.get("mean_motion_score_delta"))
    return {
        "experiment_name": str(experiment.get("name", "") or manifest_name.replace("_summary.json", "")),
        "description": str(experiment.get("description", "") or ""),
        "roi_kept_frames": int(proposal.get("roi_kept_frames", 0) or 0),
        "keep_ratio": _as_float(keep_summary.get("keep_ratio")),
        "estimated_roi_kbps": _as_float(proposal.get("estimated_roi_kbps")),
        "estimated_roi_bytes": _as_float(proposal.get("estimated_roi_bytes")),
        "budget_utilization": _as_float(proposal.get("budget_utilization")),
        "jaccard_vs_baseline": _as_float(overlap.get("jaccard")),
        "primary_delta": combined_delta if combined_delta is not None else motion_delta,
        "motion_delta": motion_delta,
        "uncertainty_delta": _as_float(comparison.get("mean_uncertainty_score_delta")),
        "amt_delta": _as_float(comparison.get("mean_amt_risk_delta")),
        "combined_delta": combined_delta,
        "mean_motion_score_kept": _as_float(proposal.get("mean_motion_score_kept")),
        "mean_uncertainty_score_kept": _as_float(proposal.get("mean_uncertainty_score_kept")),
        "mean_amt_risk_kept": _as_float(proposal.get("mean_amt_risk_kept")),
        "mean_combined_score_kept": _as_float(proposal.get("mean_combined_score_kept")),
        "amt_source": str(amt_probe.get("source", "") or ""),
        "manifest_name": manifest_name,
    }


def build_report_rows(*, manifests_dir: str | Path) -> List[Dict[str, Any]]:
    manifest_root = Path(manifests_dir).expanduser().resolve()
    baseline_manifest = manifest_root / "offline_eval_summary.json"

    if not baseline_manifest.exists():
        raise FileNotFoundError(f"Baseline manifest is missing: {baseline_manifest}")

    rows: List[Dict[str, Any]] = []
    baseline_payload = load_json(baseline_manifest)
    rows.append(_baseline_row_from_manifest(baseline_payload, manifest_name=baseline_manifest.name))

    manifest_paths = sorted(
        path
        for path in manifest_root.glob("*_summary.json")
        if path.name != baseline_manifest.name
    )
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        rows.append(_experiment_row_from_manifest(load_json(manifest_path), manifest_name=manifest_path.name))

    return rows


def write_csv_report(*, rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> str:
    p = Path(output_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REPORT_COLUMNS})
    return str(p)


def write_markdown_report(*, rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> str:
    rows_list = list(rows)
    p = Path(output_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    headers = REPORT_COLUMNS
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows_list:
        values = [_display_value(row.get(key)) for key in headers]
        lines.append("| " + " | ".join(values) + " |")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)
