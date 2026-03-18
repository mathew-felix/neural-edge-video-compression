from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict

import cv2

from roi_budgeting.data.load_frame_drop import (
    load_frame_drop,
    roi_keep_indices,
    roi_segments,
)
from roi_budgeting.data.load_roi_detections import load_roi_detections
from roi_budgeting.eval.bitrate import (
    build_roi_bitrate_features,
    estimate_roi_anchor_bytes,
    summarize_selected_roi_bytes,
)
from roi_budgeting.eval.metrics import index_overlap, summarize_keep_policy
from roi_budgeting.policy.budget_dp import (
    build_segment_scores_from_frame_scores,
    select_budgeted_roi_anchors,
)
from roi_budgeting.policy.fixed_baseline import (
    fixed_state_interval_keep,
    has_detection_from_frame_drop,
    state_from_frame_drop,
)
from roi_budgeting.policy.motion_only import (
    build_motion_scores,
    select_motion_only_roi_anchors,
    select_scored_roi_anchors,
    select_scored_roi_anchors_by_budget,
)
from roi_budgeting.signals.amt_risk import (
    build_dense_amt_risk_features,
    build_amt_risk_features,
    build_amt_risk_proxy_features,
    load_amt_probe_manifest,
)
from roi_budgeting.signals.motion import build_motion_features
from roi_budgeting.signals.uncertainty import build_uncertainty_features, build_uncertainty_scores


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the research configs. Install requirements.txt first.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI budgeting offline research runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local.yaml",
        help="Path to a research config file",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="Optional experiment overlay config",
    )
    return parser.parse_args()


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_path(raw_path: str | None, *, cfg_path: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p.resolve()

    candidates = [
        (Path.cwd() / p).resolve(),
        (cfg_path.parent / p).resolve(),
        (cfg_path.parent.parent / p).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _video_metadata(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    duration_sec = (float(frame_count) / fps) if fps > 0 else 0.0
    return {
        "path": str(video_path),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "duration_sec": float(duration_sec),
    }


def _roi_detection_summary(roi_payload: Dict[str, Any]) -> Dict[str, Any]:
    frames = roi_payload.get("frames", {}) or {}
    frame_ids = sorted(int(k) for k in frames.keys() if str(k).isdigit())
    total_boxes = 0
    frames_with_roi = 0
    for frame_idx in frame_ids:
        boxes = frames.get(str(frame_idx), [])
        if isinstance(boxes, list):
            total_boxes += len(boxes)
            if boxes:
                frames_with_roi += 1
    return {
        "frame_count_processed": int(roi_payload.get("frame_count", len(frame_ids)) or len(frame_ids)),
        "frames_with_roi": int(frames_with_roi),
        "total_boxes": int(total_boxes),
        "first_frames_with_roi": frame_ids[:15],
    }


def _segment_summary(frame_drop_json: Dict[str, Any]) -> Dict[str, Any]:
    segments = roi_segments(frame_drop_json)
    lengths = [(end - start + 1) for start, end in segments]
    if not lengths:
        return {
            "roi_segment_count": 0,
            "roi_segment_mean_len": 0.0,
            "roi_segment_median_len": 0.0,
            "roi_segment_max_len": 0,
        }
    return {
        "roi_segment_count": int(len(lengths)),
        "roi_segment_mean_len": float(sum(lengths) / len(lengths)),
        "roi_segment_median_len": float(median(lengths)),
        "roi_segment_max_len": int(max(lengths)),
    }


def _motion_feature_summary(motion_scores: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    score_vals = [
        float((rec.get("motion_score", 0.0) or 0.0))
        for rec in motion_scores.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]
    center_vals = [
        float((rec.get("center_speed", 0.0) or 0.0))
        for rec in motion_scores.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]
    area_vals = [
        float((rec.get("area_delta_abs", 0.0) or 0.0))
        for rec in motion_scores.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]

    def _safe_mean(vals: list[float]) -> float:
        return (float(sum(vals)) / float(len(vals))) if vals else 0.0

    def _safe_max(vals: list[float]) -> float:
        return float(max(vals)) if vals else 0.0

    return {
        "frames_with_motion_features": int(len(score_vals)),
        "motion_score_mean": _safe_mean(score_vals),
        "motion_score_median": float(median(score_vals)) if score_vals else 0.0,
        "motion_score_max": _safe_max(score_vals),
        "center_speed_mean": _safe_mean(center_vals),
        "center_speed_max": _safe_max(center_vals),
        "area_delta_mean": _safe_mean(area_vals),
        "area_delta_max": _safe_max(area_vals),
    }


def _uncertainty_feature_summary(uncertainty_scores: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    def _present(rec: Dict[str, float]) -> bool:
        try:
            return float(rec.get("missing_detection", 1.0)) < 1.0
        except (TypeError, ValueError):
            return False

    score_vals = [
        float((rec.get("uncertainty_score", 0.0) or 0.0))
        for rec in uncertainty_scores.values()
        if _present(rec)
    ]
    raw_vals = [
        float((rec.get("uncertainty_raw", 0.0) or 0.0))
        for rec in uncertainty_scores.values()
        if _present(rec)
    ]
    turnover_vals = [
        float((rec.get("track_turnover", 0.0) or 0.0))
        for rec in uncertainty_scores.values()
        if _present(rec)
    ]

    def _safe_mean(vals: list[float]) -> float:
        return (float(sum(vals)) / float(len(vals))) if vals else 0.0

    def _safe_max(vals: list[float]) -> float:
        return float(max(vals)) if vals else 0.0

    return {
        "frames_with_uncertainty_features": int(len(score_vals)),
        "uncertainty_score_mean": _safe_mean(score_vals),
        "uncertainty_score_median": float(median(score_vals)) if score_vals else 0.0,
        "uncertainty_score_max": _safe_max(score_vals),
        "uncertainty_raw_mean": _safe_mean(raw_vals),
        "uncertainty_raw_max": _safe_max(raw_vals),
        "track_turnover_mean": _safe_mean(turnover_vals),
        "track_turnover_max": _safe_max(turnover_vals),
    }


def _amt_feature_summary(amt_scores: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    risk_vals = [float((rec.get("amt_risk", 0.0) or 0.0)) for rec in amt_scores.values()]

    def _safe_mean(vals: list[float]) -> float:
        return (float(sum(vals)) / float(len(vals))) if vals else 0.0

    def _safe_max(vals: list[float]) -> float:
        return float(max(vals)) if vals else 0.0

    return {
        "frames_with_amt_features": int(len(risk_vals)),
        "amt_risk_mean": _safe_mean(risk_vals),
        "amt_risk_median": float(median(risk_vals)) if risk_vals else 0.0,
        "amt_risk_max": _safe_max(risk_vals),
    }


def _avg_selected(frames: list[int], lookup: Dict[int, Dict[str, float]], key: str) -> float:
    if not frames:
        return 0.0
    return float(sum((lookup.get(frame_idx, {}) or {}).get(key, 0.0) or 0.0 for frame_idx in frames)) / float(len(frames))


def _segment_guardrail_decision(
    *,
    cfg: Dict[str, Any],
    baseline_budget_utilization: float,
    proposed_budget_utilization: float,
    primary_delta: float,
) -> Dict[str, Any]:
    evaluation_cfg = cfg.get("evaluation", {}) or {}
    guardrail_cfg = (evaluation_cfg.get("guardrail", {}) or {})
    enabled = bool(guardrail_cfg.get("enabled", False))
    cheap_baseline_utilization_max = float(
        guardrail_cfg.get("cheap_baseline_utilization_max", 0.5) or 0.0
    )
    min_primary_delta = float(
        guardrail_cfg.get("min_primary_delta_when_baseline_cheap", 0.0) or 0.0
    )
    min_budget_increase = float(guardrail_cfg.get("min_budget_increase", 0.0) or 0.0)
    budget_increase = float(proposed_budget_utilization - baseline_budget_utilization)

    cheap_baseline = float(baseline_budget_utilization) <= float(cheap_baseline_utilization_max)
    large_divergence = float(budget_increase) >= float(min_budget_increase)
    weak_gain = float(primary_delta) < float(min_primary_delta)
    applied = bool(enabled and cheap_baseline and large_divergence and weak_gain)

    reason = None
    if applied:
        reason = (
            "Fallback to the fixed baseline because the baseline ROI budget was already cheap "
            "and the guarded v4 proposal did not clear the minimum primary-score gain."
        )

    return {
        "enabled": bool(enabled),
        "applied": bool(applied),
        "reason": reason,
        "selected_policy": ("fixed_baseline" if applied else "segment_budget_dp"),
        "cheap_baseline_utilization_max": float(cheap_baseline_utilization_max),
        "min_primary_delta_when_baseline_cheap": float(min_primary_delta),
        "min_budget_increase": float(min_budget_increase),
        "baseline_budget_utilization": float(baseline_budget_utilization),
        "proposed_budget_utilization": float(proposed_budget_utilization),
        "budget_utilization_increase": float(budget_increase),
        "primary_delta_before_guardrail": float(primary_delta),
        "cheap_baseline": bool(cheap_baseline),
        "large_divergence": bool(large_divergence),
        "weak_gain": bool(weak_gain),
    }


def _resolve_output_dir_from_cfg(cfg: Dict[str, Any]) -> Path | None:
    paths_cfg = cfg.get("paths", {}) or {}
    output_dir_raw = paths_cfg.get("output_dir", None)
    if not output_dir_raw:
        return None
    output_dir = Path(str(output_dir_raw)).expanduser()
    if output_dir.is_absolute():
        return output_dir.resolve()
    repo_root_raw = paths_cfg.get("repo_root", None)
    if repo_root_raw:
        repo_root = Path(str(repo_root_raw)).expanduser()
        if not repo_root.is_absolute():
            repo_root = (Path.cwd() / repo_root).resolve()
        return (repo_root / output_dir).resolve()
    return (Path.cwd() / output_dir).resolve()


def _load_reference_experiment(cfg: Dict[str, Any], manifest_name: str) -> Dict[str, Any] | None:
    output_dir = _resolve_output_dir_from_cfg(cfg)
    if output_dir is None:
        return None
    manifest_path = output_dir / "manifests" / str(manifest_name)
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = payload.get("experiment", {}) or {}
    if not isinstance(experiment, dict):
        return None
    return {
        "manifest_path": str(manifest_path),
        "experiment": experiment,
    }


def _bitrate_feature_summary(byte_estimates: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    roi_records = [rec for rec in byte_estimates.values() if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0]

    def _safe_mean(values: list[float]) -> float:
        return (float(sum(values)) / float(len(values))) if values else 0.0

    def _safe_max(values: list[float]) -> float:
        return float(max(values)) if values else 0.0

    area_vals = [float((rec.get("roi_area_ratio", 0.0) or 0.0)) for rec in roi_records]
    texture_vals = [float((rec.get("texture_raw", 0.0) or 0.0)) for rec in roi_records]
    byte_vals = [float((rec.get("estimated_bytes", 0.0) or 0.0)) for rec in roi_records]
    probe_byte_vals = [float((rec.get("probe_encoded_bytes", 0.0) or 0.0)) for rec in roi_records]
    return {
        "frames_with_bitrate_features": int(len(roi_records)),
        "roi_area_ratio_mean": _safe_mean(area_vals),
        "roi_area_ratio_max": _safe_max(area_vals),
        "texture_raw_mean": _safe_mean(texture_vals),
        "texture_raw_max": _safe_max(texture_vals),
        "probe_encoded_bytes_mean": _safe_mean(probe_byte_vals),
        "probe_encoded_bytes_max": _safe_max(probe_byte_vals),
        "estimated_anchor_bytes_mean": _safe_mean(byte_vals),
        "estimated_anchor_bytes_max": _safe_max(byte_vals),
    }


def _estimate_roi_budget(
    *,
    video_path: Path,
    roi_payload: Dict[str, Any],
    frame_drop_json: Dict[str, Any],
    video_meta: Dict[str, Any],
    motion_scores: Dict[int, Dict[str, float]],
    cfg: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Any]]:
    budget_cfg = cfg.get("budget", {}) or {}
    estimator_cfg = (budget_cfg.get("bitrate_estimator", {}) or {})
    bitrate_features = build_roi_bitrate_features(
        video_path=video_path,
        roi_payload=roi_payload,
        frame_drop_json=frame_drop_json,
        max_crop_side=int(estimator_cfg.get("max_crop_side", 160) or 160),
        probe_codec=str(estimator_cfg.get("probe_codec", "jpeg") or "jpeg"),
        probe_quality=int(estimator_cfg.get("probe_quality", 50) or 50),
        probe_max_side=int(estimator_cfg.get("probe_max_side", 96) or 96),
        probe_frame_mode=str(estimator_cfg.get("probe_frame_mode", "crop") or "crop"),
        fps=float(video_meta.get("fps", 30.0) or 30.0),
        frame_segments=roi_segments(frame_drop_json),
    )
    byte_estimates, byte_meta = estimate_roi_anchor_bytes(
        bitrate_features=bitrate_features,
        motion_scores=motion_scores,
        frame_drop_json=frame_drop_json,
        video_meta=video_meta,
        cfg=cfg,
    )
    budget_model = {
        "mode": str(budget_cfg.get("roi_budget_mode", "target_kbps") or "target_kbps"),
        "target_kbps": float(byte_meta.get("target_kbps", 0.0) or 0.0),
        "target_bytes": float(byte_meta.get("target_bytes", 0.0) or 0.0),
        "duration_sec": float(byte_meta.get("duration_sec", 0.0) or 0.0),
        "model": str(byte_meta.get("model", "roi_proxy_v1")),
        "calibration": str(byte_meta.get("calibration", "match_fixed_baseline_at_target_kbps")),
        "notes": (
            "Direct ROI probe bytes from the configured probe codec."
            if str(byte_meta.get("model", "")).startswith("roi_probe_")
            else "Calibrated ROI byte proxy. Per-anchor cost depends on ROI area, texture, "
            "box count, and motion, then scales to match the fixed baseline at the configured target_kbps."
        ),
        "estimator": byte_meta,
    }
    return byte_estimates, budget_model, _bitrate_feature_summary(byte_estimates)


def _motion_only_experiment(
    *,
    cfg: Dict[str, Any],
    frame_drop_json: Dict[str, Any],
    video_meta: Dict[str, Any],
    roi_payload: Dict[str, Any],
    video_path: Path,
) -> Dict[str, Any]:
    motion_features = build_motion_features(roi_payload)
    motion_scores = build_motion_scores(motion_features)
    byte_estimates, budget_model, bitrate_features = _estimate_roi_budget(
        video_path=video_path,
        roi_payload=roi_payload,
        frame_drop_json=frame_drop_json,
        video_meta=video_meta,
        motion_scores=motion_scores,
        cfg=cfg,
    )
    policy_cfg = cfg.get("policy", {}) or {}
    score_map = {
        int(frame_idx): float((rec.get("motion_score", 0.0) or 0.0))
        for frame_idx, rec in motion_scores.items()
    }
    cost_map = {
        int(frame_idx): float((rec.get("estimated_bytes", 0.0) or 0.0))
        for frame_idx, rec in byte_estimates.items()
    }
    proposed = select_scored_roi_anchors_by_budget(
        frame_drop_json=frame_drop_json,
        frame_scores=score_map,
        frame_costs=cost_map,
        target_bytes=float(budget_model.get("target_bytes", 0.0) or 0.0),
        force_keep_roi_birth=bool(policy_cfg.get("force_keep_roi_birth", True)),
        force_keep_roi_death=bool(policy_cfg.get("force_keep_roi_death", True)),
        force_keep_segment_bounds=bool(policy_cfg.get("force_keep_segment_bounds", True)),
        max_gap_frames=int(policy_cfg.get("max_gap_frames", 12) or 12),
    )
    expected = roi_keep_indices(frame_drop_json)
    overlap = index_overlap(expected, proposed)

    score_lookup = {
        int(frame_idx): float((rec.get("motion_score", 0.0) or 0.0))
        for frame_idx, rec in motion_scores.items()
    }
    proposed_score_mean = (
        float(sum(score_lookup.get(frame_idx, 0.0) for frame_idx in proposed)) / float(len(proposed))
        if proposed
        else 0.0
    )
    baseline_score_mean = (
        float(sum(score_lookup.get(frame_idx, 0.0) for frame_idx in expected)) / float(len(expected))
        if expected
        else 0.0
    )
    duration_sec = float(video_meta.get("duration_sec", 0.0) or 0.0)
    target_bytes = float(budget_model.get("target_bytes", 0.0) or 0.0)
    proposed_rate = summarize_selected_roi_bytes(
        frame_indices=proposed,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )
    baseline_rate = summarize_selected_roi_bytes(
        frame_indices=expected,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )

    return {
        "name": str(((cfg.get("experiment", {}) or {}).get("name", "v1_motion_only"))),
        "description": str(
            ((cfg.get("experiment", {}) or {}).get("description", "Motion-only ROI budgeting baseline."))
        ),
        "budget_model": budget_model,
        "motion_features": _motion_feature_summary(motion_scores),
        "bitrate_features": bitrate_features,
        "proposal": {
            "roi_kept_frames": int(len(proposed)),
            "keep_summary": summarize_keep_policy(
                total_frames=int(video_meta.get("frame_count", 0) or 0),
                kept_frames=proposed,
            ),
            "first_kept_frames": proposed[:20],
            "mean_motion_score_kept": float(proposed_score_mean),
            **proposed_rate,
        },
        "baseline_reference": {
            "roi_kept_frames": int(len(expected)),
            "first_kept_frames": expected[:20],
            "mean_motion_score_kept": float(baseline_score_mean),
            **baseline_rate,
        },
        "comparison": {
            "overlap": overlap,
            "count_delta_vs_baseline": int(len(proposed) - len(expected)),
            "mean_motion_score_delta": float(proposed_score_mean - baseline_score_mean),
            "estimated_roi_bytes_delta": float(
                proposed_rate["estimated_roi_bytes"] - baseline_rate["estimated_roi_bytes"]
            ),
            "estimated_roi_kbps_delta": float(
                proposed_rate["estimated_roi_kbps"] - baseline_rate["estimated_roi_kbps"]
            ),
            "exact_match": bool(proposed == expected),
        },
    }


def _motion_uncertainty_experiment(
    *,
    cfg: Dict[str, Any],
    frame_drop_json: Dict[str, Any],
    video_meta: Dict[str, Any],
    roi_payload: Dict[str, Any],
    video_path: Path,
) -> Dict[str, Any]:
    motion_features = build_motion_features(roi_payload)
    motion_scores = build_motion_scores(motion_features)
    uncertainty_features = build_uncertainty_features(roi_payload)
    uncertainty_scores = build_uncertainty_scores(uncertainty_features)
    byte_estimates, budget_model, bitrate_features = _estimate_roi_budget(
        video_path=video_path,
        roi_payload=roi_payload,
        frame_drop_json=frame_drop_json,
        video_meta=video_meta,
        motion_scores=motion_scores,
        cfg=cfg,
    )
    policy_cfg = cfg.get("policy", {}) or {}
    objective = ((cfg.get("objective", {}) or {}).get("weights", {}) or {})
    motion_w = float(objective.get("motion", 1.0) or 0.0)
    uncertainty_w = float(objective.get("uncertainty", 0.0) or 0.0)
    combined_scores: Dict[int, float] = {}
    frame_ids = sorted(
        set(int(v) for v in motion_scores.keys()) | set(int(v) for v in uncertainty_scores.keys())
    )
    for frame_idx in frame_ids:
        motion_score = float((motion_scores.get(frame_idx, {}) or {}).get("motion_score", 0.0) or 0.0)
        uncertainty_score = float(
            (uncertainty_scores.get(frame_idx, {}) or {}).get("uncertainty_score", 0.0) or 0.0
        )
        combined_scores[int(frame_idx)] = (motion_w * motion_score) + (uncertainty_w * uncertainty_score)

    cost_map = {
        int(frame_idx): float((rec.get("estimated_bytes", 0.0) or 0.0))
        for frame_idx, rec in byte_estimates.items()
    }
    proposed = select_scored_roi_anchors_by_budget(
        frame_drop_json=frame_drop_json,
        frame_scores=combined_scores,
        frame_costs=cost_map,
        target_bytes=float(budget_model.get("target_bytes", 0.0) or 0.0),
        force_keep_roi_birth=bool(policy_cfg.get("force_keep_roi_birth", True)),
        force_keep_roi_death=bool(policy_cfg.get("force_keep_roi_death", True)),
        force_keep_segment_bounds=bool(policy_cfg.get("force_keep_segment_bounds", True)),
        max_gap_frames=int(policy_cfg.get("max_gap_frames", 12) or 12),
    )
    expected = roi_keep_indices(frame_drop_json)
    overlap = index_overlap(expected, proposed)
    proposed_motion_mean = (
        float(sum((motion_scores.get(frame_idx, {}) or {}).get("motion_score", 0.0) or 0.0 for frame_idx in proposed))
        / float(len(proposed))
        if proposed
        else 0.0
    )
    proposed_unc_mean = (
        float(
            sum(
                (uncertainty_scores.get(frame_idx, {}) or {}).get("uncertainty_score", 0.0) or 0.0
                for frame_idx in proposed
            )
        )
        / float(len(proposed))
        if proposed
        else 0.0
    )
    baseline_motion_mean = (
        float(sum((motion_scores.get(frame_idx, {}) or {}).get("motion_score", 0.0) or 0.0 for frame_idx in expected))
        / float(len(expected))
        if expected
        else 0.0
    )
    baseline_unc_mean = (
        float(
            sum(
                (uncertainty_scores.get(frame_idx, {}) or {}).get("uncertainty_score", 0.0) or 0.0
                for frame_idx in expected
            )
        )
        / float(len(expected))
        if expected
        else 0.0
    )
    combined_mean = (float(sum(combined_scores.get(frame_idx, 0.0) for frame_idx in proposed)) / float(len(proposed))) if proposed else 0.0
    combined_baseline_mean = (
        float(sum(combined_scores.get(frame_idx, 0.0) for frame_idx in expected)) / float(len(expected))
        if expected
        else 0.0
    )
    duration_sec = float(video_meta.get("duration_sec", 0.0) or 0.0)
    target_bytes = float(budget_model.get("target_bytes", 0.0) or 0.0)
    proposed_rate = summarize_selected_roi_bytes(
        frame_indices=proposed,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )
    baseline_rate = summarize_selected_roi_bytes(
        frame_indices=expected,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )

    return {
        "name": str(((cfg.get("experiment", {}) or {}).get("name", "v2_motion_uncertainty"))),
        "description": str(
            ((cfg.get("experiment", {}) or {}).get("description", "Motion plus ROI uncertainty."))
        ),
        "budget_model": budget_model,
        "weights": {
            "motion": motion_w,
            "uncertainty": uncertainty_w,
        },
        "motion_features": _motion_feature_summary(motion_scores),
        "uncertainty_features": _uncertainty_feature_summary(uncertainty_scores),
        "bitrate_features": bitrate_features,
        "proposal": {
            "roi_kept_frames": int(len(proposed)),
            "keep_summary": summarize_keep_policy(
                total_frames=int(video_meta.get("frame_count", 0) or 0),
                kept_frames=proposed,
            ),
            "first_kept_frames": proposed[:20],
            "mean_motion_score_kept": float(proposed_motion_mean),
            "mean_uncertainty_score_kept": float(proposed_unc_mean),
            "mean_combined_score_kept": float(combined_mean),
            **proposed_rate,
        },
        "baseline_reference": {
            "roi_kept_frames": int(len(expected)),
            "first_kept_frames": expected[:20],
            "mean_motion_score_kept": float(baseline_motion_mean),
            "mean_uncertainty_score_kept": float(baseline_unc_mean),
            "mean_combined_score_kept": float(combined_baseline_mean),
            **baseline_rate,
        },
        "comparison": {
            "overlap": overlap,
            "count_delta_vs_baseline": int(len(proposed) - len(expected)),
            "mean_motion_score_delta": float(proposed_motion_mean - baseline_motion_mean),
            "mean_uncertainty_score_delta": float(proposed_unc_mean - baseline_unc_mean),
            "mean_combined_score_delta": float(combined_mean - combined_baseline_mean),
            "estimated_roi_bytes_delta": float(
                proposed_rate["estimated_roi_bytes"] - baseline_rate["estimated_roi_bytes"]
            ),
            "estimated_roi_kbps_delta": float(
                proposed_rate["estimated_roi_kbps"] - baseline_rate["estimated_roi_kbps"]
            ),
            "exact_match": bool(proposed == expected),
        },
    }


def _motion_uncertainty_amt_experiment(
    *,
    cfg: Dict[str, Any],
    frame_drop_json: Dict[str, Any],
    video_meta: Dict[str, Any],
    roi_payload: Dict[str, Any],
    video_path: Path,
    amt_manifest_path: Path | None,
) -> Dict[str, Any]:
    motion_features = build_motion_features(roi_payload)
    motion_scores = build_motion_scores(motion_features)
    uncertainty_features = build_uncertainty_features(roi_payload)
    uncertainty_scores = build_uncertainty_scores(uncertainty_features)
    amt_cfg = ((cfg.get("signals", {}) or {}).get("amt_risk", {}) or {})
    if amt_manifest_path is not None and amt_manifest_path.exists():
        amt_manifest = load_amt_probe_manifest(amt_manifest_path)
        amt_scores = build_amt_risk_features(amt_manifest)
        amt_meta = dict(amt_manifest.get("meta", {}) or {})
        amt_meta["manifest_path"] = str(amt_manifest_path)
        if not amt_scores:
            raise RuntimeError(f"AMT probe manifest is empty: {amt_manifest_path}")
    else:
        amt_scores, amt_meta = build_amt_risk_proxy_features(
            video_path=video_path,
            roi_payload=roi_payload,
            crop_margin_px=int(amt_cfg.get("crop_margin_px", 8) or 8),
            max_crop_side=int(amt_cfg.get("max_crop_side", 256) or 256),
        )

    byte_estimates, budget_model, bitrate_features = _estimate_roi_budget(
        video_path=video_path,
        roi_payload=roi_payload,
        frame_drop_json=frame_drop_json,
        video_meta=video_meta,
        motion_scores=motion_scores,
        cfg=cfg,
    )
    policy_cfg = cfg.get("policy", {}) or {}
    objective = ((cfg.get("objective", {}) or {}).get("weights", {}) or {})
    motion_w = float(objective.get("motion", 1.0) or 0.0)
    uncertainty_w = float(objective.get("uncertainty", 0.0) or 0.0)
    amt_w = float(objective.get("amt_risk", 0.0) or 0.0)

    combined_scores: Dict[int, float] = {}
    frame_ids = sorted(
        set(int(v) for v in motion_scores.keys())
        | set(int(v) for v in uncertainty_scores.keys())
        | set(int(v) for v in amt_scores.keys())
    )
    for frame_idx in frame_ids:
        motion_score = float((motion_scores.get(frame_idx, {}) or {}).get("motion_score", 0.0) or 0.0)
        uncertainty_score = float(
            (uncertainty_scores.get(frame_idx, {}) or {}).get("uncertainty_score", 0.0) or 0.0
        )
        amt_score = float((amt_scores.get(frame_idx, {}) or {}).get("amt_risk", 0.0) or 0.0)
        combined_scores[int(frame_idx)] = (
            (motion_w * motion_score) + (uncertainty_w * uncertainty_score) + (amt_w * amt_score)
        )

    cost_map = {
        int(frame_idx): float((rec.get("estimated_bytes", 0.0) or 0.0))
        for frame_idx, rec in byte_estimates.items()
    }
    proposed = select_scored_roi_anchors_by_budget(
        frame_drop_json=frame_drop_json,
        frame_scores=combined_scores,
        frame_costs=cost_map,
        target_bytes=float(budget_model.get("target_bytes", 0.0) or 0.0),
        force_keep_roi_birth=bool(policy_cfg.get("force_keep_roi_birth", True)),
        force_keep_roi_death=bool(policy_cfg.get("force_keep_roi_death", True)),
        force_keep_segment_bounds=bool(policy_cfg.get("force_keep_segment_bounds", True)),
        max_gap_frames=int(policy_cfg.get("max_gap_frames", 12) or 12),
    )
    expected = roi_keep_indices(frame_drop_json)
    overlap = index_overlap(expected, proposed)

    proposed_motion_mean = _avg_selected(proposed, motion_scores, "motion_score")
    proposed_unc_mean = _avg_selected(proposed, uncertainty_scores, "uncertainty_score")
    proposed_amt_mean = _avg_selected(proposed, amt_scores, "amt_risk")
    baseline_motion_mean = _avg_selected(expected, motion_scores, "motion_score")
    baseline_unc_mean = _avg_selected(expected, uncertainty_scores, "uncertainty_score")
    baseline_amt_mean = _avg_selected(expected, amt_scores, "amt_risk")
    combined_mean = (float(sum(combined_scores.get(frame_idx, 0.0) for frame_idx in proposed)) / float(len(proposed))) if proposed else 0.0
    combined_baseline_mean = (
        float(sum(combined_scores.get(frame_idx, 0.0) for frame_idx in expected)) / float(len(expected))
        if expected
        else 0.0
    )
    duration_sec = float(video_meta.get("duration_sec", 0.0) or 0.0)
    target_bytes = float(budget_model.get("target_bytes", 0.0) or 0.0)
    proposed_rate = summarize_selected_roi_bytes(
        frame_indices=proposed,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )
    baseline_rate = summarize_selected_roi_bytes(
        frame_indices=expected,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )

    return {
        "name": str(((cfg.get("experiment", {}) or {}).get("name", "v3_motion_uncertainty_amt"))),
        "description": str(
            ((cfg.get("experiment", {}) or {}).get("description", "Motion plus uncertainty plus AMT probe risk."))
        ),
        "budget_model": budget_model,
        "weights": {
            "motion": motion_w,
            "uncertainty": uncertainty_w,
            "amt_risk": amt_w,
        },
        "motion_features": _motion_feature_summary(motion_scores),
        "uncertainty_features": _uncertainty_feature_summary(uncertainty_scores),
        "amt_features": _amt_feature_summary(amt_scores),
        "bitrate_features": bitrate_features,
        "amt_probe": amt_meta,
        "proposal": {
            "roi_kept_frames": int(len(proposed)),
            "keep_summary": summarize_keep_policy(
                total_frames=int(video_meta.get("frame_count", 0) or 0),
                kept_frames=proposed,
            ),
            "first_kept_frames": proposed[:20],
            "mean_motion_score_kept": float(proposed_motion_mean),
            "mean_uncertainty_score_kept": float(proposed_unc_mean),
            "mean_amt_risk_kept": float(proposed_amt_mean),
            "mean_combined_score_kept": float(combined_mean),
            **proposed_rate,
        },
        "baseline_reference": {
            "roi_kept_frames": int(len(expected)),
            "first_kept_frames": expected[:20],
            "mean_motion_score_kept": float(baseline_motion_mean),
            "mean_uncertainty_score_kept": float(baseline_unc_mean),
            "mean_amt_risk_kept": float(baseline_amt_mean),
            "mean_combined_score_kept": float(combined_baseline_mean),
            **baseline_rate,
        },
        "comparison": {
            "overlap": overlap,
            "count_delta_vs_baseline": int(len(proposed) - len(expected)),
            "mean_motion_score_delta": float(proposed_motion_mean - baseline_motion_mean),
            "mean_uncertainty_score_delta": float(proposed_unc_mean - baseline_unc_mean),
            "mean_amt_risk_delta": float(proposed_amt_mean - baseline_amt_mean),
            "mean_combined_score_delta": float(combined_mean - combined_baseline_mean),
            "estimated_roi_bytes_delta": float(
                proposed_rate["estimated_roi_bytes"] - baseline_rate["estimated_roi_bytes"]
            ),
            "estimated_roi_kbps_delta": float(
                proposed_rate["estimated_roi_kbps"] - baseline_rate["estimated_roi_kbps"]
            ),
            "exact_match": bool(proposed == expected),
        },
    }


def _segment_budget_dp_experiment(
    *,
    cfg: Dict[str, Any],
    frame_drop_json: Dict[str, Any],
    video_meta: Dict[str, Any],
    roi_payload: Dict[str, Any],
    video_path: Path,
    amt_manifest_path: Path | None,
) -> Dict[str, Any]:
    motion_features = build_motion_features(roi_payload)
    motion_scores = build_motion_scores(motion_features)
    uncertainty_features = build_uncertainty_features(roi_payload)
    uncertainty_scores = build_uncertainty_scores(uncertainty_features)
    amt_cfg = ((cfg.get("signals", {}) or {}).get("amt_risk", {}) or {})
    if amt_manifest_path is not None and amt_manifest_path.exists():
        amt_manifest = load_amt_probe_manifest(amt_manifest_path)
        amt_scores = build_amt_risk_features(amt_manifest)
        dense_amt_scores = build_dense_amt_risk_features(amt_manifest)
        amt_meta = dict(amt_manifest.get("meta", {}) or {})
        amt_meta["manifest_path"] = str(amt_manifest_path)
        if not amt_scores:
            raise RuntimeError(f"AMT probe manifest is empty: {amt_manifest_path}")
    else:
        amt_scores, amt_meta = build_amt_risk_proxy_features(
            video_path=video_path,
            roi_payload=roi_payload,
            crop_margin_px=int(amt_cfg.get("crop_margin_px", 8) or 8),
            max_crop_side=int(amt_cfg.get("max_crop_side", 256) or 256),
        )
        dense_amt_scores = dict(amt_scores)

    byte_estimates, budget_model, bitrate_features = _estimate_roi_budget(
        video_path=video_path,
        roi_payload=roi_payload,
        frame_drop_json=frame_drop_json,
        video_meta=video_meta,
        motion_scores=motion_scores,
        cfg=cfg,
    )
    policy_cfg = cfg.get("policy", {}) or {}
    objective = ((cfg.get("objective", {}) or {}).get("weights", {}) or {})
    motion_w = float(objective.get("motion", 1.0) or 0.0)
    uncertainty_w = float(objective.get("uncertainty", 0.0) or 0.0)
    amt_w = float(objective.get("amt_risk", 0.0) or 0.0)

    candidate_frames: list[int] = []
    mandatory_frames: list[int] = []
    for start, end in roi_segments(frame_drop_json):
        seg_frames = list(range(int(start), int(end) + 1))
        candidate_frames.extend(seg_frames)
        if seg_frames:
            first = int(seg_frames[0])
            last = int(seg_frames[-1])
            if bool(policy_cfg.get("force_keep_segment_bounds", True)) or bool(policy_cfg.get("force_keep_roi_birth", True)):
                mandatory_frames.append(first)
            if (bool(policy_cfg.get("force_keep_segment_bounds", True)) or bool(policy_cfg.get("force_keep_roi_death", True))) and last != first:
                mandatory_frames.append(last)

    candidate_frames = sorted(set(int(v) for v in candidate_frames))
    mandatory_frames = sorted(set(int(v) for v in mandatory_frames))
    if not candidate_frames:
        raise RuntimeError("No ROI candidate frames were found for segment-DP scheduling.")

    dense_combined_scores: Dict[int, float] = {}
    for frame_idx in candidate_frames:
        motion_score = float((motion_scores.get(frame_idx, {}) or {}).get("motion_score", 0.0) or 0.0)
        uncertainty_score = float(
            (uncertainty_scores.get(frame_idx, {}) or {}).get("uncertainty_score", 0.0) or 0.0
        )
        amt_score = float((dense_amt_scores.get(frame_idx, {}) or {}).get("amt_risk", 0.0) or 0.0)
        dense_combined_scores[int(frame_idx)] = (
            (motion_w * motion_score) + (uncertainty_w * uncertainty_score) + (amt_w * amt_score)
        )

    cost_map = {
        int(frame_idx): float((rec.get("estimated_bytes", 0.0) or 0.0))
        for frame_idx, rec in byte_estimates.items()
    }
    segment_scores = build_segment_scores_from_frame_scores(
        candidate_frames=candidate_frames,
        mandatory_frames=mandatory_frames,
        frame_scores=dense_combined_scores,
        frame_costs=cost_map,
        max_gap_frames=int(policy_cfg.get("max_gap_frames", 12) or 12),
    )
    initial_anchor_bytes = float(cost_map.get(int(candidate_frames[0]), 0.0) or 0.0)
    selection = select_budgeted_roi_anchors(
        candidate_frames=candidate_frames,
        mandatory_frames=mandatory_frames,
        segment_scores=segment_scores,
        bit_budget=float(budget_model.get("target_bytes", 0.0) or 0.0),
        initial_bits=float(initial_anchor_bytes),
    )
    proposed = list(selection.kept_frames)
    expected = roi_keep_indices(frame_drop_json)
    overlap = index_overlap(expected, proposed)

    proposed_motion_mean = _avg_selected(proposed, motion_scores, "motion_score")
    proposed_unc_mean = _avg_selected(proposed, uncertainty_scores, "uncertainty_score")
    proposed_amt_mean = _avg_selected(proposed, dense_amt_scores, "amt_risk")
    baseline_motion_mean = _avg_selected(expected, motion_scores, "motion_score")
    baseline_unc_mean = _avg_selected(expected, uncertainty_scores, "uncertainty_score")
    baseline_amt_mean = _avg_selected(expected, dense_amt_scores, "amt_risk")
    combined_mean = (
        float(sum(dense_combined_scores.get(frame_idx, 0.0) for frame_idx in proposed)) / float(len(proposed))
        if proposed
        else 0.0
    )
    combined_baseline_mean = (
        float(sum(dense_combined_scores.get(frame_idx, 0.0) for frame_idx in expected)) / float(len(expected))
        if expected
        else 0.0
    )
    duration_sec = float(video_meta.get("duration_sec", 0.0) or 0.0)
    target_bytes = float(budget_model.get("target_bytes", 0.0) or 0.0)
    proposed_rate = summarize_selected_roi_bytes(
        frame_indices=proposed,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )
    baseline_rate = summarize_selected_roi_bytes(
        frame_indices=expected,
        byte_estimates=byte_estimates,
        duration_sec=duration_sec,
        target_bytes=target_bytes,
    )
    raw_proposal = {
        "roi_kept_frames": int(len(proposed)),
        "keep_summary": summarize_keep_policy(
            total_frames=int(video_meta.get("frame_count", 0) or 0),
            kept_frames=proposed,
        ),
        "first_kept_frames": proposed[:20],
        "mean_motion_score_kept": float(proposed_motion_mean),
        "mean_uncertainty_score_kept": float(proposed_unc_mean),
        "mean_amt_risk_kept": float(proposed_amt_mean),
        "mean_combined_score_kept": float(combined_mean),
        **proposed_rate,
    }
    raw_comparison = {
        "overlap": overlap,
        "count_delta_vs_baseline": int(len(proposed) - len(expected)),
        "mean_motion_score_delta": float(proposed_motion_mean - baseline_motion_mean),
        "mean_uncertainty_score_delta": float(proposed_unc_mean - baseline_unc_mean),
        "mean_amt_risk_delta": float(proposed_amt_mean - baseline_amt_mean),
        "mean_combined_score_delta": float(combined_mean - combined_baseline_mean),
        "estimated_roi_bytes_delta": float(
            proposed_rate["estimated_roi_bytes"] - baseline_rate["estimated_roi_bytes"]
        ),
        "estimated_roi_kbps_delta": float(
            proposed_rate["estimated_roi_kbps"] - baseline_rate["estimated_roi_kbps"]
        ),
        "exact_match": bool(proposed == expected),
    }
    guardrail = _segment_guardrail_decision(
        cfg=cfg,
        baseline_budget_utilization=float(baseline_rate.get("budget_utilization", 0.0) or 0.0),
        proposed_budget_utilization=float(proposed_rate.get("budget_utilization", 0.0) or 0.0),
        primary_delta=float(raw_comparison["mean_combined_score_delta"]),
    )
    guardrail["baseline_fallback_applied"] = bool(guardrail.get("applied", False))
    guardrail["reference_fallback_applied"] = False
    proposal_payload = dict(raw_proposal)
    comparison_payload = dict(raw_comparison)
    overlap_payload = overlap
    if bool(guardrail.get("applied", False)):
        proposal_payload = {
            "roi_kept_frames": int(len(expected)),
            "keep_summary": summarize_keep_policy(
                total_frames=int(video_meta.get("frame_count", 0) or 0),
                kept_frames=expected,
            ),
            "first_kept_frames": expected[:20],
            "mean_motion_score_kept": float(baseline_motion_mean),
            "mean_uncertainty_score_kept": float(baseline_unc_mean),
            "mean_amt_risk_kept": float(baseline_amt_mean),
            "mean_combined_score_kept": float(combined_baseline_mean),
            **baseline_rate,
        }
        overlap_payload = index_overlap(expected, expected)
        comparison_payload = {
            "overlap": overlap_payload,
            "count_delta_vs_baseline": 0,
            "mean_motion_score_delta": 0.0,
            "mean_uncertainty_score_delta": 0.0,
            "mean_amt_risk_delta": 0.0,
            "mean_combined_score_delta": 0.0,
            "estimated_roi_bytes_delta": 0.0,
            "estimated_roi_kbps_delta": 0.0,
            "exact_match": True,
        }
    else:
        reference_cfg = (((cfg.get("evaluation", {}) or {}).get("guardrail", {}) or {}).get("reference_fallback", {}) or {})
        reference_enabled = bool(reference_cfg.get("enabled", False))
        reference_manifest_name = str(reference_cfg.get("manifest_name", "") or "").strip()
        reference_min_advantage = float(reference_cfg.get("min_primary_advantage", 0.0) or 0.0)
        reference_payload = (
            _load_reference_experiment(cfg=cfg, manifest_name=reference_manifest_name)
            if reference_enabled and reference_manifest_name
            else None
        )
        if reference_payload is not None:
            reference_experiment = reference_payload.get("experiment", {}) or {}
            reference_comparison = reference_experiment.get("comparison", {}) or {}
            reference_proposal = reference_experiment.get("proposal", {}) or {}
            reference_primary_delta = float(reference_comparison.get("mean_combined_score_delta", 0.0) or 0.0)
            reference_budget_utilization = float(reference_proposal.get("budget_utilization", 0.0) or 0.0)
            reference_better = float(reference_primary_delta) > (
                float(raw_comparison["mean_combined_score_delta"]) + float(reference_min_advantage)
            )
            reference_feasible = float(reference_budget_utilization) <= 1.000001
            guardrail["reference_manifest_path"] = str(reference_payload.get("manifest_path", ""))
            guardrail["reference_primary_delta"] = float(reference_primary_delta)
            guardrail["reference_budget_utilization"] = float(reference_budget_utilization)
            guardrail["reference_min_primary_advantage"] = float(reference_min_advantage)
            guardrail["reference_better"] = bool(reference_better)
            guardrail["reference_feasible"] = bool(reference_feasible)
            if reference_better and reference_feasible:
                proposal_payload = dict(reference_proposal)
                comparison_payload = dict(reference_comparison)
                overlap_payload = dict(reference_comparison.get("overlap", {}) or {})
                guardrail["applied"] = True
                guardrail["reference_fallback_applied"] = True
                guardrail["selected_policy"] = "v3_reference"
                guardrail["reason"] = (
                    "Fallback to the v3 reference schedule because it achieved a higher primary score "
                    "than the raw v4 segment-DP proposal on this clip."
                )

    return {
        "name": str(((cfg.get("experiment", {}) or {}).get("name", "v4_segment_dp_amt"))),
        "description": str(
            ((cfg.get("experiment", {}) or {}).get("description", "Segment-aware ROI budget DP with AMT gap risk."))
        ),
        "budget_model": budget_model,
        "weights": {
            "motion": motion_w,
            "uncertainty": uncertainty_w,
            "amt_risk": amt_w,
        },
        "motion_features": _motion_feature_summary(motion_scores),
        "uncertainty_features": _uncertainty_feature_summary(uncertainty_scores),
        "amt_features": _amt_feature_summary(dense_amt_scores),
        "bitrate_features": bitrate_features,
        "amt_probe": amt_meta,
        "segment_dp": {
            "candidate_frame_count": int(len(candidate_frames)),
            "mandatory_frame_count": int(len(mandatory_frames)),
            "edge_count": int(len(segment_scores)),
            "initial_anchor_bytes": float(initial_anchor_bytes),
            "optimized_total_distortion": float(selection.total_distortion),
            "optimized_total_bits": float(selection.estimated_total_bits),
            "lambda_penalty": float(selection.lambda_penalty),
            "feasible": bool(selection.feasible),
            "selected_policy": str(guardrail.get("selected_policy", "segment_budget_dp")),
        },
        "guardrail": guardrail,
        "optimizer_proposal": raw_proposal,
        "optimizer_comparison": raw_comparison,
        "proposal": proposal_payload,
        "baseline_reference": {
            "roi_kept_frames": int(len(expected)),
            "first_kept_frames": expected[:20],
            "mean_motion_score_kept": float(baseline_motion_mean),
            "mean_uncertainty_score_kept": float(baseline_unc_mean),
            "mean_amt_risk_kept": float(baseline_amt_mean),
            "mean_combined_score_kept": float(combined_baseline_mean),
            **baseline_rate,
        },
        "comparison": comparison_payload,
    }


def _fixed_baseline_repro(frame_drop_json: Dict[str, Any]) -> Dict[str, Any]:
    stats = frame_drop_json.get("stats", {}) or {}
    dual = frame_drop_json.get("dual_timeline", {}) or {}
    policy = dual.get("policy", {}) or {}

    total_frames = int(stats.get("num_frames_read", 0) or 0)
    frame_ids = list(range(max(0, total_frames)))
    motion_interval = int(policy.get("roi_motion_interval", 2) or 2)
    still_interval = int(policy.get("roi_still_interval", 3) or 3)
    expected = roi_keep_indices(frame_drop_json)

    reproduced = fixed_state_interval_keep(
        frame_ids=frame_ids,
        has_detection=has_detection_from_frame_drop(frame_drop_json),
        state_by_frame=state_from_frame_drop(frame_drop_json),
        motion_interval=motion_interval,
        still_interval=still_interval,
    )
    overlap = index_overlap(expected, reproduced)

    first_mismatch = None
    if expected != reproduced:
        limit = min(len(expected), len(reproduced))
        for idx in range(limit):
            if expected[idx] != reproduced[idx]:
                first_mismatch = {
                    "position": int(idx),
                    "expected": int(expected[idx]),
                    "reproduced": int(reproduced[idx]),
                }
                break
        if first_mismatch is None:
            first_mismatch = {
                "position": int(limit),
                "expected": (None if limit >= len(expected) else int(expected[limit])),
                "reproduced": (None if limit >= len(reproduced) else int(reproduced[limit])),
            }

    return {
        "roi_motion_interval": int(motion_interval),
        "roi_still_interval": int(still_interval),
        "expected_kept_count": int(len(expected)),
        "reproduced_kept_count": int(len(reproduced)),
        "exact_match": bool(expected == reproduced),
        "first_mismatch": first_mismatch,
        "overlap": overlap,
    }


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    experiment_cfg_path: Path | None = None
    if args.experiment_config:
        experiment_cfg_path = Path(args.experiment_config).expanduser().resolve()
        cfg = _deep_merge(cfg, _load_yaml(experiment_cfg_path))
    paths_cfg = cfg.get("paths", {}) or {}

    video_path = _resolve_path(paths_cfg.get("video_path", None), cfg_path=cfg_path)
    roi_path = _resolve_path(paths_cfg.get("roi_detections_json", None), cfg_path=cfg_path)
    frame_drop_path = _resolve_path(paths_cfg.get("frame_drop_json", None), cfg_path=cfg_path)
    amt_manifest_path = _resolve_path(paths_cfg.get("amt_probe_manifest", None), cfg_path=cfg_path)
    output_dir = _resolve_path(paths_cfg.get("output_dir", "results"), cfg_path=cfg_path)

    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Configured video_path is missing. Update configs/local.yaml first.")
    if roi_path is None or not roi_path.exists():
        raise FileNotFoundError("Configured roi_detections_json is missing. Generate baseline ROI artifacts first.")
    if frame_drop_path is None or not frame_drop_path.exists():
        raise FileNotFoundError("Configured frame_drop_json is missing. Generate baseline frame-drop artifacts first.")
    if output_dir is None:
        raise RuntimeError("Could not resolve output_dir from config.")

    roi_payload = load_roi_detections(roi_path)
    frame_drop_json = load_frame_drop(frame_drop_path)
    video_meta = _video_metadata(video_path)
    roi_summary = _roi_detection_summary(roi_payload)
    frame_stats = frame_drop_json.get("stats", {}) or {}
    fixed_repro = _fixed_baseline_repro(frame_drop_json)

    total_frames = int(frame_stats.get("num_frames_read", video_meta["frame_count"]) or video_meta["frame_count"])
    roi_kept = roi_keep_indices(frame_drop_json)

    summary = {
        "config_path": str(cfg_path),
        "experiment_config_path": (str(experiment_cfg_path) if experiment_cfg_path is not None else None),
        "artifacts": {
            "video_path": str(video_path),
            "roi_detections_json": str(roi_path),
            "frame_drop_json": str(frame_drop_path),
        },
        "budget": cfg.get("budget", {}),
        "policy": cfg.get("policy", {}),
        "signals": cfg.get("signals", {}),
        "video": video_meta,
        "roi_detection": roi_summary,
        "frame_drop": {
            "policy_mode": str(frame_drop_json.get("policy_mode", frame_stats.get("policy_mode", "unknown"))),
            "roi_kept_frames": int(len(roi_kept)),
            "bg_kept_frames": int(len(frame_drop_json.get("bg_kept_frames", []) or [])),
            "drop_ratio": float(frame_stats.get("drop_ratio", 0.0) or 0.0),
            "motion_frames": int(
                sum(
                    1
                    for rec in (frame_drop_json.get("per_frame", {}) or {}).values()
                    if isinstance(rec, dict) and str(rec.get("state", "STILL")).upper() == "MOTION"
                )
            ),
            "still_frames": int(
                sum(
                    1
                    for rec in (frame_drop_json.get("per_frame", {}) or {}).values()
                    if isinstance(rec, dict) and str(rec.get("state", "STILL")).upper() == "STILL"
                )
            ),
            "keep_summary": summarize_keep_policy(total_frames=total_frames, kept_frames=roi_kept),
            "segment_summary": _segment_summary(frame_drop_json),
            "dual_policy": ((frame_drop_json.get("dual_timeline", {}) or {}).get("policy", {}) or {}),
        },
        "fixed_baseline_reproduction": fixed_repro,
        "status": "baseline_loaded",
    }

    exp_cfg = cfg.get("experiment", {}) or {}
    exp_name = str(exp_cfg.get("name", "") or "").strip().lower()
    policy_name = str(((cfg.get("policy", {}) or {}).get("name", "") or "").strip().lower())
    objective = ((cfg.get("objective", {}) or {}).get("weights", {}) or {})
    segment_dp_requested = bool(exp_name == "v4_segment_dp_amt") or bool(policy_name == "segment_budget_dp")
    motion_only_requested = bool(exp_name == "v1_motion_only") or (
        float(objective.get("motion", 0.0) or 0.0) > 0.0
        and float(objective.get("uncertainty", 0.0) or 0.0) == 0.0
        and float(objective.get("amt_risk", 0.0) or 0.0) == 0.0
    )
    motion_uncertainty_requested = bool(exp_name == "v2_motion_uncertainty") or (
        float(objective.get("motion", 0.0) or 0.0) > 0.0
        and float(objective.get("uncertainty", 0.0) or 0.0) > 0.0
        and float(objective.get("amt_risk", 0.0) or 0.0) == 0.0
    )
    motion_uncertainty_amt_requested = bool(exp_name == "v3_motion_uncertainty_amt") or (
        float(objective.get("motion", 0.0) or 0.0) > 0.0
        and float(objective.get("uncertainty", 0.0) or 0.0) > 0.0
        and float(objective.get("amt_risk", 0.0) or 0.0) > 0.0
    )
    if segment_dp_requested:
        summary["experiment"] = _segment_budget_dp_experiment(
            cfg=cfg,
            frame_drop_json=frame_drop_json,
            video_meta=video_meta,
            roi_payload=roi_payload,
            video_path=video_path,
            amt_manifest_path=amt_manifest_path,
        )
        summary["status"] = "experiment_loaded"
    elif motion_uncertainty_amt_requested:
        summary["experiment"] = _motion_uncertainty_amt_experiment(
            cfg=cfg,
            frame_drop_json=frame_drop_json,
            video_meta=video_meta,
            roi_payload=roi_payload,
            video_path=video_path,
            amt_manifest_path=amt_manifest_path,
        )
        summary["status"] = "experiment_loaded"
    elif motion_uncertainty_requested:
        summary["experiment"] = _motion_uncertainty_experiment(
            cfg=cfg,
            frame_drop_json=frame_drop_json,
            video_meta=video_meta,
            roi_payload=roi_payload,
            video_path=video_path,
        )
        summary["status"] = "experiment_loaded"
    elif motion_only_requested:
        summary["experiment"] = _motion_only_experiment(
            cfg=cfg,
            frame_drop_json=frame_drop_json,
            video_meta=video_meta,
            roi_payload=roi_payload,
            video_path=video_path,
        )
        summary["status"] = "experiment_loaded"

    if bool((cfg.get("evaluation", {}) or {}).get("save_manifest", True)):
        manifest_dir = Path(output_dir) / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_name = "offline_eval_summary.json"
        if exp_name:
            manifest_name = f"{exp_name}_summary.json"
        manifest_path = manifest_dir / manifest_name
        manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["manifest_path"] = str(manifest_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if segment_dp_requested:
        print("Offline ROI segment-aware budget-DP experiment loaded successfully.")
    elif motion_uncertainty_amt_requested:
        print("Offline ROI motion+uncertainty+AMT-risk experiment loaded successfully.")
    elif motion_uncertainty_requested:
        print("Offline ROI motion+uncertainty experiment loaded successfully.")
    elif motion_only_requested:
        print("Offline ROI motion-only experiment loaded successfully.")
    else:
        print("Offline ROI baseline loaded successfully.")


if __name__ == "__main__":
    main()
