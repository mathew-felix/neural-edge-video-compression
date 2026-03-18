from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from roi_budgeting.data.load_frame_drop import roi_segments


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    q_clamped = max(0.0, min(1.0, float(q)))
    pos = q_clamped * float(len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    alpha = pos - float(lo)
    return (vals[lo] * (1.0 - alpha)) + (vals[hi] * alpha)


def build_motion_scores(motion_features: Mapping[int, Mapping[str, Any]]) -> Dict[int, Dict[str, float]]:
    center_vals = [
        float((rec.get("center_speed", 0.0) or 0.0))
        for rec in motion_features.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]
    area_vals = [
        float((rec.get("area_delta_abs", 0.0) or 0.0))
        for rec in motion_features.values()
        if float((rec.get("has_roi", 0.0) or 0.0)) > 0.0
    ]

    center_scale = _quantile(center_vals, 0.90) or max(center_vals or [1.0])
    area_scale = _quantile(area_vals, 0.90) or max(area_vals or [1.0])
    center_scale = max(1e-6, float(center_scale))
    area_scale = max(1e-6, float(area_scale))

    out: Dict[int, Dict[str, float]] = {}
    for frame_idx, rec in motion_features.items():
        has_roi = float(rec.get("has_roi", 0.0) or 0.0) > 0.0
        center_speed = float(rec.get("center_speed", 0.0) or 0.0)
        area_delta = float(rec.get("area_delta_abs", 0.0) or 0.0)
        center_norm = min(1.0, center_speed / center_scale) if has_roi else 0.0
        area_norm = min(1.0, area_delta / area_scale) if has_roi else 0.0
        score = (0.7 * center_norm) + (0.3 * area_norm)
        out[int(frame_idx)] = {
            "has_roi": 1.0 if has_roi else 0.0,
            "center_speed": center_speed,
            "area_delta_abs": area_delta,
            "center_speed_norm": float(center_norm),
            "area_delta_norm": float(area_norm),
            "motion_score": float(score),
        }
    return out


def _allocate_extra_slots(
    *,
    capacities: Sequence[int],
    extra_slots: int,
) -> List[int]:
    if extra_slots <= 0 or not capacities:
        return [0 for _ in capacities]
    total_capacity = sum(max(0, int(v)) for v in capacities)
    if total_capacity <= 0:
        return [0 for _ in capacities]

    allocations = [0 for _ in capacities]
    remainders: List[Tuple[float, int]] = []
    remaining = int(extra_slots)
    for idx, cap in enumerate(capacities):
        cap_int = max(0, int(cap))
        if cap_int <= 0:
            continue
        ideal = float(extra_slots) * (float(cap_int) / float(total_capacity))
        take = min(cap_int, int(math.floor(ideal)))
        allocations[idx] = take
        remaining -= take
        remainders.append((ideal - float(take), idx))

    for _frac, idx in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[idx] >= max(0, int(capacities[idx])):
            continue
        allocations[idx] += 1
        remaining -= 1

    if remaining > 0:
        for idx, cap in enumerate(capacities):
            while remaining > 0 and allocations[idx] < max(0, int(cap)):
                allocations[idx] += 1
                remaining -= 1
            if remaining <= 0:
                break
    return allocations


def _choose_from_bins(
    *,
    candidates: Sequence[int],
    scores: Mapping[int, float],
    count: int,
) -> List[int]:
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(sorted(int(v) for v in candidates))

    ordered = sorted(int(v) for v in candidates)
    chosen: List[int] = []
    used: set[int] = set()
    n = len(ordered)
    for slot in range(int(count)):
        start = int(math.floor((slot * n) / float(count)))
        end = int(math.floor(((slot + 1) * n) / float(count)))
        if end <= start:
            end = min(n, start + 1)
        window = [ordered[i] for i in range(start, min(end, n)) if ordered[i] not in used]
        if not window:
            window = [frame_idx for frame_idx in ordered if frame_idx not in used]
        if not window:
            break
        best = max(window, key=lambda frame_idx: (float(scores.get(frame_idx, 0.0)), -frame_idx))
        chosen.append(int(best))
        used.add(int(best))

    if len(chosen) < int(count):
        leftovers = [frame_idx for frame_idx in ordered if frame_idx not in used]
        leftovers.sort(key=lambda frame_idx: (float(scores.get(frame_idx, 0.0)), -frame_idx), reverse=True)
        for frame_idx in leftovers:
            if len(chosen) >= int(count):
                break
            chosen.append(int(frame_idx))
    return sorted(set(chosen))


def select_scored_roi_anchors(
    *,
    frame_drop_json: Mapping[str, Any],
    frame_scores: Mapping[int, float],
    target_keep_count: int,
    force_keep_roi_birth: bool,
    force_keep_roi_death: bool,
    force_keep_segment_bounds: bool,
) -> List[int]:
    segments = roi_segments(dict(frame_drop_json))
    if not segments or target_keep_count <= 0:
        return []

    score_map = {int(frame_idx): float(score) for frame_idx, score in frame_scores.items()}

    segment_frames: List[List[int]] = [list(range(int(start), int(end) + 1)) for start, end in segments]
    mandatory_by_segment: List[List[int]] = []
    mandatory_total = 0
    capacities: List[int] = []
    for frames in segment_frames:
        mandatory: List[int] = []
        if frames:
            start = int(frames[0])
            end = int(frames[-1])
            if bool(force_keep_segment_bounds) or bool(force_keep_roi_birth):
                mandatory.append(start)
            if (bool(force_keep_segment_bounds) or bool(force_keep_roi_death)) and end != start:
                mandatory.append(end)
        mandatory = sorted(set(mandatory))
        mandatory_by_segment.append(mandatory)
        mandatory_total += len(mandatory)
        capacities.append(max(0, len(frames) - len(mandatory)))

    if target_keep_count <= mandatory_total:
        collapsed: List[int] = []
        for mandatory in mandatory_by_segment:
            collapsed.extend(mandatory)
        return sorted(set(collapsed))[: int(target_keep_count)]

    extra_allocations = _allocate_extra_slots(
        capacities=capacities,
        extra_slots=int(target_keep_count) - int(mandatory_total),
    )

    kept: List[int] = []
    for frames, mandatory, extra_count in zip(segment_frames, mandatory_by_segment, extra_allocations):
        mandatory_set = set(int(v) for v in mandatory)
        kept.extend(mandatory)
        if extra_count <= 0:
            continue
        candidates = [frame_idx for frame_idx in frames if int(frame_idx) not in mandatory_set]
        kept.extend(
            _choose_from_bins(
                candidates=candidates,
                scores=score_map,
                count=int(extra_count),
            )
        )

    kept_unique = sorted(set(int(v) for v in kept))
    if len(kept_unique) > int(target_keep_count):
        protected = set()
        for mandatory in mandatory_by_segment:
            protected.update(int(v) for v in mandatory)
        removable = [frame_idx for frame_idx in kept_unique if frame_idx not in protected]
        removable.sort(key=lambda frame_idx: (score_map.get(frame_idx, 0.0), -frame_idx))
        trimmed = list(kept_unique)
        while len(trimmed) > int(target_keep_count) and removable:
            victim = removable.pop(0)
            if victim in trimmed:
                trimmed.remove(victim)
        kept_unique = sorted(trimmed)
    return kept_unique


def _rank_candidates(
    *,
    candidates: Sequence[int],
    scores: Mapping[int, float],
    costs: Mapping[int, float],
) -> List[int]:
    def _key(frame_idx: int) -> tuple[float, float, float, int]:
        score = float(scores.get(frame_idx, 0.0) or 0.0)
        cost = max(1e-9, float(costs.get(frame_idx, 0.0) or 0.0))
        density = score / cost
        return (density, score, -cost, -int(frame_idx))

    return sorted((int(v) for v in candidates), key=_key, reverse=True)


def select_scored_roi_anchors_by_budget(
    *,
    frame_drop_json: Mapping[str, Any],
    frame_scores: Mapping[int, float],
    frame_costs: Mapping[int, float],
    target_bytes: float,
    force_keep_roi_birth: bool,
    force_keep_roi_death: bool,
    force_keep_segment_bounds: bool,
    max_gap_frames: int,
) -> List[int]:
    segments = roi_segments(dict(frame_drop_json))
    if not segments or target_bytes <= 0.0:
        return []

    score_map = {int(frame_idx): float(score) for frame_idx, score in frame_scores.items()}
    cost_map = {int(frame_idx): max(0.0, float(cost)) for frame_idx, cost in frame_costs.items()}

    selected: set[int] = set()
    bytes_used = 0.0
    mandatory_by_segment: List[List[int]] = []
    for start, end in segments:
        frames = list(range(int(start), int(end) + 1))
        mandatory: List[int] = []
        if frames:
            first = int(frames[0])
            last = int(frames[-1])
            if bool(force_keep_segment_bounds) or bool(force_keep_roi_birth):
                mandatory.append(first)
            if (bool(force_keep_segment_bounds) or bool(force_keep_roi_death)) and last != first:
                mandatory.append(last)
        mandatory = sorted(set(mandatory))
        mandatory_by_segment.append(mandatory)
        for frame_idx in mandatory:
            if frame_idx in selected:
                continue
            selected.add(int(frame_idx))
            bytes_used += float(cost_map.get(int(frame_idx), 0.0) or 0.0)

    primary_candidates: List[int] = []
    secondary_candidates: List[int] = []
    bin_size = max(1, int(max_gap_frames) if int(max_gap_frames) > 0 else 12)
    for (start, end), mandatory in zip(segments, mandatory_by_segment):
        mandatory_set = {int(v) for v in mandatory}
        frames = [frame_idx for frame_idx in range(int(start), int(end) + 1) if frame_idx not in mandatory_set]
        if not frames:
            continue
        for offset in range(0, len(frames), bin_size):
            chunk = frames[offset : offset + bin_size]
            if not chunk:
                continue
            ranked = _rank_candidates(candidates=chunk, scores=score_map, costs=cost_map)
            if ranked:
                primary_candidates.append(int(ranked[0]))
                secondary_candidates.extend(int(v) for v in ranked[1:])

    for candidate_pool in (
        _rank_candidates(candidates=primary_candidates, scores=score_map, costs=cost_map),
        _rank_candidates(candidates=secondary_candidates, scores=score_map, costs=cost_map),
    ):
        for frame_idx in candidate_pool:
            if frame_idx in selected:
                continue
            cost = float(cost_map.get(frame_idx, 0.0) or 0.0)
            if cost <= 0.0:
                selected.add(int(frame_idx))
                continue
            if bytes_used + cost <= float(target_bytes) + 1e-9:
                selected.add(int(frame_idx))
                bytes_used += cost

    return sorted(selected)


def select_motion_only_roi_anchors(
    *,
    frame_drop_json: Mapping[str, Any],
    motion_scores: Mapping[int, Mapping[str, Any]],
    target_keep_count: int,
    force_keep_roi_birth: bool,
    force_keep_roi_death: bool,
    force_keep_segment_bounds: bool,
) -> List[int]:
    score_map = {
        int(frame_idx): float((rec.get("motion_score", 0.0) or 0.0))
        for frame_idx, rec in motion_scores.items()
    }
    return select_scored_roi_anchors(
        frame_drop_json=frame_drop_json,
        frame_scores=score_map,
        target_keep_count=target_keep_count,
        force_keep_roi_birth=force_keep_roi_birth,
        force_keep_roi_death=force_keep_roi_death,
        force_keep_segment_bounds=force_keep_segment_bounds,
    )
