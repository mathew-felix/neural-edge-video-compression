from __future__ import annotations

from typing import Dict, Iterable, Set


def summarize_keep_policy(*, total_frames: int, kept_frames: Iterable[int]) -> Dict[str, float]:
    kept_set: Set[int] = {int(v) for v in kept_frames}
    total = max(0, int(total_frames))
    kept = len([v for v in kept_set if 0 <= v < total]) if total > 0 else len(kept_set)
    dropped = max(0, total - kept) if total > 0 else 0
    keep_ratio = float(kept) / float(total) if total > 0 else 0.0
    return {
        "total_frames": float(total),
        "kept_frames": float(kept),
        "dropped_frames": float(dropped),
        "keep_ratio": keep_ratio,
    }


def index_overlap(reference: Iterable[int], proposed: Iterable[int]) -> Dict[str, float]:
    ref = {int(v) for v in reference}
    prop = {int(v) for v in proposed}
    inter = len(ref & prop)
    union = len(ref | prop)
    return {
        "reference_count": float(len(ref)),
        "proposed_count": float(len(prop)),
        "intersection": float(inter),
        "jaccard": (float(inter) / float(union)) if union > 0 else 1.0,
    }
