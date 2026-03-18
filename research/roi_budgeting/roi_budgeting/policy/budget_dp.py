from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class SegmentScore:
    start_frame: int
    end_frame: int
    distortion: float
    estimated_bits: float


@dataclass(frozen=True)
class BudgetedSelection:
    kept_frames: List[int]
    estimated_total_bits: float
    total_distortion: float


def select_budgeted_roi_anchors(
    *,
    candidate_frames: Sequence[int],
    mandatory_frames: Iterable[int],
    segment_scores: Mapping[tuple[int, int], SegmentScore],
    bit_budget: float,
) -> BudgetedSelection:
    """
    Placeholder API for the budget-aware ROI scheduler.

    Intended final behavior:
    - choose a monotonic subset of ROI anchor frames
    - honor mandatory anchors
    - minimize distortion under the ROI budget

    The actual optimizer is intentionally left unimplemented in this scaffold so
    we can design and validate the objective first.
    """
    del candidate_frames, mandatory_frames, segment_scores, bit_budget
    raise NotImplementedError("Budget-aware ROI DP scheduler is not implemented yet.")
