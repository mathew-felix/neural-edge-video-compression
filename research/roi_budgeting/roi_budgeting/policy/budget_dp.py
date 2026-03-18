from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


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
    lambda_penalty: float
    feasible: bool


def _contiguous_ranges(frames: Sequence[int]) -> List[List[int]]:
    ordered = sorted(set(int(v) for v in frames))
    if not ordered:
        return []
    ranges: List[List[int]] = [[ordered[0]]]
    for frame_idx in ordered[1:]:
        if frame_idx == ranges[-1][-1] + 1:
            ranges[-1].append(frame_idx)
        else:
            ranges.append([frame_idx])
    return ranges


def build_segment_scores_from_frame_scores(
    *,
    candidate_frames: Sequence[int],
    mandatory_frames: Iterable[int],
    frame_scores: Mapping[int, float],
    frame_costs: Mapping[int, float],
    max_gap_frames: int,
) -> Dict[tuple[int, int], SegmentScore]:
    """
    Build a monotonic DAG of anchor-to-anchor segment scores.

    Each edge (i, j) represents keeping anchor j after anchor i and skipping
    the interior frames in between. The distortion is the sum of the dense
    frame-level risk over the skipped interval, while the rate is the byte cost
    of storing anchor j.
    """
    mandatory_set = {int(v) for v in mandatory_frames}
    ranges = _contiguous_ranges(candidate_frames)
    segment_scores: Dict[tuple[int, int], SegmentScore] = {}

    for seg_idx, frames in enumerate(ranges):
        if len(frames) < 2:
            continue
        prefix: List[float] = [0.0]
        for frame_idx in frames:
            prefix.append(prefix[-1] + float(frame_scores.get(int(frame_idx), 0.0) or 0.0))

        mandatory_positions = [
            idx for idx, frame_idx in enumerate(frames) if int(frame_idx) in mandatory_set
        ]
        next_mandatory_by_pos: Dict[int, int] = {}
        next_mandatory_pos: int | None = None
        for idx in range(len(frames) - 1, -1, -1):
            next_mandatory_by_pos[idx] = -1 if next_mandatory_pos is None else int(next_mandatory_pos)
            if int(frames[idx]) in mandatory_set:
                next_mandatory_pos = idx

        for start_pos, start_frame in enumerate(frames[:-1]):
            max_end_pos = len(frames) - 1
            if int(max_gap_frames) > 0:
                max_end_pos = min(max_end_pos, start_pos + int(max_gap_frames))
            next_mand = next_mandatory_by_pos.get(start_pos + 1, -1)
            if next_mand >= 0:
                max_end_pos = min(max_end_pos, int(next_mand))

            for end_pos in range(start_pos + 1, max_end_pos + 1):
                end_frame = int(frames[end_pos])
                distortion = float(prefix[end_pos] - prefix[start_pos + 1])
                segment_scores[(int(start_frame), end_frame)] = SegmentScore(
                    start_frame=int(start_frame),
                    end_frame=end_frame,
                    distortion=float(max(0.0, distortion)),
                    estimated_bits=float(max(0.0, frame_costs.get(end_frame, 0.0) or 0.0)),
                )

        if seg_idx + 1 >= len(ranges):
            continue
        prev_end = int(frames[-1])
        next_start = int(ranges[seg_idx + 1][0])
        segment_scores[(prev_end, next_start)] = SegmentScore(
            start_frame=int(prev_end),
            end_frame=int(next_start),
            distortion=0.0,
            estimated_bits=float(max(0.0, frame_costs.get(next_start, 0.0) or 0.0)),
        )

    return segment_scores


def _reconstruct_path(
    *,
    prev: Mapping[int, int],
    end_frame: int,
) -> List[int]:
    path = [int(end_frame)]
    cur = int(end_frame)
    while cur in prev:
        cur = int(prev[cur])
        path.append(cur)
    path.reverse()
    return path


def _solve_for_lambda(
    *,
    ordered_frames: Sequence[int],
    incoming_edges: Mapping[int, Sequence[SegmentScore]],
    lambda_penalty: float,
    initial_bits: float,
) -> BudgetedSelection:
    start_frame = int(ordered_frames[0])
    end_frame = int(ordered_frames[-1])

    lagrangian_cost: Dict[int, float] = {start_frame: 0.0}
    total_bits: Dict[int, float] = {start_frame: 0.0}
    total_distortion: Dict[int, float] = {start_frame: 0.0}
    prev: Dict[int, int] = {}

    for frame_idx in ordered_frames[1:]:
        best_cost = math.inf
        best_bits = math.inf
        best_dist = math.inf
        best_prev: int | None = None
        for edge in incoming_edges.get(int(frame_idx), []):
            start = int(edge.start_frame)
            if start not in lagrangian_cost:
                continue
            cand_dist = float(total_distortion[start]) + float(edge.distortion)
            cand_bits = float(total_bits[start]) + float(edge.estimated_bits)
            cand_cost = float(lagrangian_cost[start]) + float(edge.distortion) + (float(lambda_penalty) * float(edge.estimated_bits))
            better = cand_cost < best_cost - 1e-9
            tie = abs(cand_cost - best_cost) <= 1e-9 and (
                cand_dist < best_dist - 1e-9
                or (abs(cand_dist - best_dist) <= 1e-9 and cand_bits < best_bits - 1e-9)
            )
            if better or tie:
                best_cost = float(cand_cost)
                best_bits = float(cand_bits)
                best_dist = float(cand_dist)
                best_prev = int(start)

        if best_prev is None:
            continue
        lagrangian_cost[int(frame_idx)] = float(best_cost)
        total_bits[int(frame_idx)] = float(best_bits)
        total_distortion[int(frame_idx)] = float(best_dist)
        prev[int(frame_idx)] = int(best_prev)

    if end_frame not in total_bits:
        raise RuntimeError("Budget DP could not find a path to the final ROI anchor.")

    kept = _reconstruct_path(prev=prev, end_frame=end_frame)
    return BudgetedSelection(
        kept_frames=kept,
        estimated_total_bits=float(initial_bits + total_bits[end_frame]),
        total_distortion=float(total_distortion[end_frame]),
        lambda_penalty=float(lambda_penalty),
        feasible=True,
    )


def select_budgeted_roi_anchors(
    *,
    candidate_frames: Sequence[int],
    mandatory_frames: Iterable[int],
    segment_scores: Mapping[tuple[int, int], SegmentScore],
    bit_budget: float,
    initial_bits: float = 0.0,
    search_steps: int = 28,
) -> BudgetedSelection:
    """
    Solve a segment-aware ROI anchor selection problem with a Lagrangian DP.

    The graph is a monotonic DAG over candidate anchor frames. Each edge models
    skipping a segment between two kept anchors, incurring:
    - `distortion`: estimated reconstruction / semantic penalty over skipped frames
    - `estimated_bits`: byte cost of keeping the end anchor
    """
    ordered = sorted(set(int(v) for v in candidate_frames))
    if not ordered:
        return BudgetedSelection(
            kept_frames=[],
            estimated_total_bits=0.0,
            total_distortion=0.0,
            lambda_penalty=0.0,
            feasible=True,
        )

    mandatory_set = {int(v) for v in mandatory_frames}
    if ordered[0] not in mandatory_set or ordered[-1] not in mandatory_set:
        raise ValueError("Budget DP requires the first and last candidate frames to be mandatory anchors.")

    incoming: Dict[int, List[SegmentScore]] = {}
    for edge in segment_scores.values():
        incoming.setdefault(int(edge.end_frame), []).append(edge)

    zero_lambda_solution = _solve_for_lambda(
        ordered_frames=ordered,
        incoming_edges=incoming,
        lambda_penalty=0.0,
        initial_bits=float(initial_bits),
    )
    if zero_lambda_solution.estimated_total_bits <= float(bit_budget) + 1e-9:
        return zero_lambda_solution

    high = 1.0
    high_solution = _solve_for_lambda(
        ordered_frames=ordered,
        incoming_edges=incoming,
        lambda_penalty=float(high),
        initial_bits=float(initial_bits),
    )
    while high_solution.estimated_total_bits > float(bit_budget) + 1e-9 and high < 1e9:
        high *= 2.0
        high_solution = _solve_for_lambda(
            ordered_frames=ordered,
            incoming_edges=incoming,
            lambda_penalty=float(high),
            initial_bits=float(initial_bits),
        )

    best_feasible: BudgetedSelection | None = None
    if high_solution.estimated_total_bits <= float(bit_budget) + 1e-9:
        best_feasible = high_solution

    low = 0.0
    for _ in range(max(8, int(search_steps))):
        mid = 0.5 * (low + high)
        mid_solution = _solve_for_lambda(
            ordered_frames=ordered,
            incoming_edges=incoming,
            lambda_penalty=float(mid),
            initial_bits=float(initial_bits),
        )
        if mid_solution.estimated_total_bits <= float(bit_budget) + 1e-9:
            if (
                best_feasible is None
                or mid_solution.total_distortion < best_feasible.total_distortion - 1e-9
                or (
                    abs(mid_solution.total_distortion - best_feasible.total_distortion) <= 1e-9
                    and mid_solution.estimated_total_bits > best_feasible.estimated_total_bits + 1e-9
                )
            ):
                best_feasible = mid_solution
            high = mid
        else:
            low = mid

    if best_feasible is not None:
        return best_feasible

    fallback = _solve_for_lambda(
        ordered_frames=ordered,
        incoming_edges=incoming,
        lambda_penalty=float(high),
        initial_bits=float(initial_bits),
    )
    return BudgetedSelection(
        kept_frames=fallback.kept_frames,
        estimated_total_bits=fallback.estimated_total_bits,
        total_distortion=fallback.total_distortion,
        lambda_penalty=fallback.lambda_penalty,
        feasible=bool(fallback.estimated_total_bits <= float(bit_budget) + 1e-9),
    )
