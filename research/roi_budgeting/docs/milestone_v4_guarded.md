# Guarded V4 Milestone

Date: 2026-03-18

## Summary

The ROI-budgeting research stack now has a benchmarked leading policy:

- `v4_segment_dp_amt`
  - segment-aware ROI budget DP
  - dense AMT gap-risk term
  - conservative fallback to the fixed baseline on cheap-safe clips
  - conservative fallback to `v3` when `v3` is measurably better than raw `v4`

This is currently the strongest result in the research workspace.

## Benchmark Result

Benchmark folder:

- `research/roi_budgeting/test`

Aggregate report:

- `research/roi_budgeting/results/benchmark/_aggregate/experiment_aggregate.md`

Headline numbers on 12 clips:

- `fixed_baseline`
  - mean primary delta: `0.000000`
  - mean estimated ROI bitrate: `96.343639 kbps`
- `v2_motion_uncertainty`
  - mean primary delta: `0.035484`
  - mean estimated ROI bitrate: `126.332315 kbps`
- `v3_motion_uncertainty_amt`
  - mean primary delta: `0.147374`
  - mean estimated ROI bitrate: `126.354324 kbps`
- `v4_segment_dp_amt`
  - mean primary delta: `0.176478`
  - mean estimated ROI bitrate: `107.515719 kbps`

## Final Outcome

Compared with `v3`:

- `v4` wins on 11 clips
- `v4` ties on 1 clip
- `v4` loses on 0 clips

Compared with the fixed baseline:

- `v4` has 0 negative-primary-delta clips on the current benchmark

## Fallback Decisions

Current `v4` selected policy by clip family:

- `segment_budget_dp`
  - `multi_birds_day_2`
  - `multi_birds_day_4`
  - `multi_cow_day_1`
  - `multi_shunk_night`
  - `night_1`
  - `single_fox_night`
  - `single_raccon_night`
- `fixed_baseline`
  - `multi_racoon_night`
  - `night_2`
  - `single_deer_night`
  - `single_shunk_night`
- `v3_reference`
  - `multi_cow_day_2`

## What This Means

The research result is now stronger in two ways:

- It improves the main benchmark objective more than `v3`.
- It no longer accepts obvious regressions against the fixed baseline.

The result is still a research milestone, not a production-ready integration.

## Next Step

The next recommended phase is:

1. Freeze this milestone in git.
2. Review qualitative wins and fallback clips.
3. Test on unseen videos.
4. Add a shadow-mode integration path before replacing the production ROI heuristic.
