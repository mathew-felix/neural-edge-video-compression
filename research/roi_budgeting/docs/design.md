# ROI Budgeting Design

## Objective

Build an offline research loop for budget-aware ROI anchor selection without changing the production pipeline.

## Current Scope

- ROI budget awareness only
- BG policy remains fixed
- Offline evaluation first
- No production integration until a policy clearly wins

## Current Best Policy

The leading research policy is now a guarded hybrid `v4`:

- First, solve a segment-aware ROI anchor-selection problem under a bitrate budget.
- If the fixed baseline is already cheap and the optimized schedule does not show a clear gain, fall back to the fixed baseline.
- If the raw segment-DP schedule underperforms the already-computed `v3` schedule on the same clip, fall back to `v3`.

In practice this means the published `v4` result is not "always use segment DP." It is:

- `segment_budget_dp` when that path helps
- `fixed_baseline` when the baseline is already cheap and safe
- `v3_reference` when `v3` is measurably better than raw `v4`

## Research Questions

1. How much does motion alone explain ROI anchor density?
2. How much does calibrated ROI uncertainty improve anchor placement?
3. How predictive is AMT probe error for reconstruction failure?
4. Which objective behaves best under a fixed ROI bitrate budget?

## Proposed Research Loop

1. Read baseline artifacts from existing pipeline runs.
2. Build per-frame and per-segment features.
3. Score candidate ROI anchor gaps with motion, uncertainty, and AMT-risk terms.
4. Solve a budgeted ROI selection problem offline.
5. Compare quality and bitrate against the fixed heuristic.

## Current Experiment Stack

- `v1_motion_only`
  - Budget-aware ROI selection using motion only.
- `v2_motion_uncertainty`
  - Budget-aware ROI selection using motion and calibrated detector uncertainty.
- `v3_motion_uncertainty_amt`
  - Budget-aware ROI selection using motion, uncertainty, and AMT probe difficulty.
- `v4_segment_dp_amt`
  - Segment-aware ROI budget DP using motion, uncertainty, and dense AMT gap risk.
  - Includes the current conservative fallback layers.

## Benchmark Status

The current benchmark lives under `research/roi_budgeting/test` and the generated reports live under `research/roi_budgeting/results/benchmark/_aggregate`.

On the 12-clip benchmark, the guarded hybrid `v4` is the best current result:

- `fixed_baseline`
  - mean primary delta: `0.000000`
  - mean estimated ROI bitrate: `96.34 kbps`
- `v3_motion_uncertainty_amt`
  - mean primary delta: `0.147374`
  - mean estimated ROI bitrate: `126.35 kbps`
- `v4_segment_dp_amt`
  - mean primary delta: `0.176478`
  - mean estimated ROI bitrate: `107.52 kbps`

The final hybrid `v4` outcome is:

- better than `v3` on 11 clips
- tied with `v3` on 1 clip
- worse than `v3` on 0 clips
- worse than the fixed baseline on 0 clips

## Operational Interpretation

The current research result is strong enough to justify shadow-mode integration work, but not a direct production replacement yet.

What we now trust:

- explicit ROI bitrate accounting
- full AMT probing on local CPU for benchmark evaluation
- repeatable multi-clip wins over the old fixed heuristic
- conservative fallbacks for the obvious failure cases

What we still do not trust enough yet:

- generalization beyond the current benchmark folder
- qualitative visual quality without manual review of top wins and fallback clips
- production integration without a shadow-mode comparison path

## Integration Gate

Before touching production code, require:

- repeatable wins on several clips
- explicit bitrate tracking
- clear failure analysis for misses
- a minimal integration plan that leaves BG unchanged
