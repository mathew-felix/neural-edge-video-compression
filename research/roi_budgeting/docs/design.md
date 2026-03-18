# ROI Budgeting Design

## Objective

Build an offline research loop for budget-aware ROI anchor selection without changing the production pipeline.

## Current Scope

- ROI budget awareness only
- BG policy remains fixed
- Offline evaluation first
- No production integration until a policy clearly wins

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

## Integration Gate

Before touching production code, require:

- repeatable wins on several clips
- explicit bitrate tracking
- clear failure analysis for misses
- a minimal integration plan that leaves BG unchanged
