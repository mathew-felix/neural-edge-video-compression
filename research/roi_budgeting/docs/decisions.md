# ROI Budgeting Decisions

Use this file as a lightweight decision log.

## Template

- Date:
- Decision:
- Context:
- Why:
- Tradeoffs:
- Next validation step:

## Initial Decisions

- Date: pending
- Decision: Keep ROI-budgeting work isolated under `research/roi_budgeting/`
- Context: Production pipeline should remain stable during research iteration.
- Why: Fast iteration with low integration risk.
- Tradeoffs: Some duplication at the start.
- Next validation step: Establish offline baselines and compare policies before integration.

## Decision Log

- Date: 2026-03-18
- Decision: Keep the research pipeline ROI-only and leave BG on the fixed production policy.
- Context: AMT interpolation is applied on the ROI path, and the initial research question was about ROI budgeting rather than full dual-stream control.
- Why: This narrows the search space and keeps the work aligned with the highest-value source of reconstruction error.
- Tradeoffs: Some bitrate inefficiency may remain in BG because BG is not yet budget-aware.
- Next validation step: Revisit BG only after the ROI policy is stable enough for shadow-mode integration.

- Date: 2026-03-18
- Decision: Use direct ROI temporal probe bytes as the budget model instead of keep-count matching.
- Context: Matching the number of anchors is not the same as matching the bitrate budget.
- Why: A byte-aware objective is closer to the real codec problem and makes cross-policy comparison more honest.
- Tradeoffs: Probe encoding adds runtime and remains an approximation of the production codec.
- Next validation step: Compare the probe budget against a closer codec-side rate estimate during later integration work.

- Date: 2026-03-18
- Decision: Promote segment-aware `v4` over frame-scoring-only policies as the main research direction.
- Context: AMT difficulty is fundamentally about the gap between kept anchors, not isolated frames.
- Why: Segment-aware optimization better matches the interpolation problem and improved benchmark performance.
- Tradeoffs: More implementation complexity and less intuitive behavior than a simple frame-ranking policy.
- Next validation step: Keep `v4` as the lead policy and review its qualitative wins and fallback cases.

- Date: 2026-03-18
- Decision: Add a conservative low-budget guardrail to `v4`.
- Context: Several clips had cheap fixed baselines and did not benefit from spending more ROI bitrate on the optimized schedule.
- Why: Falling back to baseline on cheap-safe clips removed those regressions while preserving most of the wins.
- Tradeoffs: The reported `v4` policy is now a hybrid, not a pure segment-DP policy.
- Next validation step: Track how often the fallback fires on unseen clips and confirm that it is not masking a deeper modeling bug.

- Date: 2026-03-18
- Decision: Add a `v3` reference fallback for clips where raw `v4` underperforms `v3`.
- Context: After the baseline guardrail, `multi_cow_day_2` still favored `v3` over raw `v4`.
- Why: Reusing the already-computed `v3` schedule is a simple and low-risk way to avoid that remaining miss.
- Tradeoffs: The final `v4` benchmark result is a guarded selector over three possible schedules: baseline, `v3`, and raw `v4`.
- Next validation step: Replace this fallback later only if a cleaner unified objective consistently removes the need for it.

- Date: 2026-03-18
- Decision: Treat the current benchmark result as a research milestone, not a production integration decision.
- Context: The guarded hybrid `v4` now wins across the 12-clip benchmark and removes regressions vs baseline.
- Why: The result is strong enough to justify shadow-mode work, but still needs qualitative review and unseen-clip validation.
- Tradeoffs: Progress to production will be slower, but the integration decision will be better grounded.
- Next validation step: Freeze this milestone, review qualitative clips, and then test on unseen videos before integrating.
