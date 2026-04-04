# GitHub Figure Set

This folder contains the figure set that should be used for the public GitHub presentation of the thesis artifact.

The figure policy is intentional:

- keep the figures aligned to the defended thesis
- use images that explain the system and its practical tradeoffs quickly
- avoid LPIPS-based charts in the public repo figure set
- keep LPIPS only in the thesis text and deeper evaluation notes

## Included Figures

1. `08_github_summary.png`
   GitHub-specific summary figure derived from the defended thesis metrics.
   Use for: the main repo visual, recruiter links, and quick project explanation without LPIPS-based charts.

2. `01_compression_stage.png`
   Thesis source: Chapter 3 compression-stage workflow.
   Use for: showing the edge-side pipeline from sparse ROI generation through ZIP archive creation.

3. `02_decompression_stage.png`
   Thesis source: Chapter 3 decompression-stage workflow.
   Use for: showing the server-side reconstruction path and AMT-based ROI restoration.

4. `03_roi_stage_speed_vs_recall.png`
   Thesis source: Figure 4.11.
   Use for: showing the detector-cost versus ROI-recall tradeoff that justifies sparse detection plus propagation.

5. `04_roi_stage_continuity_metrics.png`
   Thesis source: Figure 4.12.
   Use for: showing that propagation keeps the ROI timeline close to the dense reference.

6. `05_runtime_profile.png`
   Thesis source: Figure 4.51.
   Use for: showing that decompression/reconstruction is the heavier systems bottleneck.

7. `06_day_example.png`
   Thesis source: Figure 4.42.
   Use for: qualitative day-scene reconstruction example.

8. `07_night_example.png`
   Thesis source: Figure 4.43.
   Use for: qualitative night-scene reconstruction example.

## Intentionally Excluded From The GitHub Figure Set

These thesis figures are valid academically, but they should not be the public GitHub figures:

- `fig_421_frame_removal_policy_tradeoffs`
- `fig_431_roi_lpips_vs_archive_size`
- `fig_432_frame_drop_tradeoffs`
- `fig_441_released_pipeline_summary`
- `fig_444_released_pipeline_safari_summary`

Reason:

they include LPIPS panels or LPIPS-centered messaging, and the GitHub presentation should stay simpler and more defensible for broad readers by emphasizing size reduction, ROI PSNR, ROI MS-SSIM, detector savings, runtime, and qualitative examples.

## Recommended Figure Order For The README Or Portfolio Links

1. GitHub summary figure
2. compression-stage workflow
3. decompression-stage workflow
4. ROI-stage speed versus recall
5. runtime profile
6. one day example
7. one night example

## Source Script

The summary figure is generated from:

- `generate_github_summary_figure.py`

Run it from the repo root with:

```bash
python docs/figures/generate_github_summary_figure.py
```
