# Dataset And Evaluation Notes

This document describes the evaluation setup used in the thesis and the meaning of the reported metrics.

## Evaluation Objective

The evaluation is designed to test the following question:

can wildlife video be compressed in a way that reduces transmitted size substantially while preserving the animal region better than treating the whole frame uniformly?

The evaluation emphasizes:

- ROI quality
- transmitted archive size
- runtime and system cost
- behavior under edge-oriented constraints

## Data Scope

The thesis reports results on unseen wildlife-video clips used as a held-out evaluation set.

Reported test-set properties:

- `20` unseen clips
- day and night conditions
- varied animals and motion patterns

The raw evaluation videos are not bundled with the repository because:

- the dataset may be too large for source control
- licensing or redistribution constraints may apply
- the repository stores code and configuration rather than experimental media

## Local Data Convention

Expected local working directories:

- `data/`
- `models/`
- `outputs/`

Example path used in commands:

- `data/tune/video.mp4`

This is a local placeholder path, not a bundled dataset.

## Main Metrics

The most important metrics in the thesis are:

- ROI PSNR
- ROI MS-SSIM
- transmitted archive size
- compression ratio
- runtime and memory cost

Full-frame metrics are reported as secondary context.

The thesis also reports ROI LPIPS as a supporting perceptual metric. The repo documentation and figure files focus on ROI PSNR, ROI MS-SSIM, size reduction, and runtime.

## Metric Definitions

### ROI PSNR

Numerical reconstruction fidelity on the animal region only.

Higher is better.

### ROI MS-SSIM

Structural similarity on the animal region.

Higher is better.

### Transmitted Archive Size

The size of the final transmitted archive written by the pipeline.

### Compression Ratio

The ratio between original input size and transmitted archive size.

## Held-Out Test-Set Summary

Reported summary on the held-out test set:

- original test-set size: `399.16 MB`
- transmitted archive size: `10.36 MB`
- transmitted-size reduction: `97.4%`
- aggregate compression ratio: `38.52x`
- mean ROI PSNR: `36.12 dB`
- mean ROI MS-SSIM: `0.9758`
- mean downstream detector proxy recall: `79.11%`

The written thesis remains the source for the full set of plots and tables, including LPIPS.

## ROI Generation Evaluation

The ROI stage is not evaluated as a detector benchmark with manual frame-level annotation.

Dense detection is used as a proxy reference to measure whether sparse detection plus propagation stays close enough to support downstream compression.

Reported stage metrics:

- detector calls reduced by `93.28%`
- ROI-stage speed improved by `4.41x`
- ROI-presence recall: `96.24%`
- frame-level agreement: `95.25%`

## Baseline Logic

The project compares against uniform full-frame compression baselines to test whether ROI-priority processing produces a better size-quality tradeoff for the animal region under the thesis objective.

The comparison is not defined as:

- the proposed method must beat every baseline on every metric

## Scope

This repository does not claim:

- bundled reproduction of the full original dataset
- universal superiority over every alternative on every metric
- production deployment validation on every edge device class
- perfect detector-ground-truth evaluation with manually labeled frame boxes

The reported result is limited to the evaluation setup described above.
