# Dataset And Evaluation Notes

This document explains the evaluation setup at a thesis-artifact level so external readers can understand what the reported metrics refer to and what they do not claim.

## Evaluation Purpose

The project is not trying to prove a generic video-compression theorem. The evaluation is designed to test a narrower thesis question:

can wildlife video be compressed in a way that reduces transmitted size substantially while preserving the animal region better than treating the whole frame uniformly?

That means the evaluation emphasizes:

- region-of-interest quality
- transmitted archive size
- runtime and system cost
- practical behavior under edge-oriented constraints

## Data Scope

The defended thesis reports results on unseen wildlife-video clips used as a held-out evaluation set.

From the thesis notes:

- the held-out test set contains `20` unseen clips
- the clips span varied day and night conditions
- the clips include different animals and motion patterns

This repository does not bundle those raw videos.

That is intentional because:

- the dataset may be too large for source control
- licensing or redistribution constraints may apply
- the thesis artifact focuses on code and documented results rather than shipping raw experimental media

## Local Data Convention

When running the repo locally, expected working directories are:

- `data/`
- `models/`
- `outputs/`

Example sample path used in commands:

- `data/tune/video.mp4`

This should be understood as a local placeholder path, not as a bundled public dataset.

## What Metrics Matter Most

The thesis is ROI-centered, so the most important metrics are:

- ROI PSNR
- ROI MS-SSIM
- transmitted archive size
- compression ratio
- runtime and memory cost

Full-frame metrics are still useful, but they are secondary context rather than the main objective.

The defended thesis also reports ROI LPIPS as a supporting perceptual metric. The public GitHub figure set intentionally leaves LPIPS out and emphasizes ROI PSNR, ROI MS-SSIM, size reduction, and runtime for a cleaner external presentation.

## Meaning Of The Main Metrics

### ROI PSNR

Measures numerical reconstruction fidelity on the animal region only.

Higher is better.

### ROI MS-SSIM

Measures structural similarity on the animal region.

Higher is better.

### Transmitted Archive Size

Measures what would actually be transmitted by the pipeline after compression.

This is more meaningful than talking only about internal codec outputs.

### Compression Ratio

Compares original input size to transmitted archive size.

This gives a more intuitive summary of size reduction for external readers.

## Released-Pipeline Aggregate Result

From the defended thesis notes, the released-pipeline aggregate held-out summary is:

- original test-set size: `399.16 MB`
- transmitted archive size: `10.36 MB`
- transmitted-size reduction: `97.4%`
- aggregate compression ratio: `38.52x`
- mean ROI PSNR: `36.12 dB`
- mean ROI MS-SSIM: `0.9758`
- mean downstream detector proxy recall: `79.11%`

These numbers are useful for communicating what the final released methodology achieved on the held-out set.

If deeper thesis-oriented evaluation detail is needed, the written thesis remains the source for LPIPS reporting and LPIPS-based interpretation.

## ROI Generation Evaluation

The ROI stage is not evaluated like a detector paper with manual frame-by-frame annotation.

Instead, the thesis uses dense detection as a proxy reference to test whether sparse detection plus propagation stays close enough to support downstream compression.

Reported thesis-stage highlights:

- detector calls reduced by `93.28%`
- ROI-stage speed improved by `4.41x`
- ROI-presence recall: `96.24%`
- frame-level agreement: `95.25%`

This supports the claim that sparse ROI generation is practical enough for the pipeline.

## Baseline Logic

The project compares against plain uniform full-frame DCVC baselines to answer whether ROI-priority processing helps when the animal region is the main target.

The evaluation logic is not:

- "the proposed method must beat every baseline on every metric"

The evaluation logic is:

- does ROI-priority processing produce a better size-quality tradeoff for the animal region under the thesis objective?

That claim boundary matters and should remain explicit.

## What This Repository Does Not Claim

This artifact does not claim:

- a bundled reproduction of the full original dataset
- universal superiority over every alternative on every metric
- production deployment validation on every edge device class
- perfect detector-ground-truth evaluation with manually labeled frame boxes

The thesis claim is narrower and more defensible.

## How To Use This Document

Use this file when:

- linking the project to recruiters
- explaining the project in a PhD or lab context
- clarifying that the repo is a thesis artifact and not a fully packaged public benchmark suite
