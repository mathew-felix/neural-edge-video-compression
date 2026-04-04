# Project Summary

## One-Sentence Description

This project implements an ROI-aware wildlife video compression pipeline that reduces transmitted size under edge-computing constraints while preserving the animal region more faithfully than the frame as a whole.

## Why It Matters

Wildlife camera-trap video often contains long stretches of static background while the animal occupies only a small, high-value region of the frame. Treating the full frame uniformly wastes transmission, storage, and compute.

This project addresses that mismatch by prioritizing the animal region throughout the pipeline instead of compressing the full frame uniformly.

## What The Pipeline Does

The system:

1. detects or propagates animal ROIs sparsely
2. places ROI and background on separate timelines
3. compresses ROI and background streams independently
4. packages a self-contained transmitted archive
5. reconstructs the final video from that archive alone

## Main Reported Results

- `97.4%` transmitted-size reduction on held-out evaluation
- `38.52x` aggregate compression ratio
- `93.28%` fewer detector calls in the ROI stage
- `4.41x` ROI-stage speedup versus dense detection
- released-pipeline mean ROI PSNR of `36.12 dB`
- released-pipeline mean ROI MS-SSIM of `0.9758`

## Technical Focus Areas

- machine learning systems
- computer vision
- edge AI
- learned video compression
- runtime and deployment tradeoffs

## Good Use Cases For This Summary

This file is meant to be easy to link from:

- LinkedIn featured items
- resume project links
- recruiter outreach
- faculty or lab outreach
- GitHub repository landing pages

## Claim Boundary

The project does not claim universal superiority over every baseline on every metric.

The defended thesis claim is narrower:

ROI-priority processing can substantially reduce transmitted size while preserving the animal region better than the frame as a whole under practical edge-oriented constraints.
