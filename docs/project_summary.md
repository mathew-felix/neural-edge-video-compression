# Project Summary

This project implements an ROI-aware wildlife video compression pipeline that reduces transmitted size under edge-computing constraints while preserving the animal region more faithfully than the frame as a whole.

## Motivation

Wildlife camera-trap video often contains long stretches of static background while the animal occupies only a small part of the frame. Treating the entire frame uniformly spends bitrate and compute on low-value background regions.

## Pipeline

The system:

1. detects or propagates animal ROIs sparsely
2. places ROI and background on separate timelines
3. compresses ROI and background streams independently
4. packages the transmitted result into a single archive
5. reconstructs the final video from that archive

## Reported Results

- `97.4%` transmitted-size reduction on held-out evaluation
- `38.52x` aggregate compression ratio
- `93.28%` fewer detector calls in the ROI stage
- `4.41x` ROI-stage speedup versus dense detection
- mean ROI PSNR: `36.12 dB`
- mean ROI MS-SSIM: `0.9758`

## Technical Areas

- machine learning systems
- computer vision
- edge AI
- learned video compression
- runtime and deployment tradeoffs

## Scope

The project does not claim universal superiority over every baseline on every metric.

The reported result is narrower:

ROI-priority processing can substantially reduce transmitted size while preserving the animal region better than the frame as a whole under practical edge-oriented constraints.
