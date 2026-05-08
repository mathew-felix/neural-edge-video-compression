# Paper Section Checklist (DGX Encode + Laptop Decode)

## 1. Introduction

- problem statement: ROI-aware compression under edge bandwidth constraints
- one-sentence claim aligned with measured data only
- contribution bullets limited to implemented and evaluated components

## 2. Method

- describe ROI detection, frame-removal, dual-stream encoding, and reconstruction
- map algorithm narrative to implementation paths in `src/compression/` and `src/decompression/`
- include configuration controls for QP and frame-removal ablations

## 3. Experimental Setup

- dataset split and clip manifest path
- hardware disclosure:
  - DGX for compression
  - laptop for decompression
- software versions (CUDA, cuDNN, PyTorch, ffmpeg)
- baseline definitions and fairness constraints

## 4. Results

- bitrate vs ROI quality
- bitrate vs full-frame quality
- DGX encode vs laptop decode runtime breakdown
- qualitative success/failure examples

## 5. Ablations

- ROI/BG QP sweep
- frame-removal aggressiveness
- detector format variant if included in claim

## 6. Limitations

- no Jetson measurement in this submission
- discuss expected transferability to edge GPU devices as future validation

## 7. Reproducibility

- exact commands
- config hashes
- model checksums
- location of machine-readable experiment logs
