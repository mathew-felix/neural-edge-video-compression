# Sanity-Check Scripts

These scripts are development and artifact-validation helpers. They exist to verify that major stages of the pipeline are behaving sensibly before running larger experiments or showing the repo to others.

They should not be treated as a full replacement for the defended thesis evaluation protocol.

## Model Bootstrap

Download all required models:

```bash
python scripts/download_models.py
```

Compression-only bootstrap:

```bash
python scripts/download_compression_models.py --repo owner/repo --tag artifact-v1
```

Decompression-only bootstrap:

```bash
python scripts/download_decompression_models.py --repo owner/repo --tag artifact-v1
```

Notes:

- scripts save model files into `models/` by default
- `download_models.py` verifies SHA256 checksums against `docs/model_checksums.sha256`
- if the GitHub repo is private, set `GITHUB_TOKEN`
- for a public artifact, use a fixed release tag instead of `latest`

## 1. ROI Detection Sanity

```bash
python scripts/test_roi_detection.py --video data/tune/video.mp4
```

Inspect:

- `outputs/sanity_checks/roi_detection/roi_overlay.mp4`
- whether animal boxes look plausible and temporally stable

## 2. Frame-Removal Sanity

```bash
python scripts/test_frame_removal.py --video data/tune/video.mp4
```

Inspect:

- `outputs/sanity_checks/frame_removal/frame_drop_overlay.mp4`
- `outputs/sanity_checks/frame_removal/roi_kept_preview.mp4`
- `outputs/sanity_checks/frame_removal/bg_kept_preview.mp4`
- whether dropped frames look redundant relative to retained anchors

## 3. Compression Sanity And Reproducibility

```bash
python scripts/test_compression.py --video data/tune/video.mp4 --repeat 2
```

Checks:

- archive is smaller than source video
- repeated runs with the same inputs produce stable outputs and summaries

## 4. Decompression Sanity

```bash
python scripts/test_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml
```

Checks:

- decompression completes successfully
- reconstructed video is readable and non-empty
- output size and frame count match archive metadata expectations

## 5. Full Visual End-To-End Check

```bash
python run_compression.py data/tune/video.mp4 --config configs/gpu/compression.yaml --output outputs/video.zip
python run_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --output outputs/sanity_checks/reconstructed/video.mp4
```

Inspect:

- reconstructed visual quality
- ROI coherence
- major temporal artifacts or reconstruction failures

## Recommended Protocol

Before running larger experiments:

1. verify model downloads
2. run ROI detection sanity
3. run frame-removal sanity
4. run compression sanity
5. run decompression sanity
6. inspect a small set of day and night clips visually

## Path Conventions

Run from the repository root.

Expected local directories:

- `data/`
- `models/`
- `outputs/`

The scripts write debug outputs under `outputs/sanity_checks/`.
