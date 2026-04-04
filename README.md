# Neural ROI-Aware Video Compression for Wildlife Monitoring Under Edge Constraints

This repository contains the code artifact for a master's-thesis pipeline that compresses wildlife video by prioritizing the animal region of interest (ROI) over the background. The system combines sparse ROI generation, dual-timeline frame selection, learned compression, and archive-only reconstruction.

The intended audience is:

- thesis committee members
- recruiters or hiring managers reviewing the project
- researchers who want to understand the system layout
- future lab members or collaborators extending the pipeline

This repository should read as a thesis artifact, not as an internal scratch repo.

## What The System Does

At a high level, the pipeline:

1. detects or propagates animal ROIs sparsely instead of running dense detection on every frame
2. keeps ROI and background on separate timelines
3. compresses ROI and background streams independently
4. packages the transmitted result into a self-contained archive
5. reconstructs the video on the server from that archive alone

Core entry points:

- `run_compression.py`
- `run_decompression.py`

## Thesis-Facing Result Snapshot

These numbers summarize the defended thesis result and help readers quickly understand why the project matters:

- aggregate transmitted-size reduction on held-out evaluation: `97.4%`
- aggregate compression ratio on held-out evaluation: `38.52x`
- detector-call reduction in the sparse ROI stage: `93.28%`
- ROI-stage speedup versus dense detection: `4.41x`
- released-pipeline mean ROI PSNR: `36.12 dB`
- released-pipeline mean ROI MS-SSIM: `0.9758`

These are thesis-level artifact numbers, not generic promises for every possible deployment setting.

See [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) for a cleaner thesis-facing summary.
See [docs/project_summary.md](docs/project_summary.md) for a shorter recruiter-friendly version.

## Reproducibility

This repo has been validated as a runnable code artifact on a real Windows GPU setup, not just as a static code dump.

The strongest reproducibility anchors are:

- locked package versions in [docker/requirements.gpu.txt](docker/requirements.gpu.txt)
- pinned `pip`, `torch`, and `torchvision` versions in the bootstrap paths
- SHA256-verified model downloads via [docs/model_checksums.sha256](docs/model_checksums.sha256)
- a recorded smoke-test validation note in [docs/reproducibility_validation.md](docs/reproducibility_validation.md)

Validated environment snapshot:

- Windows `10.0.26200.8037`
- Python `3.12.0`
- PyTorch `2.10.0+cu126`
- `NVIDIA GeForce RTX 3070 Ti Laptop GPU`

## GitHub Figure Set

The public figure set in [docs/figures/](docs/figures/) is aligned to the defended thesis, but it intentionally avoids LPIPS-based charts. For GitHub, the emphasis is:

- pipeline structure
- detector savings and ROI continuity
- runtime bottlenecks
- qualitative day and night examples
- size and quality discussion in text using ROI PSNR and ROI MS-SSIM

![GitHub summary figure](docs/figures/08_github_summary.png)

See [docs/figures/README.md](docs/figures/README.md) for the exact figure list and thesis-to-repo mapping.

## Repository Layout

Top-level structure:

- `configs/`
  runtime configuration profiles
- `src/`
  project source code organized by pipeline stage
- `scripts/`
  stage-wise sanity checks and model download helpers
- `tests/`
  test suite for configuration, runtime contracts, and stage logic
- `docker/`
  GPU-focused container setup
- `DCVC/`
  vendored third-party compression dependency
- `_third_party_amt/`
  vendored third-party interpolation dependency
- `run_compression.py`
  main compression entry point
- `run_decompression.py`
  main decompression entry point

See [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) for a more explicit repo map and what each area should contain.
See [docs/dataset_and_eval.md](docs/dataset_and_eval.md) for thesis-facing evaluation context and metric interpretation.

## What Is Not Stored In Git

The repo intentionally does not ship large runtime assets or evaluation data:

- dataset videos
- downloaded model weights
- generated archives
- reconstructed videos
- experiment outputs

The expected local working directories are:

- `data/`
- `models/`
- `outputs/`

These are gitignored.

## Required Model Files

Place these files in `models/`:

- `MDV6-yolov9-c.pt`
- `cvpr2025_image.pth.tar`
- `cvpr2025_video.pth.tar`
- `amt-s.pth` if AMT interpolation is enabled

Optional:

- `MDV6-yolov9-c.onnx`

Model bootstrap helper:

```bash
python scripts/download_models.py
```

The download script verifies every model against the SHA256 manifest in [docs/model_checksums.sha256](docs/model_checksums.sha256).

## Quick Start

### 1. Create local working directories

```bash
mkdir -p data models outputs
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data, models, outputs
```

### 2. Set up the environment

Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip==26.0.1 setuptools==82.0.0 wheel==0.46.3
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
pip install -r docker/requirements.gpu.txt
cd DCVC/src/cpp
pip install --no-build-isolation .
cd ../layers/extensions/inference
pip install --no-build-isolation .
cd ../../../..
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.0.1 setuptools==82.0.0 wheel==0.46.3
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
pip install -r docker\requirements.gpu.txt
cd DCVC\src\cpp
pip install --no-build-isolation .
cd ..\layers\extensions\inference
pip install --no-build-isolation .
cd ..\..\..\..
```

### 3. Download models

```bash
python scripts/download_models.py
```

### 4. Run compression

```bash
python run_compression.py data/tune/video.mp4 --config configs/gpu/compression.yaml --output outputs/video.zip
```

### 5. Run decompression

```bash
python run_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --output outputs/video_reconstructed.mp4
```

## Stage-Wise Sanity Checks

These scripts are for development sanity checks, not for presenting the entire thesis by themselves.

### ROI detection

```bash
python scripts/test_roi_detection.py --config configs/gpu/compression.yaml --video data/tune/video.mp4
```

Outputs:

- `outputs/sanity_checks/roi_detection/roi_detections.json`
- `outputs/sanity_checks/roi_detection/roi_overlay.mp4`

### Frame removal

```bash
python scripts/test_frame_removal.py --config configs/gpu/compression.yaml --video data/tune/video.mp4
```

Outputs:

- `outputs/sanity_checks/frame_removal/frame_drop.json`
- `outputs/sanity_checks/frame_removal/frame_drop_overlay.mp4`
- `outputs/sanity_checks/frame_removal/roi_kept_preview.mp4`
- `outputs/sanity_checks/frame_removal/bg_kept_preview.mp4`

### Compression sanity

```bash
python scripts/test_compression.py --config configs/gpu/compression.yaml --video data/tune/video.mp4 --repeat 2
```

### Decompression sanity

```bash
python scripts/test_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --repeat 1
```

See [scripts/README.md](scripts/README.md) for the sanity-check workflow.

## Docker

Build:

```bash
docker build -f docker/Dockerfile.gpu -t edge-roi-gpu .
```

Run the default composed pipeline:

```bash
docker compose -f docker/compose.gpu.yaml run --rm pipeline-gpu
```

This uses the sample paths wired in the compose file:

- input video: `data/tune/video.mp4`
- output archive: `outputs/video.zip`
- reconstructed video: `outputs/video_reconstructed.mp4`

## Reproducibility Notes

This repository is best understood as a defended thesis code artifact with runnable pipeline entry points and sanity checks. It is not yet a perfect one-command public benchmark package.

That means:

- large datasets are not bundled
- model weights are not committed
- exact experiment splits and released result tables should be documented separately when preparing a public artifact release
- readers should not confuse the dev sanity scripts with the full thesis evaluation protocol

Concrete reproducibility anchors in this repo:

- locked dependency file: [docker/requirements.gpu.txt](docker/requirements.gpu.txt)
- model checksum manifest: [docs/model_checksums.sha256](docs/model_checksums.sha256)
- validated smoke-test note: [docs/reproducibility_validation.md](docs/reproducibility_validation.md)

## Limitations And Scope

This repository does not claim:

- universal superiority over every baseline on every metric
- CPU-friendly runtime
- production-ready deployment on all edge hardware
- bundled reproduction of the entire thesis dataset

The defended thesis claim is narrower:

an ROI-priority wildlife video pipeline can reduce transmitted size substantially while preserving the animal region better than treating the whole frame uniformly, under practical edge-oriented constraints.

## Recommended Next Public-Artifact Improvements

If you are polishing this repo for thesis, portfolio, or PhD use, the next highest-value additions are:

1. a clean thesis summary PDF linked from the repo
2. a short thesis artifact note that explains the held-out test split and locked released configuration
3. a release tag with fixed model artifacts
4. a citation file once the final thesis bibliographic metadata is locked
