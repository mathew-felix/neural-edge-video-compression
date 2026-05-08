# Reproducibility Notes (DGX Encode + Laptop Decode)

This document defines the required environment records for conference experiments.

## DGX Record

Store in `outputs/paper_runs/env_dgx.json`:

- GPU model(s)
- CUDA version
- cuDNN version
- Python version
- PyTorch and torchvision versions
- ffmpeg version
- OS and driver versions

## Laptop Record

Store in `outputs/paper_runs/env_laptop.json`:

- CPU model
- GPU model (if any)
- Python version
- ffmpeg version
- OS version

## Command Repro Path

1. Run DGX compression matrix:

```bash
python scripts/run_conference_matrix.py --host-role dgx --video data/test.mp4 --clip-id clip01 --run-label run1
```

2. Transfer archive(s) to laptop.

3. Run laptop decode logging:

```bash
python scripts/run_conference_matrix.py --host-role laptop --video data/test.mp4 --archive outputs/paper_runs/clip01_roi_aware_main_default.zip --clip-id clip01 --run-label run1
```

4. Fill quality metrics and generate figures:

```bash
python scripts/plot_conference_results.py --results-csv outputs/paper_runs/results.csv --fig-dir outputs/paper_runs/figures
```

## Required Output Artifacts

- `outputs/paper_runs/results.jsonl`
- `outputs/paper_runs/results.csv`
- `outputs/paper_runs/figures/*.png`
- frozen clip manifest and matrix JSON used for the paper
