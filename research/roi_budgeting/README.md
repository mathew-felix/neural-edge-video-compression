# ROI Budgeting Research Workspace

This folder is an isolated sandbox for experiments on budget-aware ROI anchor selection.

The current local budgeting path uses actual ROI temporal probe bytes by default rather than the earlier fixed keep-count proxy. It runs a masked-frame `x264` probe over ROI segments, spends anchors under the configured `roi_target_kbps`, and still allows the older calibrated proxy model as a fallback research option.

The current leading policy is a guarded hybrid `v4`:

- raw segment-aware ROI budget DP when that helps
- fallback to the fixed baseline on cheap-safe clips
- fallback to `v3` when `v3` is measurably better than raw `v4`

The production pipeline remains untouched:

- no edits to the repo root `src/`
- no edits to `run_compression.py`
- no edits to `run_decompression.py`
- no writes to the normal `outputs/` tree unless we explicitly choose to compare against an existing run

## Goal

Design and evaluate a budget-aware ROI keep policy that uses:

- motion
- ROI uncertainty
- AMT reconstruction risk

Background selection stays on the current fixed policy for now.

## Current Status

The current 12-clip benchmark result is:

- `fixed_baseline`
  - mean primary delta: `0.000000`
  - mean estimated ROI bitrate: `96.343639 kbps`
- `v3_motion_uncertainty_amt`
  - mean primary delta: `0.147374`
  - mean estimated ROI bitrate: `126.354324 kbps`
- `v4_segment_dp_amt`
  - mean primary delta: `0.176478`
  - mean estimated ROI bitrate: `107.515719 kbps`

The current aggregate report is:

- `results/benchmark/_aggregate/experiment_aggregate.md`

The current milestone note is:

- `docs/milestone_v4_guarded.md`

## Layout

- `configs/`: local, Colab, and experiment YAMLs
- `docs/`: design notes and research decisions
- `notebooks/`: local analysis notebooks
- `colab/`: Colab notebook entrypoint
- `roi_budgeting/`: experiment package
- `results/`: generated manifests, tables, and plots

## Guardrails

- Treat the current pipeline as the baseline and source of artifacts.
- Prefer reading existing outputs such as `roi_detections.json`, `frame_drop.json`, and archives.
- Keep heavyweight experiments in this subtree.
- Do not import the repo's top-level `src` package under the same name here. This workspace uses the package name `roi_budgeting` to avoid collisions.

## Suggested Workflow

1. Generate baseline artifacts with the existing scripts when needed.
2. Inspect the data in `notebooks/01_data_audit.ipynb`.
3. Measure motion, uncertainty, and AMT probe error.
4. Evaluate alternative ROI schedules offline.
5. Compare against the current fixed heuristic before any production integration.

## Quick Start

```bash
cd research/roi_budgeting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m roi_budgeting.runners.run_offline_eval --config configs/local.yaml
```

## Local End-To-End Run

For a full local baseline plus `v1`/`v2`/`v3` sweep on the research clip:

```bash
cd research/roi_budgeting
source .venv/bin/activate
python -m roi_budgeting.runners.run_local_pipeline
```

On non-CUDA machines, the research workspace can now run full AMT on CPU. Use `--strict-amt` when you want to require the full path and fail instead of falling back.

To generate a compact experiment comparison table and plot from the saved manifests:

```bash
cd research/roi_budgeting
source .venv/bin/activate
python -m roi_budgeting.runners.run_experiment_report --config configs/local.yaml
```

## Batch Benchmark

To run the benchmark folder under `test/`:

```bash
cd research/roi_budgeting
source .venv/bin/activate
python -m roi_budgeting.runners.run_batch_benchmark --skip-v1 --skip-existing --continue-on-error
```

This writes per-clip manifests plus aggregate tables under:

- `results/benchmark/`
- `results/benchmark/_aggregate/`

The batch runner is:

- `roi_budgeting/runners/run_batch_benchmark.py`

## Colab

Use `colab/roi_budgeting_experiments.ipynb` as the Colab entry notebook. Keep notebooks and configs versioned here, and keep large generated outputs under `results/` or your mounted Drive location.

The notebook now runs the full research loop on one clip:

- installs the research dependencies in Colab
- generates baseline `roi_detections.json` and `frame_drop.json`
- writes a runtime `configs/colab_runtime.yaml`
- generates a reusable AMT probe manifest
- runs `v1`, `v2`, and `v3`
- writes manifests into `research/roi_budgeting/results/`
- optionally syncs that `results/` tree back to Google Drive

If you want a one-command Colab run instead of stepping through notebook cells, use:

```bash
cd "/content/drive/MyDrive/Colab Notebooks/neural-edge-video-compression/research/roi_budgeting"
python -m roi_budgeting.runners.run_colab_pipeline \
  --repo-root "/content/drive/MyDrive/Colab Notebooks/neural-edge-video-compression" \
  --video "/content/drive/MyDrive/Colab Notebooks/neural-edge-video-compression/video.mp4"
```

## GPU AMT Probes

When a CUDA environment is available, generate a reusable AMT probe manifest first:

```bash
cd research/roi_budgeting
python -m roi_budgeting.runners.run_amt_probes --config configs/colab_runtime.yaml
```

Then run the v3 experiment, which will consume `paths.amt_probe_manifest` when present:

```bash
python -m roi_budgeting.runners.run_offline_eval \
  --config configs/colab_runtime.yaml \
  --experiment-config configs/experiments/v3_motion_uncertainty_amt.yaml
```

On non-CUDA machines, the evaluator falls back to a clearly labeled local proxy instead of full AMT inference.
