# Conference Experiment Protocol (DGX Encode + Laptop Decode)

## Scope

This protocol freezes the evaluation flow for the conference submission:

- compression runs on DGX GPU using `run_compression.py`
- archive transfer is treated as a separate step
- decompression runs on laptop using `run_decompression.py`

No Jetson latency or throughput claims are included in this protocol version.

## Primary Claim

ROI-aware dual-stream compression preserves ROI quality better than uniform compression at comparable bitrate, while maintaining practical decode runtime on commodity client hardware.

## Fixed Runtime Paths

- Compression entrypoint: `run_compression.py`
- Decompression entrypoint: `run_decompression.py`
- Compression config: `configs/gpu/compression.yaml`
- Decompression config: `configs/gpu/decompression.yaml`

## Dataset Freeze Template

Create and freeze a clip manifest before experiments:

- file: `outputs/paper_runs/clip_manifest.json`
- include:
  - `clip_id`
  - `video_path`
  - `split` (`val` or `test`)
  - `notes` (day/night, motion profile)

Do not change the manifest after first baseline run.

## Metrics Contract

Per-run required metrics:

- `bitrate_bps`
- `archive_size_bytes`
- `encode_time_sec_dgx`
- `decode_time_sec_laptop`
- `roi_psnr`
- `roi_ms_ssim`
- `full_psnr`
- `full_ms_ssim`

Optional:

- `transfer_time_sec`
- `lpips_roi`
- `lpips_full`

## Methods To Run

- `uniform_ffmpeg` (single-stream baseline)
- `roi_unaware_control` (dual-stream control with ROI advantage disabled)
- `roi_aware_main` (proposed method)

## Codec Selection Pre-Step (Before Full Matrix)

Run ROI/BG codec pair benchmark on `data/test_videos`:

```bash
python scripts/benchmark_codec_pairs.py --video-dir data/test_videos --max-videos 6 --out-dir outputs/codec_benchmark
```

Use `outputs/codec_benchmark/codec_pair_results.csv` to choose one ROI codec profile and one BG codec profile based on:

- lower encode/decode time
- lower archive size
- acceptable visual quality on sampled clips

If codec-pair benchmark is skipped due to time, freeze one practical profile and run DGX encode with resume-safe logging:

```bash
python scripts/run_dgx_fixed_codec_encode.py \
  --video-dir data/test_videos \
  --max-videos 6 \
  --out-dir outputs/paper_runs \
  --roi-codec av1 --roi-encoder libsvtav1 --roi-preset fast --roi-qp 28 \
  --bg-codec hevc --bg-encoder libx265 --bg-preset fast --bg-qp 36
```

Re-run the same command to resume remaining clips. Use `--no-resume` for a fresh run.

## Minimal Ablation Set

- ROI/BG QP sweep (`small`, fixed grid)
- frame-removal aggressiveness (`low`, `medium`, `high`)
- detector format variant (`pt` vs `onnx`) only if used in paper claim

## Reproducibility Requirements

- record exact software versions for DGX and laptop
- record config hash for every run
- keep machine-readable results in:
  - `outputs/paper_runs/results.jsonl`
  - `outputs/paper_runs/results.csv`
- keep run manifests and command logs under `outputs/paper_runs/`

## Reporting Rules

- report encode and decode runtimes separately
- if transfer time is reported, keep it separate from codec runtime
- do not present DGX numbers as embedded edge-device numbers
- include explicit limitation: no Jetson benchmark in this submission
