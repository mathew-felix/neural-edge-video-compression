# Reproducibility Validation

This file records the environment and hashes from a successful Windows smoke test of the repository.

## Validated Environment

- date: `2026-04-04`
- operating system: `Microsoft Windows 10.0.26200.8037`
- python: `3.12.0`
- pip: `26.0.1`
- torch: `2.10.0+cu126`
- torchvision: `0.25.0+cu126`
- GPU: `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
- ffmpeg: `8.0-full_build-www.gyan.dev`

Dependency file:

- [docker/requirements.gpu.txt](../docker/requirements.gpu.txt)

Model checksum file:

- [docs/model_checksums.sha256](model_checksums.sha256)

## Validated Procedure

The validation run executed the following command sequence:

```cmd
python -m venv venv
python -m pip install --upgrade pip==26.0.1 setuptools==82.0.0 wheel==0.46.3
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
python -m pip install -r docker\requirements.gpu.txt
python scripts\download_models.py
python scripts\test_roi_detection.py --config configs\gpu\compression.yaml --video test_video.mp4 --out-dir outputs\windows_smoke\roi_detection --overlay-max-frames 120
python scripts\test_frame_removal.py --config configs\gpu\compression.yaml --video test_video.mp4 --out-dir outputs\windows_smoke\frame_removal --debug-max-frames 120
python run_compression.py test_video.mp4 --config configs\gpu\compression.yaml --output outputs\windows_smoke\test_video.zip
python scripts\test_decompression.py outputs\windows_smoke\test_video.zip --config configs\gpu\decompression.yaml --out-dir outputs\windows_smoke\decompression --repeat 1 --max-frames 120
```

## Smoke-Test Coverage

- virtual environment creation
- pinned Python dependency installation
- DCVC C++ extension build
- DCVC CUDA inference extension build
- CUDA availability check
- model download and checksum verification
- ROI detection sanity run
- frame-removal sanity run
- end-to-end compression
- decompression reproducibility check

## Recorded Inputs And Outputs

Input video:

- path: `test_video.mp4`
- SHA256: `d89d6f2fc4cd42450cf591535a69cebac267b970a90ed33d1ac9b2fa21adf82c`

ROI detection output:

- `frames_with_roi`: `405`
- `total_boxes`: `405`
- `keyframe_interval`: `15`
- `estimated_detector_calls`: `54`
- `propagation_enabled`: `true`
- `roi_json_sha256`: `78a4c1c7b31709f1746941273cb0f266a72c5b6f4a67b9f04b614fbc7e6dbdba`

Frame-removal output:

- `kept_frames`: `201`
- `dropped_frames`: `595`
- `drop_ratio`: `0.7474874371859297`
- `bg_kept_frames`: `110`
- `bg_dropped_frames`: `686`
- `frame_drop_sha256`: `678688e84783940523bad83724a25555f62b9206191b8fa40a56e6c4f41613cf`

Compression output:

- archive path: `outputs/windows_smoke/test_video.zip`
- archive size: `328586` bytes
- archive SHA256: `545d4ef3e0ae1c898761704b379a71231d96b60dea2038ef54bd01d041e90d63`

Decompression output:

- reconstructed MP4 SHA256: `f3c542db4570651da201cf08e5920bd9cde828ddbb03770683680afe8d0bab41`
- reconstructed pixel SHA256: `7a6681158eff9113327635add59ad6784aca163a7e340fef5c0b842aaee0719c`
- lossless AVI SHA256: `231fb94a138fffe807659e4be7d68ad15a07b5b48046db0737b3f8ac9ef15e67`
- expected frames checked: `120`
- output geometry: `1280x720 @ 30.0 fps`
- reproducibility result: `PASS`

## Scope

This validation supports the following statements:

- the repository setup works from scratch on the validated machine
- the main pipeline entry points execute successfully
- the decompression path is deterministic on the smoke-test sample

This validation does not prove:

- reproduction of the full 20-video thesis benchmark
- cross-machine reproducibility on every GPU, driver, or operating system variant
- long-term immunity to package or model-host changes beyond the locked versions and checksums recorded here
