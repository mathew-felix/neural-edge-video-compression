# Jetson Setup

This guide is for Jetson devices running JetPack 6 / Python 3.10 on `aarch64`.

It uses the Jetson-specific files in this repo:

- `configs/jetson/compression.yaml`
- `configs/jetson/decompression.yaml`
- `docker/compose.jetson.yaml`
- `docker/jetson/Dockerfile.jetson`

## 1. Venv Install

Create and activate a virtual environment:

```bash
cd ~/neural-video-codec
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Install Jetson-compatible Python packages:

```bash
pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu numpy
pip install "numpy<2"
pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 torchvision==0.23.0
pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  onnxruntime-gpu==1.23.0
pip install -r docker/jetson/requirements.jetson.txt
```

Verify the install:

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected:

- `torch.__version__` should be `2.8.0`
- `torchvision.__version__` should be `0.23.0`
- `torch.cuda.is_available()` should be `True`
- `onnxruntime` should list `CUDAExecutionProvider`
- Use `opencv-python==4.11.0.86` with `numpy==1.26.4` on this setup.

If you copied only the Jetson requirements file into the repo root, use `pip install -r requirements.jetson.txt` instead.

## 2. Venv Run

Use the Jetson configs:

```bash
python run_compression.py test.mp4 --config configs/jetson/compression.yaml --output outputs/video.zip
python run_decompression.py outputs/video.zip --config configs/jetson/decompression.yaml --output outputs/video_reconstructed.mp4
```

If ONNX causes trouble on the device, keep only the `.pt` detector model and leave the `.onnx` file absent from `models/`.

## 3. Docker Install

Jetson Docker uses the Jetson-specific Dockerfile and compose file:

```bash
docker build -f docker/jetson/Dockerfile.jetson -t edge-roi-jetson .
```

If your Docker install needs NVIDIA runtime support, make sure the NVIDIA Container Toolkit is installed on the Jetson first.

## 4. Docker Run

Full pipeline:

```bash
docker compose -f docker/compose.jetson.yaml run --rm pipeline-jetson
```

Individual stages:

```bash
docker compose -f docker/compose.jetson.yaml --profile stage-tools run --rm compression-jetson
docker compose -f docker/compose.jetson.yaml --profile stage-tools run --rm decompression-jetson
```

By default the Jetson Docker wrappers:

- read `data/test.mp4`
- write `outputs/video.zip`
- write `outputs/video_reconstructed.mp4`

## 5. Docker Env Vars

You can override the paths with:

- `WILDROI_INPUT_VIDEO`
- `WILDROI_ARCHIVE_PATH`
- `WILDROI_RECON_PATH`
- `WILDROI_COMPRESSION_CONFIG`
- `WILDROI_DECOMPRESSION_CONFIG`

Example:

```bash
WILDROI_INPUT_VIDEO=data/test.mp4 \
WILDROI_ARCHIVE_PATH=outputs/video.zip \
WILDROI_RECON_PATH=outputs/video_reconstructed.mp4 \
docker compose -f docker/compose.jetson.yaml run --rm pipeline-jetson
```

## 6. Notes

- Use `configs/jetson/compression.yaml` and `configs/jetson/decompression.yaml` on the Jetson.
- The Jetson config is set up to use the `.pt` ROI detector path by default.
- If you need to debug package mismatches, start by checking `torch`, `torchvision`, and `onnxruntime-gpu` versions before looking at the code.
