from __future__ import annotations

from .base import CodecBackend


def _encode_frames(*args, **kwargs):
    from compression.dcvc_encoder import encode_dcvc_frames_to_bytes

    return encode_dcvc_frames_to_bytes(*args, **kwargs)


def _decode_stream_bytes(*args, **kwargs):
    from decompression.roi_bg_decompress import _decode_stream_bytes_dcvc

    return _decode_stream_bytes_dcvc(*args, **kwargs)


BACKEND = CodecBackend(
    backend_id="dcvc_pytorch",
    display_name="DCVC PyTorch/CUDA",
    codec_family="dcvc",
    integerized=False,
    cross_device_consistent=False,
    integration_status="implemented",
    notes="Legacy project wrapper using the configured DCVC repo and thresholds as-is.",
    encode_frames=_encode_frames,
    decode_stream_bytes=_decode_stream_bytes,
)
