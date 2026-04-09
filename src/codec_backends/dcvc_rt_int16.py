from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .base import CodecBackend


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RT_REPO = ROOT / "DCVC"
DEFAULT_FORCE_ZERO_THRES = 0.12


def _default_repo_dir() -> str:
    return str(DEFAULT_RT_REPO.resolve())


def _prepare_dcvc_cfg(dcvc_cfg):
    out = dict(dcvc_cfg or {})
    repo_dir = str(out.get("repo_dir", "") or "").strip()
    if not repo_dir or repo_dir == "DCVC":
        out["repo_dir"] = _default_repo_dir()
    if out.get("force_zero_thres", None) is None:
        out["force_zero_thres"] = float(DEFAULT_FORCE_ZERO_THRES)
    return out


def _prepare_runtime_cfg(cfg):
    out = deepcopy(cfg)
    out["dcvc"] = _prepare_dcvc_cfg((out.get("dcvc", {}) or {}))
    return out


def _encode_frames(*args, **kwargs):
    from compression.dcvc_encoder import encode_dcvc_frames_to_bytes

    call_kwargs = dict(kwargs)
    call_kwargs["cfg"] = _prepare_runtime_cfg(call_kwargs.get("cfg", {}))
    return encode_dcvc_frames_to_bytes(*args, **call_kwargs)


def _decode_stream_bytes(*args, **kwargs):
    from decompression.roi_bg_decompress import _decode_stream_bytes_dcvc

    call_args = list(args)
    if len(call_args) < 2:
        raise TypeError("dcvc_rt_int16 decode backend expects stream bytes and dcvc_cfg arguments.")
    call_args[1] = _prepare_dcvc_cfg(call_args[1])
    return _decode_stream_bytes_dcvc(*call_args, **kwargs)


BACKEND = CodecBackend(
    backend_id="dcvc_rt_int16",
    display_name="DCVC-RT integerized",
    codec_family="dcvc_rt",
    integerized=True,
    cross_device_consistent=True,
    integration_status="implemented",
    notes=(
        "Official DCVC-RT code path with deterministic upstream defaults. "
        "Uses DCVC by default and defaults force_zero_thres to 0.12."
    ),
    encode_frames=_encode_frames,
    decode_stream_bytes=_decode_stream_bytes,
)
