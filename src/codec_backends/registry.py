from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from .base import CodecBackend


DEFAULT_BACKEND_ID = "dcvc_pytorch"

_BACKEND_MODULES: Dict[str, str] = {
    "dcvc_pytorch": "codec_backends.dcvc_pytorch",
    "dcvc_rt_int16": "codec_backends.dcvc_rt_int16",
}

_BACKEND_ALIASES: Dict[str, str] = {
    "dcvc": "dcvc_pytorch",
    "dcvc-current": "dcvc_pytorch",
    "dcvc_current": "dcvc_pytorch",
    "dcvc_pytorch": "dcvc_pytorch",
    "dcvc-rt": "dcvc_rt_int16",
    "dcvc_rt": "dcvc_rt_int16",
    "dcvc_rt_int16": "dcvc_rt_int16",
}


def normalize_codec_backend_id(raw_backend: Any) -> str:
    if raw_backend is None:
        return DEFAULT_BACKEND_ID
    s = str(raw_backend).strip().lower()
    if not s:
        return DEFAULT_BACKEND_ID
    if s not in _BACKEND_ALIASES:
        raise ValueError(
            "Unknown codec backend {!r}. Valid values: {}.".format(
                raw_backend,
                ", ".join(sorted(_BACKEND_MODULES.keys())),
            )
        )
    return _BACKEND_ALIASES[s]


def is_known_codec_backend(raw_backend: Any) -> bool:
    try:
        normalize_codec_backend_id(raw_backend)
    except ValueError:
        return False
    return True


def list_codec_backend_ids() -> Tuple[str, ...]:
    return tuple(sorted(_BACKEND_MODULES.keys()))


def load_codec_backend(raw_backend: Any) -> CodecBackend:
    backend_id = normalize_codec_backend_id(raw_backend)
    module = import_module(_BACKEND_MODULES[backend_id])
    backend = getattr(module, "BACKEND", None)
    if not isinstance(backend, CodecBackend):
        raise RuntimeError(f"Codec backend module did not expose a valid BACKEND object: {_BACKEND_MODULES[backend_id]}")
    return backend
