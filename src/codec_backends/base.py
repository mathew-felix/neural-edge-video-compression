from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


EncodeFramesFn = Callable[..., Dict[str, Any]]
DecodeStreamFn = Callable[..., Any]


@dataclass(frozen=True)
class CodecBackend:
    backend_id: str
    display_name: str
    codec_family: str
    integerized: bool
    cross_device_consistent: bool
    integration_status: str
    notes: str
    encode_frames: EncodeFramesFn
    decode_stream_bytes: DecodeStreamFn

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "backend": str(self.backend_id),
            "display_name": str(self.display_name),
            "codec_family": str(self.codec_family),
            "integerized": bool(self.integerized),
            "cross_device_consistent": bool(self.cross_device_consistent),
            "integration_status": str(self.integration_status),
            "notes": str(self.notes),
        }
