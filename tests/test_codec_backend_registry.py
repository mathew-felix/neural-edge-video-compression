from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codec_backends import load_codec_backend, normalize_codec_backend_id  # noqa: E402


class TestCodecBackendRegistry(unittest.TestCase):
    def test_default_backend_is_current_pytorch_wrapper(self) -> None:
        backend = load_codec_backend(None)

        self.assertEqual(backend.backend_id, "dcvc_pytorch")
        self.assertFalse(backend.integerized)
        self.assertEqual(backend.integration_status, "implemented")

    def test_dcvc_rt_alias_resolves_to_integerized_backend(self) -> None:
        self.assertEqual(normalize_codec_backend_id("dcvc_rt"), "dcvc_rt_int16")
        backend = load_codec_backend("dcvc-rt")

        self.assertEqual(backend.backend_id, "dcvc_rt_int16")
        self.assertTrue(backend.integerized)
        self.assertEqual(backend.integration_status, "implemented")


if __name__ == "__main__":
    unittest.main()
