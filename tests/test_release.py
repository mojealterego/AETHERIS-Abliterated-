import json
import struct
import tempfile
import unittest
from pathlib import Path

from aetheris_abliteration.release import (
    create_release_manifest,
    verify_release_manifest,
)


class ReleaseTests(unittest.TestCase):
    def _make(self, root: Path, name: str = "m.gguf") -> Path:
        path = root / name
        path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 310, 34) + b"tensor data")
        return path

    def test_gguf_release_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = self._make(root)
            manifest = create_release_manifest(
                root, ["m.gguf"], upstream_sha="c" * 40,
                source_weights_sha256="d" * 64, license_id="apache-2.0",
            )
            self.assertEqual(manifest["models"][0]["size_bytes"], file.stat().st_size)
            self.assertEqual(manifest["status"], "RESEARCH_ONLY")
            (root / "release-manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(verify_release_manifest(root))
            file.write_bytes(file.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                verify_release_manifest(root)

    def test_invalid_magic_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._make(root, "good.gguf")
            with self.assertRaises(ValueError):
                create_release_manifest(root, ["../good.gguf"], upstream_sha="c" * 40,
                                        source_weights_sha256="d" * 64, license_id="apache-2.0")
            (root / "bad.gguf").write_bytes(b"NOTGGUF " + b"x" * 50)
            with self.assertRaises(ValueError):
                create_release_manifest(root, ["bad.gguf"], upstream_sha="c" * 40,
                                        source_weights_sha256="d" * 64, license_id="apache-2.0")

    def test_unsigned_production_deployment_is_never_approved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._make(root)
            manifest = create_release_manifest(root, ["m.gguf"], upstream_sha="c" * 40,
                                               source_weights_sha256="d" * 64, license_id="apache-2.0")
            self.assertIs(manifest["production_approved"], False)
            self.assertEqual(manifest["tenant_policy"], "EXTERNAL_FAIL_CLOSED")
