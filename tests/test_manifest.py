import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aetheris_abliteration.dataset import sha256_file
from aetheris_abliteration.manifest import (
    calculate_bundle_digest,
    create_manifest,
    verify_manifest,
)


class ManifestTests(unittest.TestCase):
    def test_bundle_hash_is_deterministic_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "model.safetensors").write_bytes(b"dummy weights")
            m = create_manifest(
                path, base_model="Qwen/Qwen3-8B", base_revision="a" * 40,
                calibration_sha256="a" * 64, layer=12,
                direction_sha256="b" * 64, metrics={"control_loss_delta": 0.1},
                edit={"edited_matrices": 40}, research_only=True,
            )
            (path / "aetheris-manifest.json").write_text(json.dumps(m), encoding="utf-8")
            self.assertTrue(verify_manifest(path))
            self.assertEqual(calculate_bundle_digest(path), m["bundle_sha256"])
            (path / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_manifest(path)

    def test_verification_hashes_each_file_once(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "model.safetensors").write_bytes(b"weights")
            m = create_manifest(path, base_model="Qwen/Qwen3-8B", base_revision="b" * 40, calibration_sha256="a" * 64, layer=1, direction_sha256="b" * 64, metrics={}, edit={}, research_only=True)
            (path / "aetheris-manifest.json").write_text(json.dumps(m), encoding="utf-8")
            with patch("aetheris_abliteration.manifest.sha256_file", wraps=sha256_file) as hashed:
                verify_manifest(path)
                self.assertEqual(hashed.call_count, 1)

    def test_manifest_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            create_manifest(
                    Path(root), base_model="Qwen/Qwen3-8B", base_revision="b" * 40,
                    calibration_sha256="a" * 64, layer=0,
                    direction_sha256="b" * 64, metrics={},
                    edit={}, research_only=True,
                )
