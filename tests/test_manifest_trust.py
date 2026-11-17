import hashlib
import json
from pathlib import Path

import pytest

from aetheris_abliteration.manifest import create_manifest, verify_manifest
from aetheris_abliteration.release import (
    create_release_manifest,
    verify_release_manifest,
)


def make_model(root: Path) -> Path:
    (root / 'model.safetensors').write_bytes(b'weights')
    doc = create_manifest(root, base_model='Qwen/Qwen3-0.6B', base_revision='a'*40,
                          calibration_sha256='b'*64, layer=1, direction_sha256='c'*64,
                          metrics={}, edit={}, research_only=True)
    manifest = root / 'aetheris-manifest.json'
    manifest.write_text(json.dumps(doc))
    return manifest


@pytest.mark.parametrize('field,value', [('production_approved', True), ('status', 'PRODUCTION'),
                                        ('license_id', 'Unknown'), ('base_revision', '../latest')])
def test_model_manifest_rejects_bad_metadata(tmp_path, field, value):
    manifest = make_model(tmp_path)
    doc = json.loads(manifest.read_text())
    doc[field] = value
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verify_manifest(tmp_path)


def test_model_manifest_rejects_policy_engine_bypass(tmp_path):
    manifest = make_model(tmp_path)
    doc = json.loads(manifest.read_text())
    doc['integration']['auxiliary_policy_model_required'] = False
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verify_manifest(tmp_path)


def test_trusted_manifest_digest_detects_metadata_change(tmp_path):
    manifest = make_model(tmp_path)
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify_manifest(tmp_path, expected_manifest_sha256=expected)
    doc = json.loads(manifest.read_text())
    doc['base_model'] = 'Other/base'
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verify_manifest(tmp_path, expected_manifest_sha256=expected)


def test_release_manifest_rejects_invalid_source_revision(tmp_path):
    import struct
    (tmp_path/'test.gguf').write_bytes(struct.pack('<4sIQQ', b'GGUF', 3, 2, 3) + b'data')
    doc = create_release_manifest(tmp_path, ['test.gguf'], upstream_sha='a'*40,
                                  source_weights_sha256='b'*64, license_id='Apache-2.0')
    doc['original_upstream_revision'] = 'main'
    (tmp_path/'release-manifest.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verify_release_manifest(tmp_path)
