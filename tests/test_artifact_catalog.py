import hashlib
import json
import struct

import pytest

from aetheris_abliteration.artifact_catalog import verify_catalog


def fixture(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"test weights")
    doc = {
        "schema": "aetheris.artifact-catalog.v1",
        "model_status": "RESEARCH_ONLY",
        "production_approved": False,
        "artifacts": [{
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "integrity_status": "SHA256_COMPUTED",
        }],
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(doc))
    return p, catalog, doc


def test_complete_catalog(tmp_path):
    _, catalog, _ = fixture(tmp_path)
    report = verify_catalog(catalog, tmp_path)
    assert report["verified"] is True
    assert report["artifacts"][0]["status"] == "VERIFIED"
    assert report["production_approved"] is False


@pytest.mark.parametrize("change", ["bytes", "missing", "symlink"])
def test_bad_artifact(tmp_path, change):
    p, catalog, _ = fixture(tmp_path)
    if change == "bytes":
        p.write_bytes(p.read_bytes()[:-1] + b"x")
    elif change == "missing":
        p.unlink()
    else:
        target = tmp_path / "target"
        p.rename(target)
        p.symlink_to(target)
    assert verify_catalog(catalog, tmp_path)["verified"] is False


@pytest.mark.parametrize("change", ["pending", "traversal", "duplicate", "approval", "bad_size"])
def test_bad_catalog(tmp_path, change):
    _, catalog, doc = fixture(tmp_path)
    row = doc["artifacts"][0]
    if change == "pending":
        row["sha256"] = None
        row["integrity_status"] = "FULL_SHA256_PENDING"
    elif change == "traversal":
        row["filename"] = "../model.gguf"
    elif change == "duplicate":
        doc["artifacts"].append(row.copy())
    elif change == "approval":
        doc["production_approved"] = True
    else:
        row["size_bytes"] = True
    catalog.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verify_catalog(catalog, tmp_path)


def test_trusted_catalog_hash(tmp_path):
    _, catalog, _ = fixture(tmp_path)
    expected = hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert verify_catalog(catalog, tmp_path, expected_sha256=expected)["verified"]
    with pytest.raises(ValueError):
        verify_catalog(catalog, tmp_path, expected_sha256="a" * 64)
