"""Authenticated-by-hash GGUF release manifests for research-only model artifacts."""
from __future__ import annotations

import json
import re
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import sha256_file

_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _gguf_file(root: Path, name: str) -> dict[str, Any]:
    if not isinstance(name, str) or not name or name != Path(name).name or "/" in name or "\\" in name:
        raise ValueError("Only local artifact basenames are permitted")
    if not name.endswith(".gguf") or name.startswith("."):
        raise ValueError("Release files must be ordinary .gguf files")
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing, nonregular, or symlink artifact: {name}")
    size = path.stat().st_size
    if size <= 24:
        raise ValueError(f"Artifact too small: {name}")
    with path.open("rb") as stream:
        header = stream.read(24)
    magic, version, tensors, metadata = struct.unpack("<4sIQQ", header)
    if magic != b"GGUF" or version != 3 or not tensors or not metadata:
        raise ValueError(f"Unsupported/corrupt GGUF header: {name}")
    return {
        "filename": name,
        "size_bytes": size,
        "sha256": sha256_file(path),
        "gguf_version": version,
        "tensor_count": tensors,
        "metadata_count": metadata,
    }


def create_release_manifest(
    root: Path, names: list[str], *,
    upstream_sha: str, source_weights_sha256: str, license_id: str,
) -> dict[str, Any]:
    if not _HEX40.fullmatch(upstream_sha):
        raise ValueError("Upstream revision must be a full 40-character Git commit")
    if not _HEX64.fullmatch(source_weights_sha256):
        raise ValueError("Source weights digest must be SHA256")
    if license_id.lower() != "apache-2.0":
        raise ValueError("Unapproved model license")
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected at least one unique artifact")
    models = [_gguf_file(root, name) for name in sorted(names)]
    return {
        "schema": "aetheris.release-manifest.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": "Qwen3",
        "original_upstream_revision": upstream_sha.lower(),
        "source_weights_sha256": source_weights_sha256.lower(),
        "license_id": "Apache-2.0",
        "status": "RESEARCH_ONLY",
        "production_approved": False,
        "tenant_policy": "EXTERNAL_FAIL_CLOSED",
        "models": models,
        "model_note": "Quantization integrity is not proof of NSFW ability or refusal removal.",
    }


def verify_release_manifest(root: Path) -> bool:
    path = root / "release-manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Release manifest missing or is a symlink")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "aetheris.release-manifest.v1":
        raise ValueError("Unexpected release manifest schema")
    if doc.get("status") != "RESEARCH_ONLY" or doc.get("production_approved") is not False:
        raise ValueError("Research release must not be marked production approved")
    if doc.get("tenant_policy") != "EXTERNAL_FAIL_CLOSED":
        raise ValueError("Tenant policy is not fail-closed")
    if doc.get("license_id") != "Apache-2.0":
        raise ValueError("License mismatch")
    names = doc.get("models")
    if not isinstance(names, list) or not names:
        raise ValueError("No release artifacts")
    seen: set[str] = set()
    for item in names:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise TypeError("Malformed artifact record")
        if item["filename"] in seen:
            raise ValueError("Duplicate release artifact")
        seen.add(item["filename"])
        current = _gguf_file(root, item["filename"])
        if current != item:
            raise ValueError(f"Release artifact hash or header mismatch: {item['filename']}")
    return True
