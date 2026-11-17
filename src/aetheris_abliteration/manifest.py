"""Content-addressed model bundle manifest for AETHERIS / Chronos integration."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import sha256_file


def _artifact_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError("Model directory does not exist")
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name != "aetheris-manifest.json"),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlinks are not allowed in model bundles")
    return files


def _file_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in _artifact_files(root)}


def _digest_file_hashes(files: dict[str, str]) -> str:
    h = hashlib.sha256(b"AETHERIS_MODEL_BUNDLE_V1\0")
    for name, digest in files.items():
        encoded = name.encode("utf-8")
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


def calculate_bundle_digest(root: Path) -> str:
    """Domain-separated hash of sorted relative paths and per-file SHA256 digests."""
    return _digest_file_hashes(_file_hashes(root))


def create_manifest(
    root: Path, *,
    base_model: str, base_revision: str, calibration_sha256: str,
    layer: int, direction_sha256: str, metrics: dict[str, Any],
    edit: dict[str, Any], research_only: bool,
) -> dict[str, Any]:
    files = _file_hashes(root)
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("No safetensors weight files in output directory")
    return {
        "schema": "aetheris.model-manifest.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": "aetheris-qwen3-adult-abliterated",
        "family": "Qwen3",
        "base_model": base_model,
        "base_revision": base_revision,
        "architecture": "Qwen3ForCausalLM",
        "modification": "rank-one residual output orthogonalization",
        "calibration_sha256": calibration_sha256,
        "selected_layer": layer,
        "direction_sha256": direction_sha256,
        "edit": edit,
        "metrics": metrics,
        "license_id": "Apache-2.0",
        "status": "RESEARCH_ONLY" if research_only else "CANDIDATE_UNVERIFIED",
        "production_approved": False,
        "policy_scope": "consensual_adult_creative_text_only",
        "provenance": {"immutable_source": True, "local_processing": True},
        "files_sha256": files,
        "bundle_sha256": _digest_file_hashes(files),
        "integration": {
            "omni_graph_type": "PromptSpec",
            "primary_gguf_role": "PRIMARY_GGUF",
            "auxiliary_policy_model_required": True,
            "identity_and_consent_controls": "external_fail_closed_policy_engine",
        },
    }


def verify_manifest(root: Path, *, expected_manifest_sha256: str | None = None) -> bool:
    """Validate bundle bytes and fail-closed metadata.

    An independently trusted digest pins manifest bytes; without it this
    verifies consistency and permitted policy values, not publisher identity.
    """
    manifest_path = root / "aetheris-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Missing or invalid manifest")
    if manifest_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Oversized model manifest")
    raw = manifest_path.read_bytes()
    if expected_manifest_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
            raise ValueError("Trusted manifest digest must be lowercase SHA-256")
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            raise ValueError("Trusted manifest digest mismatch")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise TypeError("Model manifest must be an object")
    if manifest.get("schema") != "aetheris.model-manifest.v1":
        raise ValueError("Unsupported model manifest")
    if manifest.get("production_approved") is not False:
        raise ValueError("Research artifacts cannot carry production approval")
    if manifest.get("status") not in ("RESEARCH_ONLY", "CANDIDATE_UNVERIFIED"):
        raise ValueError("Invalid research status")
    if manifest.get("license_id") != "Apache-2.0":
        raise ValueError("Unapproved model license")
    if manifest.get("family") != "Qwen3" or manifest.get("architecture") != "Qwen3ForCausalLM":
        raise ValueError("Unsupported model family or architecture")
    for key, length in (("base_revision", 40), ("calibration_sha256", 64), ("direction_sha256", 64)):
        value = manifest.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{" + str(length) + r"}", value):
            raise ValueError(f"Invalid immutable provenance: {key}")
    integration = manifest.get("integration")
    if not isinstance(integration, dict):
        raise TypeError("Missing external policy integration")
    if (integration.get("auxiliary_policy_model_required") is not True
        or integration.get("identity_and_consent_controls") != "external_fail_closed_policy_engine"
        or integration.get("primary_gguf_role") != "PRIMARY_GGUF"):
        raise ValueError("Unsafe policy integration metadata")
    files = _file_hashes(root)
    if files != manifest.get("files_sha256"):
        raise ValueError("Model file hash mismatch")
    if _digest_file_hashes(files) != manifest.get("bundle_sha256"):
        raise ValueError("Model bundle hash mismatch")
    return True
