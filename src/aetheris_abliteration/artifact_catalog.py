"""Fail-closed streaming verification of local GGUF artifact catalogs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}\.gguf")


def _read_regular(path: Path, limit: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Catalog must be a bounded regular file")
        raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Oversized artifact catalog")
        return raw


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_file(root_fd: int, item: dict[str, Any]) -> dict[str, Any]:
    name = item["filename"]
    report: dict[str, Any] = {"filename": name, "status": "FAILED"}
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != item["size_bytes"]:
                raise ValueError("Artifact type or size mismatch")
            header = stream.read(24)
            if len(header) != 24:
                raise ValueError("Truncated GGUF header")
            magic, version, tensors, metadata = struct.unpack("<4sIQQ", header)
            if magic != b"GGUF" or version != 3 or not tensors or not metadata:
                raise ValueError("Invalid GGUF v3 header")
            digest = hashlib.sha256(header)
            count = len(header)
            while block := stream.read(4 * 1024 * 1024):
                count += len(block)
                if count > before.st_size:
                    raise ValueError("Artifact grew during verification")
                digest.update(block)
            after = os.fstat(stream.fileno())
            current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if _identity(before) != _identity(after) or _identity(after) != _identity(current):
                raise ValueError("Artifact changed during verification")
            if count != before.st_size or digest.hexdigest() != item["sha256"]:
                raise ValueError("Artifact SHA-256 mismatch")
            report.update(status="VERIFIED", sha256=digest.hexdigest(), size_bytes=count,
                          gguf_version=version, tensor_count=tensors, metadata_count=metadata)
    except (OSError, ValueError) as exc:
        report["error"] = str(exc)
    return report


def verify_catalog(catalog: Path, root: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Verify all declared artifacts, never treat an incomplete catalog as success.

    This verifies bytes and GGUF headers, not publisher identity, complete tensor
    layouts or model behavior. A separately trusted digest can pin the catalog.
    """
    raw = _read_regular(catalog, 2 * 1024 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and (not _SHA256.fullmatch(expected_sha256) or digest != expected_sha256):
        raise ValueError("Trusted catalog SHA-256 mismatch")
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise TypeError("Catalog must be an object")
    if (doc.get("schema") != "aetheris.artifact-catalog.v1"
        or doc.get("model_status") != "RESEARCH_ONLY"
        or doc.get("production_approved") is not False):
        raise ValueError("Invalid catalog schema or research policy")
    rows = doc.get("artifacts")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("Catalog must declare 1..128 artifacts")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Artifact row must be an object")
        name, size, sha = row.get("filename"), row.get("size_bytes"), row.get("sha256")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in seen:
            raise ValueError("Invalid or duplicate artifact basename")
        if type(size) is not int or size <= 24:
            raise ValueError("Invalid artifact size")
        if (row.get("integrity_status") != "SHA256_COMPUTED"
            or not isinstance(sha, str) or not _SHA256.fullmatch(sha)):
            raise ValueError("Catalog contains an unverified artifact")
        seen.add(name)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        reports = [_verify_file(root_fd, row) for row in rows]
    finally:
        os.close(root_fd)
    return {
        "schema": "aetheris.artifact-verification.v1",
        "catalog_sha256": digest,
        "catalog_digest_pinned": expected_sha256 is not None,
        "verified": all(row["status"] == "VERIFIED" for row in reports),
        "production_approved": False,
        "artifacts": reports,
    }
