"""Fail-closed model and resource preflight gates."""
from __future__ import annotations

import math

import psutil


def require_approved_model(model_type: str, license_id: str | None) -> None:
    if model_type != "qwen3":
        raise ValueError(f"Unsupported model architecture: {model_type!r}")
    if not license_id or license_id.strip().lower() != "apache-2.0":
        raise ValueError(f"Unverified or unsupported model license: {license_id!r}")


def check_cpu_memory(parameter_count: int | None, available_bytes: int) -> int:
    """Conservative FP32 model estimate with 50% workspace overhead."""
    if not isinstance(parameter_count, int) or parameter_count <= 0:
        raise ValueError("Parameter count must be verified before CPU loading")
    if available_bytes <= 0:
        raise ValueError("Available memory must be positive")
    required = math.ceil(parameter_count * 4 * 1.5)
    if required > available_bytes:
        raise MemoryError(
            f"CPU memory preflight failed: require >= {required / 2**30:.1f} GiB, "
            f"available {available_bytes / 2**30:.1f} GiB"
        )
    return required


def available_memory_bytes() -> int:
    """Use OS-reported reclaimable memory (cgroup-aware in container runtime)."""
    available = int(psutil.virtual_memory().available)
    if available <= 0:
        raise MemoryError("No available system memory reported")
    return available
