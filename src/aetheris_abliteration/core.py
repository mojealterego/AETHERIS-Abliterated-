"""Numerically controlled refusal-direction estimation and rank-one weight edits."""
from __future__ import annotations

from typing import Literal

import torch

Axis = Literal["output", "input"]


def refusal_direction(target: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Return normalized difference of group means, computed in FP32."""
    if target.ndim != 2 or control.ndim != 2 or target.shape[1] != control.shape[1]:
        raise ValueError("Calibration matrices must be 2D and share the hidden dimension")
    if min(target.shape[0], control.shape[0]) < 2:
        raise ValueError("At least two samples per group are required")
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(control).all()):
        raise ValueError("Calibration activations contain NaN or Inf")
    direction = target.float().mean(dim=0) - control.float().mean(dim=0)
    magnitude = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(magnitude)) or float(magnitude) <= 1e-8:
        raise ValueError("Mean difference is zero or numerically degenerate")
    return (direction / magnitude).detach()


def estimate_separation(target: torch.Tensor, control: torch.Tensor, direction: torch.Tensor) -> float:
    """Held-out standardized separation along a training-only direction."""
    if target.ndim != 2 or control.ndim != 2 or target.shape[1] != control.shape[1]:
        raise ValueError("Incompatible held-out activation shapes")
    if target.shape[0] < 2 or control.shape[0] < 2:
        raise ValueError("Held-out groups require at least two observations")
    if direction.ndim != 1 or direction.numel() != target.shape[1]:
        raise ValueError("Direction has incompatible shape")
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(control).all()):
        raise ValueError("Non-finite held-out activation")
    d = direction.float()
    norm = torch.linalg.vector_norm(d)
    if float(norm) <= 1e-8 or not bool(torch.isfinite(norm)):
        raise ValueError("Invalid direction")
    d = d / norm
    a = target.float() @ d
    b = control.float() @ d
    pooled_var = (a.var(unbiased=True) + b.var(unbiased=True)) / 2
    return float((a.mean() - b.mean()).abs() / (pooled_var.sqrt() + 1e-6))


@torch.no_grad()
def project_weight_(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    axis: Axis,
    strength: float = 1.0,
    chunk_size: int = 256,
) -> None:
    """In-place W <- (I-alpha rr^T)W for output or W(I-alpha rr^T) for input.

    Chunking limits temporary FP32 memory to one slice. BF16/FP16 parameters
    are updated in FP32 and cast back to their original storage dtype.
    """
    if axis not in ("input", "output"):
        raise ValueError("axis must be input or output")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a 2D floating-point tensor")
    dim = weight.shape[0] if axis == "output" else weight.shape[1]
    if direction.ndim != 1 or direction.numel() != dim:
        raise ValueError("direction dimension mismatch")
    if not bool(torch.isfinite(direction).all()):
        raise ValueError("non-finite direction")
    d = direction.detach().to(device=weight.device, dtype=torch.float32)
    norm = torch.linalg.vector_norm(d)
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-8:
        raise ValueError("zero or invalid direction")
    d = d / norm
    if strength == 0:
        return

    if axis == "output":
        for start in range(0, weight.shape[1], chunk_size):
            end = min(start + chunk_size, weight.shape[1])
            block = weight[:, start:end].float()
            coefficient = d @ block
            block -= strength * d[:, None] * coefficient[None, :]
            weight[:, start:end].copy_(block.to(dtype=weight.dtype))
    else:
        for start in range(0, weight.shape[0], chunk_size):
            end = min(start + chunk_size, weight.shape[0])
            block = weight[start:end, :].float()
            coefficient = block @ d
            block -= strength * coefficient[:, None] * d[None, :]
            weight[start:end, :].copy_(block.to(dtype=weight.dtype))
