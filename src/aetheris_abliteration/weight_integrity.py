"""Strong BF16/FP32 weight integrity checks for precisely scoped abliteration edits.

Allows the omission of lm_head.weight only when the source checkpoint
explicitly has tied embeddings and every element of source lm_head and
source embedding is identical. Protects all non-projected parameters.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_ALLOWED_EDIT = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.o_proj|mlp\.down_proj)\.weight$"
)
_EMBED = "model.embed_tokens.weight"
_HEAD = "lm_head.weight"


def _equal_values(a: Any, b: Any, chunk_rows: int) -> bool:
    if tuple(a.get_shape()) != tuple(b.get_shape()):
        return False
    dims = a.get_shape()
    if not dims:
        return torch.equal(a[:].float(), b[:].float())
    for start in range(0, dims[0], chunk_rows):
        left, right = a[start:start + chunk_rows], b[start:start + chunk_rows]
        if not torch.equal(left.float(), right.float()):
            return False
    return True


def validate_modified_weights(
    original: Path,
    modified: Path,
    *,
    tied_embeddings: bool,
    chunk_rows: int = 256,
) -> dict[str, Any]:
    """Validate exact weight-key parity and limit numerical changes to output projections."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    for path in (original, modified):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Weights must be local regular safetensors files without symlinks")
    edited = 0
    protected = 0
    tied_verified = False
    with safe_open(original, framework="pt", device="cpu") as src, safe_open(
        modified, framework="pt", device="cpu"
    ) as dest:
        original_keys = set(src.keys())
        modified_keys = set(dest.keys())
        missing = original_keys - modified_keys
        extra = modified_keys - original_keys
        if extra:
            raise ValueError(f"Unexpected model tensors: {sorted(extra)}")
        if missing:
            if missing != {_HEAD} or not tied_embeddings:
                raise ValueError(f"Missing non-tied tensors: {sorted(missing)}")
            if _HEAD not in original_keys or _EMBED not in original_keys:
                raise ValueError("Source lacks the weights required to verify tied embeddings")
            if not _equal_values(src.get_slice(_HEAD), src.get_slice(_EMBED), chunk_rows):
                raise ValueError("Omitted output head differs from source token embeddings")
            tied_verified = True
        if _EMBED in original_keys and _EMBED not in modified_keys:
            raise ValueError("Source embeddings were omitted")
        for name in sorted(modified_keys):
            left = src.get_slice(name)
            right = dest.get_slice(name)
            if tuple(left.get_shape()) != tuple(right.get_shape()):
                raise ValueError(f"Tensor shape changed: {name}")
            same = _equal_values(left, right, chunk_rows)
            if _ALLOWED_EDIT.fullmatch(name):
                edited += int(not same)
            elif not same:
                raise ValueError(f"Prohibited change outside projections: {name}")
            else:
                protected += 1
        if edited < 1:
            raise ValueError("No weight-projection edits found")
    return {
        "schema": "aetheris.weight-integrity.v1",
        "source_tensor_count": len(original_keys),
        "modified_tensor_count": len(modified_keys),
        "edited_tensors": edited,
        "protected_tensors": protected,
        "unexpected_changes": 0,
        "tied_alias_verified": tied_verified,
        "status": "STRUCTURE_VERIFIED",
        "content_effectiveness": "NOT_TESTED",
    }
