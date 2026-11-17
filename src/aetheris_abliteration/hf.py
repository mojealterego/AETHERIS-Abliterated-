"""Qwen3/Hugging Face adapter: activation extraction and residual output editing."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .core import estimate_separation, project_weight_, refusal_direction


def select_direction(
    train: dict[int, tuple[torch.Tensor, torch.Tensor]],
    holdout: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[int, torch.Tensor, dict[int, float]]:
    if not train or set(train) != set(holdout):
        raise ValueError("Train/holdout layer sets must match and be nonempty")
    directions = {}
    scores = {}
    for layer in sorted(train):
        target, control = train[layer]
        direction = refusal_direction(target, control)
        val_target, val_control = holdout[layer]
        score = estimate_separation(val_target, val_control, direction)
        if not math.isfinite(score):
            raise ValueError("Non-finite holdout score")
        directions[layer] = direction
        scores[layer] = score
    winner = max(sorted(scores), key=lambda layer: scores[layer])
    return winner, directions[winner], scores


def _tokenize(tokenizer: Any, prompt: str) -> torch.Tensor:
    """Use the native chat template; never inject untrusted prompt as system role."""
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
        raise ValueError("Tokenizer returned invalid input_ids")
    return ids


@torch.inference_mode()
def collect_activations(
    model: Any,
    tokenizer: Any,
    prompts: Iterable[str],
    layers: list[int],
    *,
    max_input_tokens: int = 512,
) -> dict[int, torch.Tensor]:
    """Collect last-position hidden states on CPU; never retain GPU activations."""
    if not layers:
        raise ValueError("At least one layer is required")
    count = len(model.model.layers)
    if any(not 0 <= layer < count for layer in layers):
        raise ValueError("Layer index outside model depth")
    device = model.get_input_embeddings().weight.device
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for prompt in prompts:
        ids = _tokenize(tokenizer, prompt)
        if ids.shape[1] > max_input_tokens:
            raise ValueError("Prompt exceeds max_input_tokens; truncation is forbidden")
        result = model(
            input_ids=ids.to(device), output_hidden_states=True,
            use_cache=False, return_dict=True,
        )
        if result.hidden_states is None or len(result.hidden_states) != count + 1:
            raise RuntimeError("Model did not expose expected layer hidden states")
        for layer in layers:
            collected[layer].append(result.hidden_states[layer + 1][0, -1].detach().float().cpu())
        del result
    if any(not values for values in collected.values()):
        raise ValueError("Empty activation group")
    return {layer: torch.stack(values) for layer, values in collected.items()}


def edit_qwen3_model(
    model: Any,
    direction: torch.Tensor,
    *,
    strength: float = 1.0,
    start_fraction: float = 0.15,
    end_fraction: float = 0.90,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Rank-one orthogonalization of attention and MLP residual-output projections.

    Restrict editing to Qwen3 decoder blocks. No tokenizer, output head, or
    input embedding is modified. A model-specific validation is mandatory.
    """
    if not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError("Invalid layer fraction range")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected Qwen3-style model.model.layers")
    layers = model.model.layers
    n = len(layers)
    if n < 1:
        raise ValueError("No transformer blocks")
    begin, end = int(n * start_fraction), math.ceil(n * end_fraction)
    if begin >= end:
        raise ValueError("Selected layer interval is empty")
    edited = 0
    for layer in layers[begin:end]:
        for module in (layer.self_attn.o_proj, layer.mlp.down_proj):
            project_weight_(module.weight, direction, axis="output", strength=strength, chunk_size=chunk_size)
            edited += 1
    return {"edited_matrices": edited, "layer_start": begin, "layer_end_exclusive": end, "strength": strength}


def candidate_layers(model: Any, *, start: float = 0.20, end: float = 0.80, count: int = 8) -> list[int]:
    if not 0 <= start < end <= 1 or count < 1:
        raise ValueError("Invalid layer search configuration")
    n = len(model.model.layers)
    low, high = int(n * start), min(n - 1, math.ceil(n * end) - 1)
    return sorted({round(low + (high - low) * i / max(1, count - 1)) for i in range(count)})


def make_groups(
    model: Any, tokenizer: Any, calibration: Any, layers: list[int]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    target = collect_activations(model, tokenizer, calibration.target, layers)
    control = collect_activations(model, tokenizer, calibration.control, layers)
    return {layer: (target[layer], control[layer]) for layer in layers}
