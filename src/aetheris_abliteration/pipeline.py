"""Full, auditable local build of a modified Qwen3 open-weight model."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from .dataset import load_calibration, sha256_file, split_calibration
from .evaluation import average_control_loss, refusal_rate
from .hf import candidate_layers, edit_qwen3_model, make_groups, select_direction
from .manifest import create_manifest, verify_manifest
from .preflight import available_memory_bytes, check_cpu_memory, require_approved_model


def build(args: Any) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not 0 < args.strength <= 1:
        raise ValueError("strength must be in (0, 1]")
    if not 0 <= args.min_baseline_refusal_rate <= 1:
        raise ValueError("Invalid minimum baseline refusal rate")
    if args.max_control_loss_delta < 0:
        raise ValueError("Loss regression threshold cannot be negative")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    min_count = 4 if args.allow_small_calibration else args.min_per_class
    calibration_path = Path(args.calibration).expanduser().resolve()
    calibration = load_calibration(calibration_path, min_per_class=min_count)
    train, holdout = split_calibration(calibration, validation_fraction=0.2, seed=42)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; specify --allow-cpu for small research models only")

    # Resolve a mutable branch/tag to an immutable commit SHA for provenance.
    info = HfApi().model_info(args.model, revision=args.revision)
    revision = info.sha
    if not revision or len(revision) < 16:
        raise RuntimeError("Unable to resolve immutable model revision")
    config = AutoConfig.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    license_value = getattr(getattr(info, "card_data", None), "license", None)
    require_approved_model(getattr(config, "model_type", None), license_value)
    if not torch.cuda.is_available():
        parameter_count = getattr(getattr(info, "safetensors", None), "total", None)
        available = available_memory_bytes()
        check_cpu_memory(parameter_count, available)

    dtype = torch.float32 if not torch.cuda.is_available() else torch.bfloat16
    torch.set_num_threads(min(os.cpu_count() or 4, 8))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=dtype,
        device_map=args.device_map, trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    model.eval()
    layers = candidate_layers(model, count=args.candidate_layers)
    train_states = make_groups(model, tokenizer, train, layers)
    val_states = make_groups(model, tokenizer, holdout, layers)
    selected_layer, direction, separation = select_direction(train_states, val_states)
    del train_states, val_states

    sample_count = min(args.eval_samples, len(holdout.control), len(holdout.target))
    controls = holdout.control[:sample_count]
    targets = holdout.target[:sample_count]
    loss_before = average_control_loss(model, tokenizer, controls, max_tokens=args.max_input_tokens)
    baseline_refusal = refusal_rate(
        model, tokenizer, targets, max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    if baseline_refusal < args.min_baseline_refusal_rate:
        raise ValueError(
            f"Refusal baseline {baseline_refusal:.3f} below threshold "
            f"{args.min_baseline_refusal_rate:.3f}; do not edit an already-permissive model"
        )

    edit_info = edit_qwen3_model(
        model, direction, strength=args.strength,
        start_fraction=args.edit_start, end_fraction=args.edit_end,
    )
    loss_after = average_control_loss(model, tokenizer, controls, max_tokens=args.max_input_tokens)
    loss_delta = loss_after - loss_before
    if loss_delta > args.max_control_loss_delta:
        raise RuntimeError(
            f"Control loss regression {loss_delta:.4f} exceeds "
            f"{args.max_control_loss_delta:.4f}; modified weights not saved"
        )
    refusal_after = refusal_rate(
        model, tokenizer, targets, max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    direction_hash = hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()
    metrics = {
        "baseline_refusal_rate_heuristic": baseline_refusal,
        "modified_refusal_rate_heuristic": refusal_after,
        "control_loss_before": loss_before,
        "control_loss_after": loss_after,
        "control_loss_delta": loss_delta,
        "holdout_separation_by_layer": {str(k): v for k, v in separation.items()},
        "eval_sample_count_per_class": sample_count,
        "warning": "Heuristic refusal detection; no independent safety or NSFW quality certification",
    }
    research_only = args.allow_small_calibration or min_count < 32

    # Atomic publication only after all checks, hashes and manifest are complete.
    staging = Path(tempfile.mkdtemp(prefix=".aetheris-build-", dir=output.parent))
    try:
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(staging)
        manifest = create_manifest(
            staging, base_model=args.model, base_revision=revision,
            calibration_sha256=sha256_file(calibration_path),
            layer=selected_layer, direction_sha256=direction_hash,
            metrics=metrics, edit=edit_info, research_only=research_only,
        )
        (staging / "aetheris-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        verify_manifest(staging)
        os.replace(staging, output)
        return {"output": str(output), "manifest": manifest}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
