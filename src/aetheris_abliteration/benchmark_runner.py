"""Deterministic local llama.cpp execution for human adjudication of model outputs."""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def load_prompts(path: Path, *, max_bytes: int = 1_048_576) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("Prompt suite must be a local regular JSONL file < 1 MiB")
    prompts = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}") from exc
        if not isinstance(obj, dict) or set(obj) != {"id", "prompt"}:
            raise ValueError("Each prompt must have id and prompt only")
        if not isinstance(obj["id"], str) or not _ID.fullmatch(obj["id"]):
            raise ValueError("Invalid prompt id")
        if obj["id"] in seen:
            raise ValueError("Duplicate prompt id")
        text = obj["prompt"]
        if not isinstance(text, str) or not 5 <= len(text) <= 4096:
            raise ValueError("Prompt must be a string of 5..4096 characters")
        seen.add(obj["id"])
        prompts.append(obj)
        if len(prompts) > 500:
            raise ValueError("Too many prompts")
    if not prompts:
        raise ValueError("Empty prompt suite")
    return prompts


def run_one_turn(
    binary: Path, model: Path, prompt: str, *, timeout: int = 180,
    max_new_tokens: int = 96, seed: int = 42, threads: int = 4,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("Binary must be a local, executable file (no symlinks)")
    if model.is_symlink() or not model.is_file():
        raise ValueError("Model must be a local, regular GGUF file")
    if not isinstance(prompt, str) or not 5 <= len(prompt) <= 4096:
        raise ValueError("Invalid prompt length")
    if timeout < 1 or max_new_tokens < 1 or threads < 1:
        raise ValueError("Invalid execution bounds")
    with model.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or struct.unpack("<4sIQQ", header)[0] != b"GGUF":
        raise ValueError("Model GGUF magic not found")
    input_text = prompt
    if tokenizer is not None:
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(input_text, str) or not input_text:
            raise ValueError("Tokenizer did not return a valid chat template")
    result = subprocess.run(
        [
            str(binary.resolve()), "-m", str(model.resolve()), "-p", input_text,
            "-n", str(max_new_tokens), "-c", "1024",
            "-t", str(threads), "--temp", "0", "--seed", str(seed),
            "--single-turn", "--simple-io",
            *(["--no-conversation"] if tokenizer is not None else []),
        ],
        capture_output=True, text=True, timeout=timeout, check=True,
    )
    return {
        "model": model.name,
        "input": prompt,
        "raw_output": result.stdout,
        "stderr": result.stderr[-10000:],
        "model_provenance": "LOCAL_MODEL_FILE",
        "model_output_mode": "QWEN_NO_THINK" if tokenizer is not None else "MODEL_DEFAULT",
        "judgement": "UNREVIEWED",
    }


def benchmark_to_file(
    binary: Path,
    model: Path,
    prompts: Path,
    output: Path,
    *,
    timeout: int = 180,
    max_new_tokens: int = 96,
    seed: int = 42,
    threads: int = 4,
    tokenizer_source: Path | None = None,
) -> int:
    """Produce raw inference for human labelling; never overwrite existing evidence."""
    records = load_prompts(prompts)
    tokenizer = None
    if tokenizer_source is not None:
        if tokenizer_source.is_symlink() or not tokenizer_source.is_dir():
            raise ValueError("Tokenizer source must be a local directory without symlink")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_source), local_files_only=True, trust_remote_code=False,
        )
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"Refusing to overwrite prior benchmark evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir():
        raise ValueError("Benchmark output parent is not a directory")
    temp_path: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=".aetheris-evidence-", suffix=".tmp", dir=output.parent)
        temp_path = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as dest:
            for prompt in records:
                result = run_one_turn(
                    binary, model, prompt["prompt"], timeout=timeout,
                    max_new_tokens=max_new_tokens, seed=seed, threads=threads,
                    tokenizer=tokenizer,
                )
                result["id"] = prompt["id"]
                dest.write(json.dumps(result, ensure_ascii=False) + "\n")
            dest.flush()
            os.fsync(dest.fileno())
        os.link(temp_path, output)  # Atomic creation; fails if output already exists.
        return len(records)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
