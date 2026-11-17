"""Strict, deterministic JSONL calibration loader with no implicit remote fetching."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Calibration:
    target: tuple[str, ...]
    control: tuple[str, ...]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_calibration(path: Path, *, min_per_class: int = 32) -> Calibration:
    if min_per_class < 1:
        raise ValueError("min_per_class must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Calibration dataset exceeds 8 MiB")
    classes: dict[str, list[str]] = {"target": [], "control": []}
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at line {lineno}") from exc
        if not isinstance(record, dict) or set(record) != {"kind", "prompt"}:
            raise ValueError(f"Invalid record schema at line {lineno}")
        kind, prompt = record["kind"], record["prompt"]
        if kind not in classes or not isinstance(prompt, str):
            raise ValueError(f"Invalid kind/prompt at line {lineno}")
        if not 5 <= len(prompt.strip()) <= 4096:
            raise ValueError(f"Invalid prompt length at line {lineno}")
        normalized = prompt.strip().casefold()
        if normalized in seen:
            raise ValueError(f"Duplicate prompt at line {lineno}")
        seen.add(normalized)
        classes[kind].append(prompt.strip())
    if any(len(group) < min_per_class for group in classes.values()):
        raise ValueError(f"Each group must contain >= {min_per_class} unique prompts")
    return Calibration(tuple(classes["target"]), tuple(classes["control"]))


def split_calibration(data: Calibration, *, validation_fraction: float = 0.2, seed: int = 42) -> tuple[Calibration, Calibration]:
    if not 0.1 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be in [0.1, 0.5]")
    rng = random.Random(seed)

    def split(prompts: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        values = list(prompts)
        if len(values) < 4:
            raise ValueError("At least four samples per class are needed for holdout")
        rng.shuffle(values)
        n_val = max(2, min(len(values) - 2, round(len(values) * validation_fraction)))
        return tuple(values[n_val:]), tuple(values[:n_val])

    train_t, val_t = split(data.target)
    train_c, val_c = split(data.control)
    return Calibration(train_t, train_c), Calibration(val_t, val_c)
