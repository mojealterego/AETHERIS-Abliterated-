"""AETHERIS Ω CLI: inspect, build, verify, and explicit GGUF conversion."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .dataset import load_calibration, sha256_file
from .manifest import verify_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aetheris-abliterate", description="Controlled Qwen3 refusal-direction research")
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect", help="Validate local calibration JSONL without loading a model")
    inspect.add_argument("--calibration", required=True)
    inspect.add_argument("--min-per-class", type=int, default=32)

    build = subs.add_parser("build", help="Produce a locally modified safetensors model")
    build.add_argument("--calibration", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--model", default="Qwen/Qwen3-8B")
    build.add_argument("--revision", default="main")
    build.add_argument("--device-map", default="auto")
    build.add_argument("--strength", type=float, default=0.8)
    build.add_argument("--candidate-layers", type=int, default=8)
    build.add_argument("--edit-start", type=float, default=0.15)
    build.add_argument("--edit-end", type=float, default=0.9)
    build.add_argument("--min-per-class", type=int, default=64)
    build.add_argument("--allow-small-calibration", action="store_true")
    build.add_argument("--allow-cpu", action="store_true")
    build.add_argument("--eval-samples", type=int, default=4)
    build.add_argument("--max-input-tokens", type=int, default=512)
    build.add_argument("--max-new-tokens", type=int, default=64)
    build.add_argument("--min-baseline-refusal-rate", type=float, default=0.25)
    build.add_argument("--max-control-loss-delta", type=float, default=0.35)

    verify = subs.add_parser("verify", help="Verify local safetensors bundle integrity")
    verify.add_argument("--output", required=True)
    verify.add_argument("--expected-manifest-sha256", default=None)

    catalog = subs.add_parser("verify-artifacts", help="Verify local GGUF catalog against recorded hashes")
    catalog.add_argument("--catalog", required=True)
    catalog.add_argument("--root", required=True)
    catalog.add_argument("--expected-catalog-sha256", default=None)

    export = subs.add_parser("export-gguf", help="Run an explicitly supplied trusted llama.cpp converter")
    export.add_argument("--input", required=True)
    export.add_argument("--converter", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--outtype", choices=("f16", "bf16"), default="f16")
    audit = subs.add_parser("audit-evidence", help="Human-labelled paired refusal/quality/safety gate")
    audit.add_argument("--input", required=True)

    bench = subs.add_parser("benchmark", help="Capture local GGUF outputs for independent human review")
    bench.add_argument("--binary", required=True)
    bench.add_argument("--tokenizer", default=None, help="Local Qwen3 tokenizer directory; disables thinking")
    bench.add_argument("--model", required=True)
    bench.add_argument("--prompts", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--timeout", type=int, default=180)
    bench.add_argument("--max-new-tokens", type=int, default=96)
    bench.add_argument("--threads", type=int, default=4)
    bench.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        data = load_calibration(Path(args.calibration), min_per_class=args.min_per_class)
        print(json.dumps({
            "target_count": len(data.target),
            "control_count": len(data.control),
            "calibration_sha256": sha256_file(Path(args.calibration)),
            "schema": "aetheris.calibration.v1",
        }, indent=2))
        return 0
    if args.command == "build":
        from .pipeline import build
        result = build(args)
        print(json.dumps({
            "output": result["output"],
            "status": result["manifest"]["status"],
            "bundle_sha256": result["manifest"]["bundle_sha256"],
            "metrics": result["manifest"]["metrics"],
        }, indent=2))
        return 0
    if args.command == "verify":
        verify_manifest(Path(args.output), expected_manifest_sha256=args.expected_manifest_sha256)
        print(json.dumps({"verified": True, "output": args.output}))
        return 0
    if args.command == "verify-artifacts":
        from .artifact_catalog import verify_catalog

        report = verify_catalog(
            Path(args.catalog), Path(args.root), expected_sha256=args.expected_catalog_sha256,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["verified"] else 2
    if args.command == "export-gguf":
        source = Path(args.input).resolve()
        converter = Path(args.converter).resolve()
        output = Path(args.output).resolve()
        verify_manifest(source)
        if not converter.is_file() or converter.is_symlink():
            raise ValueError("Converter must be a local, explicitly supplied Python script")
        if output.exists():
            raise FileExistsError(output)
        subprocess.run([
            sys.executable, str(converter), str(source),
            "--outfile", str(output), "--outtype", args.outtype,
        ], check=True, timeout=7200)
        print(json.dumps({"gguf": str(output), "size_bytes": output.stat().st_size}))
        return 0
    if args.command == "audit-evidence":
        from .evidence_gate import audit_evidence, load_evidence_jsonl

        report = audit_evidence(load_evidence_jsonl(Path(args.input)))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "RESEARCH_CANDIDATE" else 2
    if args.command == "benchmark":
        from .benchmark_runner import benchmark_to_file

        count = benchmark_to_file(
            Path(args.binary), Path(args.model), Path(args.prompts), Path(args.output),
            timeout=args.timeout, max_new_tokens=args.max_new_tokens,
            threads=args.threads, seed=args.seed,
            tokenizer_source=Path(args.tokenizer) if args.tokenizer else None,
        )
        print(json.dumps({"captured": count, "file": args.output, "judgement": "UNREVIEWED"}))
        return 0
    raise AssertionError("Unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
