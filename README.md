# AETHERIS Ω Abliteration Lab — v0.3.2

A local, auditable Qwen3 refusal-direction research and rank-one weight-editing toolkit designed to fit **AETHERIS OmniFlow Ω** Dual-GGUF planning. It does not depend on ComfyUI and does not replace OmniFlow's independent Policy Engine.

## Scope and status

- **Implemented**: strict calibration JSONL validation, deterministic holdout, layer search, residual-direction estimation, chunked FP32 rank-one weight edits for Qwen3 attention `o_proj` and MLP `down_proj`, quality regression gate, atomic safetensors publication, cryptographic provenance manifest, bundle verification, explicit GGUF converter integration.
- **Verified locally**: unit tests and a tiny randomly initialized Qwen3 architecture test; see `tests/`.
- **Not yet verified**: quality of an 8B abliterated checkpoint, NSFW compliance quality, third-party red-team results, end-to-end OmniFlow deployment.
- **Not a safety certification**: refusal rate is a heuristic string match, and no model should be deployed solely on that metric.

## Technical rationale

Arditi et al. (NeurIPS 2024) identified a low-dimensional direction mediating refusal in tested models. This project applies a configurable rank-one output-space orthogonalization to a Qwen3 instruct checkpoint. It estimates the direction on the *training* calibration split and selects a layer using a disjoint holdout split. It evaluates control-language-model loss before and after editing, and aborts publication if degradation exceeds a configurable threshold. A direction estimated from adult-vs-romance prompts is **not automatically a verified refusal direction**; baseline refusal detection is mandatory by default.

### Install (Python 3.12+)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

For GPU execution, install a compatible PyTorch CUDA build for the installed NVIDIA driver first. Do not assume a CPU-only environment can load 8B weights. Qwen3-8B BF16 weights alone require roughly 16.4 GB; allow additional memory for activations, calibration and saving.

### Inspect calibration data

```bash
aetheris-abliterate inspect \
  --calibration examples/calibration-demo.jsonl \
  --min-per-class 8
```

The bundled dataset is **only a smoke-test fixture** (12 target + 12 control). A production calibration must contain at least 64 varied, legally sourced, consent-focused prompts in each class, with a truly independent test set beyond the layer-selection holdout.

### Build (GPU, research candidate)

```bash
aetheris-abliterate build \
  --model Qwen/Qwen3-8B \
  --revision main \
  --calibration /secure/datasets/validated_adult_refusal_calibration.jsonl \
  --output /secure/models/aetheris-qwen3-8b-abliterated \
  --strength 0.8
```

The build resolves the upstream revision to an immutable commit SHA. It requires an Apache-2.0 licensed Qwen3 model. By default, it refuses to edit a model unless the baseline refusal heuristic is >= 25% on the target prompts. Use `--allow-small-calibration` **only** for research smoke tests. Use `--allow-cpu` only on small models with adequate RAM. The builder never uploads weights.

### Verify and export

```bash
aetheris-abliterate verify --output /secure/models/aetheris-qwen3-8b-abliterated
aetheris-abliterate export-gguf \
  --input /secure/models/aetheris-qwen3-8b-abliterated \
  --converter /opt/llama.cpp/convert_hf_to_gguf.py \
  --output /secure/models/aetheris-qwen3-8b-f16.gguf
```

The converter is a trusted, explicitly provided local script; its version should be pinned and separately audited. Quantize with llama.cpp tools only after evaluating precision and output quality. GGUF conversion is not performed automatically.

### Security boundaries

The modified model has **no direct access** to network, biometric data, file storage, tool execution or publication. The AETHERIS OmniFlow Policy Engine must independently enforce consent, adult-only access, non-consensual intimate imagery restrictions, tenant isolation, licensing, provenance and human review where applicable. A prompt or model output must never be treated as authorization.

### Integration

`docs/omni-integration.json` describes the proposed PRIMARY_GGUF adapter contract. The AUXILIARY_GGUF remains a separate schema and policy verifier. The adapter is a **design contract**, not an installed OmniFlow node.

### References

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024: https://papers.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html
- Reproduction repository: https://github.com/andyrdt/refusal_direction
- Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B

## Build provenance

Each completed build produces `aetheris-manifest.json` with immutable source commit, calibration hash, direction hash, layer selection, loss/refusal probes, per-file SHA256 and deterministic bundle digest. Status remains `CANDIDATE_UNVERIFIED` until independent acceptance testing and production authorization.


## Release v0.2.0

See `docs/RELEASE_EVIDENCE.md` for test and compilation evidence and `docs/ANDROID_TERMUX.md` for mobile build instructions. F16, Q4_K_M and Q8_0 are separate GGUF model artifacts; source ZIP and wheel intentionally exclude weights. The 8B model has not been built.

## Release v0.3.0
Research-only evidence gate, Qwen3 no-thinking benchmark, original-vs-modified raw responses, and strict weight integrity verification.
See docs/RESEARCH_AUDIT_v0.3.md. The Qwen3-0.6B research weights remain unchanged. No 8B checkpoint was produced.

## Release 0.3.1 — Trust and artifact integrity

Manifest verification rejects unsupported production status, license, malformed provenance and unsafe policy integration metadata. An independently trusted SHA-256 of the manifest can be supplied via the verify command. Without a trusted digest, consistency checks do not establish publisher identity. See docs/ARTIFACTS.json for the actual model artifact sizes and full hashes completed so far. F16 full hashing remains pending. This is a toolkit update, not a new training run.

## Release 0.3.2 — Artifact verification

All three GGUF artifact SHA-256 values are recorded in docs/ARTIFACTS.json. On Linux/Termux, run the following command against a directory containing the files. A trusted digest of the catalog can optionally pin the catalog bytes. Verification checks file size, header, SHA-256 and file identity; it does not certify model quality or publisher identity.

```bash
aetheris-abliterate verify-artifacts --catalog docs/ARTIFACTS.json --root /path/to/gguf
```
