# AETHERIS Ω Abliteration Lab — Release evidence v0.2.0

## Confirmed
- Actual Qwen3-0.6B checkpoint after rank-one residual projection, 44 edited matrices, stored as Safetensors.
- Model source revision: `c1899de289a04d12100db370d81485cdf75e47ca` (Qwen/Qwen3-0.6B, Apache-2.0).
- Model weights SHA256 recorded in immutable model manifest: `5386d60c653d409a30af59c2fcb650917c9d9c91faf917a7566b1ddda0800cd6`.
- llama.cpp native x86-64 build from commit `18b74fa41dcac1f55ae1371fe58153f7225941f6`: CMake/Ninja 150/150.
- Prebuilt, upstream-official llama.cpp b7600 binary archive verified against upstream SHA256; used for initial quantization and model inference.
- GGUF v3 F16, Q4_K_M, Q8_0: 310 tensors, 34 metadata entries, valid magic.
- Q4_K_M / Q8_0 successfully loaded for one-turn local CPU inference by llama-cli.
- Dataset inspection and 29 regression/unit/integration tests passing, Ruff check clean as of build verification.

## Critical limitations
- Research calibration fixture contains 12 prompts per class, only two holdout evaluation prompts per class.
- Baseline refusal rate was 0.0 on the very small tested set, so neither removal of refusals nor NSFW quality is demonstrated.
- Regeneration of fluent, compliant prose and real-world robustness have not been independently benchmarked.
- This research checkpoint is not a production-approved generative service, has no adult-content compliance certification, and does not remove OmniFlow external policy gates.
- Larger 8B checkpoint was **not** built; an attempted third-party 8B download remained incomplete and is not in the release.
- Local CPU build provides no proof of CUDA, Vulkan, Android ARM, or smartphone-native binary.

## Release acceptance
`src/aetheris_abliteration/release.py` verifies GGUF header, size and SHA256, and keeps the manifest RESEARCH_ONLY / EXTERNAL_FAIL_CLOSED.
Use `python -m pytest -q` and `python -m aetheris_abliteration verify --output <model-dir>` for repeatable checks.
