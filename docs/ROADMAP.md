# AETHERIS Ω — Roadmap

## Verified baseline and v0.3.x progress

- [x] Qwen3-0.6B research checkpoint built in isolated lab.
- [x] F16, Q4_K_M, Q8_0 GGUF exports; CPU inference smoke tests.
- [x] Weight comparison: 44 changed projections, 254 protected tensors unchanged; verified tied embeddings.
- [x] Paired execution harness and research evidence gate; 51 local tests passed for the lab source.
- [x] Clean checkout validation of import commit 30753069f2007b86eae61eef54435b62117cdad0 (51 tests; Ruff clean).
- [x] Compute full SHA-256 for Q4_K_M and Q8_0; publish partial artifact catalog.
- [x] Compute full SHA-256 of F16.
- [x] Independently re-read all three GGUF files and compare SHA-256/size/header with the measured catalog.
- [x] Cross-check all 310 tensor names/shapes and nonoverlapping byte ranges using pinned llama.cpp GGUFReader.
- [x] Add streaming, fail-closed catalog verifier and ten regression tests.
- [x] Harden manifest status, provenance and policy checks; add optional trusted digest pinning and seven regression tests (58 tests in the development checkout).
- [ ] Independent, representative evaluation of language, factual accuracy, repetition and safety.
- [ ] Larger checkpoint infrastructure preflight, actual construction and independent evaluation.
- [ ] Android ARM64 build and on-device testing.
- [ ] OmniFlow integration under external policy enforcement.

No production approval. Existing heuristic comparison did not establish refusal reduction (0/2 before; 0/2 after). Code releases are distinct from model training or model-quality improvements. Large artifacts stay outside ordinary Git history.

Clean checkout at 3e57283d1c3de72fde53ee39243f7367339103fd: 58 tests passed, Ruff clean.
