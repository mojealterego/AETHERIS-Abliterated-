# AETHERIS Ω — Roadmap

## Verified baseline v0.3.0

- [x] Qwen3-0.6B research checkpoint built in isolated lab.
- [x] F16, Q4_K_M, Q8_0 GGUF exports; CPU inference smoke tests.
- [x] Weight comparison: 44 changed projections, 254 protected tensors unchanged; verified tied embeddings.
- [x] Paired execution harness and research evidence gate; 51 local tests passed for the lab source.
- [ ] Git checkout validation at exact published commit.
- [ ] Full SHA-256 manifest of completed GGUF release artifacts.
- [ ] Independent, representative evaluation of language, factual accuracy, repetition and safety.
- [ ] Larger checkpoint infrastructure preflight, actual construction and independent evaluation.
- [ ] Android ARM64 build and on-device testing.
- [ ] OmniFlow integration under external policy enforcement.

No production approval. Existing heuristic comparison did not establish refusal reduction (0/2 before; 0/2 after). Code releases are distinct from model training or model-quality improvements. Large artifacts stay outside ordinary Git history.
