# Synthetic replay manifest

All files here are hand-authored **synthetic fixtures**, created 2026-09-10. No model,
GPU, sampling or training run produced them. `hf-fixed` and `vllm-fixed` are interface
example names, not recovered historical experiments. Parent archive search found no
original receipts. No performance or learning claim is supported by these examples.

| Output | Inputs | Reproduction command (from repo root) |
|---|---|---|
| `replay.txt` | both backend directories' `meta.json` and `events.jsonl` | `python3 scripts/analyse.py --runs-dir evidence hf-fixed vllm-fixed` |
| `phase-times.svg` | same inputs | `python3 scripts/analyse.py --runs-dir evidence hf-fixed vllm-fixed --svg evidence/phase-times.svg` |
| `comparison/report.json` | `comparison/hf.json`, `comparison/vllm.json` | `python3 scripts/compare_backends.py --backend report --out-dir evidence/comparison` |

Replay requires only Python 3.9+ standard library, no installation or network. Training
and CPU tests require Python 3.12+. Step zero is excluded by step identifier, including
when it is the only record. HF fixture means: 11 seconds and 120/11 generated tokens/s;
vLLM fixture means: 7 seconds and 120/7 tokens/s. These are arithmetic checks only.
Unavailable CUDA-event measurements are null. Host phase bars are separate, not stacked:
there is no assumption of additive phases or host/device overlap. The comparison fixture
has identical rewards and divergent text/tokens to expose why rewards cannot certify
equivalence. It intentionally has no positive-control learning evidence.
