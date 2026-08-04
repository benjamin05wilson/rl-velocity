# rl-velocity

An instrumented GRPO harness for a question that is rarely measured: **where does RL
training compute actually go, and how much of it buys no learning at all?**

## The question

In GRPO, each prompt is answered `G` times and the group's rewards are turned into
relative advantages. When all `G` completions score identically — all correct, or all
wrong — every advantage in that group is zero. Those tokens were still generated. They
were still paid for in GPU-seconds. They contributed nothing to the gradient.

On a partly-trained model against a maths dataset, that is routinely a third to a half
of the generation budget, and generation is already the dominant cost in the loop. It is
widely known and rarely quantified, because the standard logging stack records what the
*model* did (reward, KL, loss) and not what the *hardware* did.

This repo measures both, from the first commit, on one 24GB GPU.

## Why one GPU is the point, not a limitation

Almost all RL infrastructure work is evaluated where compute is abundant. The regime
where it is scarce — where rollout and training must cohabit on a single card, and every
wasted token is a token you cannot afford — is materially different and comparatively
unexplored, because the people with clusters have no reason to care.

Sample efficiency per GPU-hour is a different objective from sample efficiency per
environment step, and it is the one that matters if you are paying for the GPU.

## Design commitments

**Timing that survives async CUDA.** Wrapping `perf_counter` around kernel launches
measures nothing, and synchronising every phase to fix it destroys the overlap being
measured. Phase timing uses CUDA events, with a single sync per step. Wall time and
device time are both recorded — their divergence is the GPU-starvation signal, and in an
RL loop that gap is usually where the wins are.

**Append-only, flushed on write.** RL runs die hours in. A run killed at step 4000 leaves
3999 readable steps on disk. JSONL rather than a metrics service, so the log diffs,
greps, and replays offline.

**Waste accounting as a first-class metric.** `degenerate_frac` and
`wasted_token_frac` are in the step schema alongside reward and KL, because the whole
point is to make the cost of a zero-advantage group visible rather than inferred.

## Status

| Stage | State |
|---|---|
| Blackwell `sm_120` toolchain verified | done — native kernels, 101 TFLOP/s bf16 |
| Instrumentation layer | done |
| vLLM generation on `sm_120` | done — 10.9k tok/s at 45% of the card |
| GRPO loop end to end | done — naive baseline, HF `generate` |
| Rollout/train overlap via vLLM | not started |
| Adaptive rollout allocation | not started |

### First measurement

Six steps of GRPO on Qwen2.5-0.5B-Instruct, 4 prompts × 8 completions, 400 max new
tokens, single 24GB card:

- **Rollout is 88–92% of step wall time**, every step, without exception.
- Generation runs at ~800–1200 tok/s through HF `generate`. The same card does
  **10,913 tok/s** under vLLM at 45% memory — so the dominant cost in the loop is
  running roughly an order of magnitude below what the hardware does.
- Degenerate groups ranged 0–100% of the batch across six steps, and wasted tokens
  tracked it closely (0–100%).

Six steps is a smoke test, not a result. The degenerate fraction is far too noisy at
4 prompts per step to characterise, and two steps showed a reward collapse that has not
been explained yet. What the numbers do establish is that the bottleneck is where the
thesis expected it, and that there is a large measured gap to close.

Nothing else goes in this table until it has been measured.

## Environment

WSL2 Ubuntu 22.04, RTX PRO 5000 Blackwell (24GB, `sm_120`), Python 3.12.

```bash
uv venv --python 3.12
uv pip install -e .
python scripts/verify_gpu.py      # gate 1: torch really has sm_120 kernels
python scripts/verify_vllm.py     # gate 2: vLLM generates on sm_120
```

Both gates run before any training code, because a silently degraded stack costs weeks.
`verify_gpu.py` checks `torch.cuda.get_arch_list()` explicitly rather than trusting
`is_available()`, and measures achieved TFLOP/s rather than assuming a correct matmul
took the tensor-core path.
