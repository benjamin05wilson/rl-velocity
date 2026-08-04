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

## Results

Qwen2.5-0.5B-Instruct, GSM8K, 4 prompts × 8 completions, 400 max new tokens, single
24GB card. Both backends run the identical loss, data, seed and optimiser — only the
rollout path differs. Regenerate with `python scripts/analyse.py hf-fixed vllm-fixed`.

| | HF `generate` | vLLM | |
|---|---|---|---|
| step wall time | 10.34 s | 2.14 s | **4.83×** |
| rollout device time | 9.29 s | 1.11 s | **8.40×** |
| generation throughput | 1,076 tok/s | 8,898 tok/s | **8.27×** |
| rollout share of step | 90% | 52% | |
| weight sync | — | 0.010 s | |
| peak memory | 12.3 GB | 20.4 GB | |

Weight sync costs 0.010s per step and is counted inside the step, because an
optimisation that relocates work somewhere you stopped measuring is not an
optimisation. The Amdahl ceiling from eliminating rollout entirely was 9.9×; 4.83×
of it is realised, and the remaining step time is now 48% non-rollout — the next
bottleneck is the training forward/backward, not generation.

### The speedup was confounded, and the check caught it

The first measurement of this comparison was invalid. Verified before publishing, by
`scripts/compare_backends.py`:

| | before | after |
|---|---|---|
| greedy first-100-char agreement | 31% | **97%** |
| sampled mean reward (hf vs vllm) | 0.262 vs 0.406 | 0.398 vs 0.406 |
| significance of that gap | z = 3.51 | z = 0.18 |
| mean completion length | 327 vs 302 | 304 vs 302 |

Cause: Qwen2.5 ships `generation_config.json` with `top_k=20` and
`repetition_penalty=1.1`. HF `generate` silently applies all of it unless each field is
overridden individually; vLLM's `SamplingParams` defaults to neutral. Overriding only
temperature and top_p left the two backends sampling different distributions, and the
throughput comparison was measuring two things at once.

**This is not only a benchmarking artefact.** On-policy RL requires that the
distribution you sample from is the distribution you compute logprobs under. A
repetition penalty applied during rollout but absent from the training forward pass
makes them differ, so the policy gradient is computed against the wrong distribution —
silently, with no error and a plausible-looking reward curve. On maths it is actively
harmful: correct arithmetic means re-emitting digits, which is exactly what the penalty
suppresses. Pinning neutral sampling raised HF reward from 0.262 to 0.398 on its own.

Both backends now pin `top_k`, `top_p` and `repetition_penalty` explicitly
(`NEUTRAL_SAMPLING` in `rollout.py`).

## Status

| Stage | State |
|---|---|
| Blackwell `sm_120` toolchain verified | done — native kernels, 101 TFLOP/s bf16 |
| Instrumentation layer | done |
| GRPO loop end to end | done |
| vLLM rollout backend + weight sync | done — 4.83× step time, equivalence verified |
| Degenerate-waste characterisation | not started — needs multi-seed runs |
| Adaptive rollout allocation | not started |

Degenerate groups averaged 29–37% of the batch across these runs, but at 4 prompts per
step that is far too noisy to characterise and no claim is made from it yet. That
number is the premise for adaptive allocation, so it needs multi-seed runs at a larger
batch before anything is built on top of it.

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
