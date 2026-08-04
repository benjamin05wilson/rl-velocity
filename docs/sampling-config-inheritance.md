# Sampling parameters you did not set are not neutral

**What this is.** The leak of `generation_config.json` values into RL rollouts is
**already known**, reported repeatedly since March 2025, and its consequence for
importance sampling was stated by DeepSeek-V3.2 in December 2025. This document does not
claim to have discovered any of that, and cites it up front.

What appears to be missing from the public record, and is added here:

1. **A per-token importance-ratio characterisation.** Prior work reports aggregate KL,
   ESS and sequence-level weights. Nobody has published the per-token distribution.
2. **The repetition-penalty case.** All prior treatment is top-p/top-k, which is
   truncate-and-renormalise. A repetition penalty is stateful and non-uniform, so
   DeepSeek's Keep Sampling Mask does not repair it. It is unaddressed anywhere found.
3. **A transformers v5 regression** that removed both the warning and the opt-out,
   making the leak strictly quieter than the version everyone wrote up.
4. **Two live unreported instances** in maintained frameworks (veRL, NeMo-RL).

Measured on transformers 5.14.1, vLLM 0.26.0, Qwen2.5-0.5B-Instruct, sm_120.

---

## Prior art — read this first

| source | date | what it said |
|---|---|---|
| [verl#702](https://github.com/verl-project/verl/issues/702) | 2025-03-21 | "during the rollout, if we do not explicitly assign the decoding params (e.g., top_p, temperature, repetition_penalty), then it will load the params in `generation_config.json`. This can lead to unexpected behaviors and results" |
| [vLLM#12622](https://github.com/vllm-project/vllm/pull/12622) | 2025-03 | had to lower an lm_eval accuracy threshold because Qwen2-1.5B-Instruct ships `repetition_penalty: 1.1` |
| [kalomaze](https://x.com/kalomaze/status/1926751357983154606) | 2025-05-25 | "ALL of my GRPO runs have been done with an extremely narrow subset of the probability distribution" — `top_k` silently 50 |
| [transformers#42762](https://github.com/huggingface/transformers/issues/42762) | 2025-12-10 | checkpoint config overrides explicit config; names TRL GRPO/RLOO as victims. Fixed by #42702; TRL workaround [trl#4647](https://github.com/huggingface/trl/pull/4647) |
| [trl#5783](https://github.com/huggingface/trl/issues/5783) | 2026-05-18 | fullest public writeup; lists all four leaked fields for Qwen2.5-VL-3B and correctly notes the vLLM path is unaffected |

And on the *consequence* — that non-neutral rollout sampling violates the importance-
sampling assumption — the prior art is stronger still:

| source | date | what it said |
|---|---|---|
| [DeepSeek-V3.2 tech report](https://arxiv.org/abs/2512.02556) §3.1 | 2025-12-02 | "such truncation … introduces a mismatch between the action spaces of π_old and π_θ, **which violates the principles of importance sampling and destabilizes training**". Ships a fix — "Keep Sampling Mask" — preserving truncation masks from sampling and applying them in training. |
| [Fang & Khazi, "Mismatch Praxis"](https://www.paperinstruments.com/blog/mismatch-praxis) | 2025-12-02 | Ran the experiment on Qwen3-8B/GSM8K using Qwen3's own shipped values (τ=0.7, top_p=0.8, top_k=20): rollout-train KL **0.001 → 0.033 (~33×)**, ~100% of sequences given negligible Seq-TIS weight, length plateau at the predicted ~600 tokens. |
| [verl#6240](https://github.com/verl-project/verl/issues/6240) | 2026-05-04 | asks exactly this question about truncated-vs-full-vocabulary logprobs. **Still zero replies.** |

The first sentence of verl#702 is, in substance, the entire mechanism — including
`repetition_penalty` by name, in an RL rollout context, seventeen months ago. Anyone
presenting the mechanism as a discovery has not looked. The broader
training-inference mismatch literature ([Yao et al.](https://fengyao.notion.site/off-policy-rl),
[Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/),
[FP16 paper](https://arxiv.org/abs/2510.26788)) is large and active — but attributes
mismatch to kernels, precision and staleness, essentially never to sampling config.

## The mechanism, stated correctly

**HF `transformers`.** `GenerationConfig`'s defaults for `temperature`, `top_k`,
`top_p` and `repetition_penalty` are all `None`. Any field left `None` is back-filled
from the checkpoint's `generation_config.json`. Setting a non-`None` value blocks it.

Verified behaviourally (`scripts/audit_generation_config.py`), greedy decode so any
difference is configuration rather than sampling noise:

| call | result |
|---|---|
| `GenerationConfig(do_sample=False)` — omits the field | 757 chars |
| `GenerationConfig(do_sample=False, repetition_penalty=1.0)` | 734 chars |
| `GenerationConfig(do_sample=False, repetition_penalty=1.1)` | 757 chars |

The omitted-field output is byte-identical to the checkpoint's 1.1. **Passing an
explicit `generation_config` does not isolate you from the checkpoint.**

**vLLM — and here is the correction.** It is widely repeated, including in trl#5783,
that "the vLLM path is not affected." That is true only of the *offline* API. There are
two paths and they behave differently:

| vLLM call shape | omitted field resolves to |
|---|---|
| `llm.generate(prompts, SamplingParams(...))` — explicit object | **neutral** library default |
| `llm.generate(prompts)` — no SamplingParams | the checkpoint |
| OpenAI-compatible **server** request | the checkpoint |

The server path re-introduces the leak because `--generation-config` defaults to
`"auto"`; `protocol.py` fills unset `repetition_penalty`/`temperature`/`top_p` from
`model_config.get_diff_sampling_param()`. Only `generation_config="vllm"` neutralises
it. So **both stacks leak — they differ in which call shape leaks**, not in whether they
do. Saying "vLLM does the opposite" without that qualifier, as an earlier draft of this
document did, is wrong.

## New: transformers v5 made the leak silent

transformers v4.57 *warned* on this — `"generation_config default values have been
modified to match model-specific defaults: {...}"` — and offered an opt-out,
`use_model_defaults=False`.

On transformers 5.14.1, `grep` for both returns **zero occurrences**. The warning is
gone and so is the escape hatch. `_prepare_generation_config` now documents the priority
as `user kwargs > self.generation_config > global defaults` and merges with no notice.
The v5.0.0 release notes do not mention this.

Every existing writeup, including trl#5783, rests on a warning that no longer fires.
The fix for #42762 changed the failure mode rather than removing it, and the surviving
mode is quieter than the one people documented.

## New: how far from 1 is the importance ratio

On-policy RL assumes that for freshly sampled tokens the ratio
`pi_policy / pi_sampling` is exactly 1, and drops it. A penalty applied at rollout and
absent from the training forward pass breaks that.

**Why a repetition penalty is not the case already covered.** Every prior treatment —
DeepSeek's Keep Sampling Mask, Fang & Khazi, the verl/TRL correction machinery — is about
**top-p / top-k**, which is pure truncate-and-renormalise: in log space a near-constant
per-token shift equal to the retained probability mass, and repairable by replaying the
truncation mask during training. A repetition penalty is a different object. It is
**stateful and history-dependent**, reshaping the surviving distribution non-uniformly
according to what has already been generated. Keep Sampling Mask does not fix it, and
"just set top_p=1.0" does not name it.

Two consequences worth stating plainly. First, no public source quantifies the ratio a
repetition penalty induces. Second, vLLM's default `logprobs_mode` is `raw_logprobs` —
logprobs taken *before* any logit processor — so where a framework has not overridden it,
this bias is invisible to every existing TIS/MIS diagnostic. verl only defaulted to
`processed_logprobs` on 2025-12-31 ([PR #4755](https://github.com/verl-project/verl/pull/4755)),
and TRL only after [#4159](https://github.com/huggingface/trl/issues/4159).

`scripts/measure_importance_ratio.py` samples with `repetition_penalty=1.1` active and
everything else neutral, then scores the *same* tokens under plain `log_softmax` and
under transformers' own `RepetitionPenaltyLogitsProcessor` — the library's processor
rather than a reimplementation, so the sampling distribution is exactly what `generate`
used. 32 sequences, 8,209 completion tokens:

| statistic | value |
|---|---|
| tokens where ratio == 1 (the assumed value) | **14.1%** |
| tokens off by more than 1% | **58.8%** |
| tokens off by more than 10% | **37.7%** |
| median | 1.0002 |
| p1 / p99 | 0.131 / 2.959 |
| **geometric mean** | **0.895** |

The arithmetic mean is 1.0003, which is exactly why it must not be quoted alone — a thin
right tail offsets a fat left one. The geometric mean of 0.895 is the honest per-token
summary. The distortion is signed, not noisy: a repetition penalty suppresses tokens
already in context, so the sampler favours unseen tokens the policy scores lower.

### Where this sits against the known magnitudes

| source | cause | magnitude |
|---|---|---|
| Thinking Machines (2025-09) | kernel non-determinism, corrected | KL ~0.001 |
| Thinking Machines | bitwise-deterministic sampler/trainer | KL exactly 0 |
| Yao et al. (2025-08) | vLLM-vs-FSDP numerics, worst token | ratio up to 16 |
| Fang & Khazi (2025-12) | Qwen3 top_p/top_k/temperature | KL 0.001 → 0.033 (~33×) |
| **this document** | **`repetition_penalty=1.1` alone** | **geo-mean ρ 0.895; 37.7% of tokens >10% off** |

The comparison worth drawing is that a single inherited sampling field produces a
per-token distortion far larger than the BF16 kernel noise that the mismatch literature
was built to correct — and unlike that noise, it is signed rather than symmetric.

Scope note: mean sequence-level log-weight was −28.5 over ~257 tokens. Reported for
scale only, not as a correction factor — importance-ratio products degenerate
exponentially with length regardless of this bug.

## Prevalence across checkpoints

18 widely-used instruct checkpoints (`scripts/survey_generation_configs.py`, gated repos
excluded):

| field | non-neutral |
|---|---|
| `top_p` | 9 / 18 |
| `temperature` | 9 / 18 |
| `top_k` | 5 / 18 |
| `repetition_penalty` | 4 / 18 |

9 of 18 ship at least one. The `repetition_penalty` cases are Qwen 2/2.5 — **1.1** on
Qwen2.5-0.5B/1.5B-Instruct, **1.05** on Qwen2.5-7B-Instruct and Qwen2-7B-Instruct. Note
that `Qwen2.5-Math-7B-Instruct` ships a neutral config while the general 7B does not, and
Qwen3 dropped `repetition_penalty` entirely. Quote the value with the checkpoint; it is
not uniform across the family.

## Framework audit

11 frameworks at `main`, 2026-08-04. Verdicts require a file:line and quoted code.

| framework | verdict |
|---|---|
| **TRL** | SAFE. Pins `temperature`/`top_p`/`top_k`/`min_p`/`repetition_penalty` on both paths and overwrites `model.generation_config` (`grpo_trainer.py:1881`, citing #42762). Claims of broad TRL vulnerability are false. |
| **OpenRLHF** | SAFE incidentally. Zero references to `repetition_penalty`; its offline `SamplingParams` omits the field and is rescued by vLLM's neutral default. |
| **SkyRL** | SAFE, deliberately — sets `generation_config="vllm"` with an explanatory comment. |
| **open-instruct** | SAFE, deliberately — forces `generation_config="vllm"` at the only engine-construction site. |
| **torchtune** | SAFE / N-A. PPO uses a native sampler with no `repetition_penalty` concept. |
| **LLaMA-Factory** | SAFE. Pins every relevant field non-`None` in `generating_args`. |
| **unsloth**, **axolotl** | SAFE, but *inherited* from TRL rather than independent — recheck on a TRL major bump. |
| **veRL** | **BUG (HF rollout path).** `RolloutConfig` defines `repetition_penalty: float = 1.0` and the vLLM path honours it, but `hf_rollout.py:82-91` builds its `GenerationConfig` without it while reading `temperature`/`top_p`/`top_k` from that same config object. |
| **NeMo-RL** | **BUG (NeMo-Gym HTTP path).** `repetition_penalty` appears **nowhere** in the repo and no `generation_config="vllm"` is set, so the exposed-server rollout path inherits the checkpoint while the direct-engine path uses 1.0. The handler asserts `top_k`, `temperature` and `top_p` match — under a comment reading *"If they do not match, the inference will be off policy and destroy training stability"* — and omits `repetition_penalty`. |
| **trlx** | BUG, but dormant (last commit 2024-01-08) and predating the checkpoints that ship non-neutral configs. Listed for completeness, not as a live issue. |

**2 of 11 have a live bug.** Both are narrow: veRL's is a reference rather than
production rollout path, and NeMo-RL's affects only the exposed-HTTP-server
configuration. Neither is critical. Stating that plainly matters more than the finding.

The audit's more useful output is the *pattern*: SkyRL and open-instruct both close the
server vector explicitly and comment on why, which means the vector is known to people
who have hit it — and NeMo-RL, which guards three of the four fields, shows how easily
the remaining one slips through.

## Recommendation

Pin every sampling field explicitly, on every backend, rather than relying on either
library's notion of a default. If you serve rollouts over vLLM's HTTP server, also pass
`generation_config="vllm"`. This repo does the former via `NEUTRAL_SAMPLING` in
`src/rlv/rollout.py`, and verifies the backends agree with `scripts/compare_backends.py`
before quoting any throughput number.

## Reproducing

```bash
python scripts/audit_generation_config.py     # HF: explicit config still inherits
python scripts/audit_vllm_config.py           # vLLM: offline vs bare-call resolution
python scripts/measure_importance_ratio.py    # the ratio distribution
python scripts/survey_generation_configs.py   # prevalence across checkpoints
```
