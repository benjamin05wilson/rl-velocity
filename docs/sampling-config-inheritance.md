# Sampling parameters you did not set are not neutral

**Summary.** HuggingFace `generate` and vLLM resolve an *omitted* sampling parameter
differently. HF falls back to the checkpoint's `generation_config.json`; vLLM falls back
to its own neutral library defaults. For RL rollouts this is a correctness issue rather
than a preference, because on-policy methods require that the distribution you sample
from is the distribution you compute logprobs under.

Measured on `Qwen/Qwen2.5-0.5B-Instruct`, transformers 5.14.1, vLLM 0.26.0, sm_120.

---

## The checkpoint's defaults are not neutral

`Qwen2.5-0.5B-Instruct/generation_config.json`:

```json
{ "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1 }
```

These are serving defaults, tuned for chat quality — not for being a policy you
differentiate. And they are common. Surveying 18 widely-used instruct checkpoints
(`scripts/survey_generation_configs.py`, gated repos excluded):

| field | checkpoints shipping a non-neutral value |
|---|---|
| `top_p` | 9 / 18 |
| `temperature` | 9 / 18 |
| `top_k` | 5 / 18 |
| `repetition_penalty` | 4 / 18 |

**9 of 18 ship at least one.** The `repetition_penalty` cases are the entire Qwen 2/2.5
line (1.05–1.1) — the family most heavily used for maths RL. `top_p` and `temperature`
matter less in practice because every RL framework sets those explicitly; `top_k` and
`repetition_penalty` are the ones callers forget, which is precisely why they leak.

One telling detail: `Qwen2.5-Math-7B-Instruct` ships a **neutral** config while the
general-purpose `Qwen2.5-7B-Instruct` ships `repetition_penalty=1.05`. Whoever packaged
the maths variant appears to have known this mattered.

## HF: an explicit GenerationConfig still inherits

`GenerationConfig()`'s own default for `repetition_penalty` is `None`, not `1.0`. An
unset field therefore resolves against the model's config. Greedy decode, so any
difference is configuration rather than sampling noise (`scripts/audit_generation_config.py`):

| call | result |
|---|---|
| `GenerationConfig(do_sample=False)` — omits the field | 757 chars |
| `GenerationConfig(do_sample=False, repetition_penalty=1.0)` | 734 chars |
| `GenerationConfig(do_sample=False, repetition_penalty=1.1)` | 757 chars |

The omitted-field output is byte-identical to the checkpoint's 1.1 and differs from
neutral. **Passing an explicit `generation_config` does not isolate you from the
checkpoint.** TRL documents hitting this and patches around it by overwriting
`model.generation_config` outright, citing `transformers#42762`.

## vLLM: the opposite resolution

`scripts/audit_vllm_config.py`. The engine does read the checkpoint —
`default_sampling_params` resolves to
`{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}` — but an
explicitly constructed `SamplingParams` is used as-is:

| call | result |
|---|---|
| `llm.generate(prompts)` — no SamplingParams | 75 chars (uses checkpoint defaults) |
| `SamplingParams(temperature=0.0)` — omits the field | 713 chars |
| `SamplingParams(temperature=0.0, ..., repetition_penalty=1.0)` | 713 chars |
| `SamplingParams(temperature=0.0, ..., repetition_penalty=1.1)` | 757 chars |

Omitting the field gives the **neutral** result — the reverse of HF. Calling
`generate` with no `SamplingParams` at all does inherit the checkpoint.

## Why this matters for RL specifically

Two distinct problems.

**1. Cross-backend comparisons silently measure two things.** In this repo, comparing an
HF-`generate` rollout path against a vLLM path while overriding only `temperature` and
`top_p` produced mean reward 0.262 vs 0.406 — z = 3.51, comfortably beyond sampling
noise. After pinning `top_k`, `top_p` and `repetition_penalty` on both sides, the same
comparison gave 0.398 vs 0.406, z = 0.18, and greedy first-100-character agreement rose
from 31% to 97%. Any throughput result measured across that gap is uninterpretable.

**2. The policy gradient is computed against the wrong distribution.** A repetition
penalty applied during rollout but absent from the training forward pass means the
sampling distribution is not the policy distribution. The importance ratio GRPO and PPO
assume to be 1 for freshly sampled tokens is not 1. Nothing errors, and the reward curve
looks plausible.

On mathematical reasoning the penalty is not merely a mismatch but actively harmful:
correct arithmetic requires re-emitting digits, and that is precisely what a repetition
penalty suppresses. Removing it raised measured reward from 0.262 to 0.398 here.

### How far from 1 is the ratio, actually

`scripts/measure_importance_ratio.py` samples with `repetition_penalty=1.1` active and
everything else neutral, then scores the *same* tokens two ways: plain `log_softmax` of
the logits (what the trainer differentiates) and `log_softmax` after transformers' own
`RepetitionPenaltyLogitsProcessor` (what actually drew the token). Using the library's
processor rather than a reimplementation means the sampling distribution is exactly the
one `generate` used. 32 sequences, 8,209 completion tokens, Qwen2.5-0.5B-Instruct:

| statistic | value |
|---|---|
| tokens where ratio == 1 (the assumed value) | **14.1%** |
| tokens off by more than 1% | **58.8%** |
| tokens off by more than 10% | **37.7%** |
| median ratio | 1.0002 |
| p1 / p99 | 0.131 / 2.959 |
| min / max | 0.043 / 8.680 |
| **geometric mean** | **0.895** |

The median token is barely touched; the tail is not. The arithmetic mean sits at 1.0003,
which is exactly why it must not be quoted alone — a thin right tail offsets a fat left
one. The geometric mean of 0.895 is the honest per-token summary: roughly a 10%
systematic distortion, on a majority of tokens.

The distortion is signed, not noisy. A repetition penalty suppresses tokens already in
the context, so the sampler selects unseen tokens more often than the policy would;
those tokens carry ratio < 1, dragging the geometric mean below 1.

Note on scope: mean sequence-level log-weight came to −28.5 over ~257 tokens. That is
reported for scale only, not as a correction factor — products of importance ratios
degenerate exponentially with length regardless of this bug, which is why token-level
ratios are the ones that matter. The claim here is deliberately narrow: **the estimator
assumes a ratio of 1 that it does not have, on 86% of tokens.**

## Framework audit

Checked at `main` on 2026-08-04.

| framework | status |
|---|---|
| **TRL** | Safe. `GRPOConfig` defaults to `top_k=0`, `top_p=1.0`, `repetition_penalty=1.0`, passes all of them to both backends, and explicitly overwrites `model.generation_config` (`grpo_trainer.py:1881`, citing `transformers#42762`). 66 references to `repetition_penalty`. |
| **OpenRLHF** | Safe in practice, but only incidentally. Zero references to `repetition_penalty` anywhere; its vLLM `SamplingParams` (`ppo_utils/samples_generator.py:203`) omits the field and is rescued by vLLM's neutral default. An HF-based rollout path added later would inherit the checkpoint silently. |
| **veRL** | **Bug on the HF rollout path.** Pins `repetition_penalty=1.0` in its vLLM and agent-loop paths, and `RolloutConfig` defines `repetition_penalty: float = 1.0` (`workers/config/rollout.py:168`). But `HFRollout` builds its `GenerationConfig` from `do_sample`, `num_beams`, `top_p`, `top_k`, `temperature`, `num_return_sequences` only (`workers/rollout/hf_rollout.py:82-91`) — it receives that same config object and reads `temperature`/`top_p`/`top_k` from it, yet never reads `repetition_penalty`. The configured value is honoured on one backend and silently ignored on the other. |

### Severity, stated honestly

veRL's `HFRollout` is a reference/small-scale path; production runs use the vLLM or
SGLang rollouts, which are correct. So this is a real correctness bug with limited blast
radius, not a critical one. The fix is one line — read the field that already exists in
the config:

```python
kwargs = {
    "do_sample": True,
    "num_beams": 1,
    "top_p": top_p,
    "top_k": top_k,
    "temperature": temperature,
    "repetition_penalty": self.config.get("repetition_penalty", 1.0),  # add
    "num_return_sequences": 1,
}
```

## Recommendation

Pin every sampling field explicitly on every backend, rather than relying on either
library's notion of a default. This repo does so via `NEUTRAL_SAMPLING` in
`src/rlv/rollout.py`, and verifies the backends agree with
`scripts/compare_backends.py` before any throughput number is quoted.

## Reproducing

```bash
python scripts/audit_generation_config.py   # HF side
python scripts/audit_vllm_config.py         # vLLM side
python scripts/compare_backends.py --backend hf
python scripts/compare_backends.py --backend vllm
python scripts/compare_backends.py --backend report
```
