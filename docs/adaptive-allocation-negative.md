# Adaptive rollout allocation: tested, not supported, and already invented

**Summary.** In GRPO, groups whose completions all score identically produce zero
gradient. Steering generation budget away from them looks like free throughput. It was
tested here and did not hold up — and the idea was published as
[GRESO](https://arxiv.org/abs/2506.02177) in June 2025, with a substantial literature
since. This document records both, because a tested-and-rejected idea is worth more
written down than deleted.

## What was measured

Weights frozen, 256 GSM8K prompts x 8 completions, 5 independent rounds
(`scripts/measure_degeneracy_predictability.py`):

| | |
|---|---|
| base degenerate rate | 27.7% of groups |
| P(degenerate \| degenerate last round) | 53.1% |
| P(degenerate \| informative last round) | 17.1% |
| lift over base rate | **1.91x** |

Degeneracy is a property of the prompt, not only of the draw. But the obvious policy is
a poor trade: skipping on one round of history avoids 16% of generation while discarding
18.4% of informative groups.

## The principled version, and why it fails

For a prompt with pass rate `p`, a group of size `G` is informative unless every
completion lands on one side:

```
P(informative | p, G) = 1 - p^G - (1-p)^G
```

This is concave in `G`, so greedy allocation by marginal gain is provably optimal — no
learned predictor needed. `p` estimated on early rounds, scored on held-out later rounds
(`scripts/simulate_allocation.py`), at equal total spend:

| objective | gain over uniform | oracle ceiling |
|---|---|---|
| informative groups | **0.999x** | 1.062x |
| total advantage magnitude | 1.443x | 1.620x |

**The disagreement is the result.**

Counting informative groups shows no gain and almost no headroom — even a perfect
allocator reaches 1.062x. The reason is structural: `1 - p^G - (1-p)^G` is flat across
most of the `p` range at `G=8`. Large groups are already robust to prompt difficulty, and
only genuinely extreme prompts degenerate — 16% of the set here. *The concavity that
makes the problem tractable is the same property that makes it not worth solving.*

The advantage-magnitude gain is an artefact. That objective is linear in `G`, so its
optimum is bang-bang: it assigned `G=32` to 51 prompts and `G=2` to 204, collapsing
prompt diversity to a fifth of uniform. It also assumes signal grows linearly with group
size, which is false — advantages are centred within a group, so additional samples of
the same prompt are correlated and yield diminishing returns the model ignores.

Neither proxy is trustworthy. They bracket the answer rather than settle it.

## Why this null does not contradict the literature

GRESO reports `P(zero-variance now | zero-variance previously) > 90%`. This measured
53.1%. The gap is regime, not contradiction:

- **Frozen weights, early training.** GRESO measures during training, where prompts
  progressively become permanently solved or permanently unsolved. Persistence rises as
  the policy converges.
- **Degeneracy rate.** GRESO's effective-prompt ratio falls to roughly 20% (about 80%
  degenerate) later in training. At 27.7% there is far less to reclaim.
- **Model scale.** Qwen2.5-0.5B at ~42% mean pass rate sits close to the maximally
  informative region, which is precisely where uniform allocation is hardest to beat.

**The experiment measured the regime with the least available gain.** That is the honest
lesson: these methods pay off late in training on stronger models, not early on a small
one.

## Prior art

The idea is not new, and the framing in GPU-hours is not a differentiator.

| work | date | relation |
|---|---|---|
| [DAPO](https://arxiv.org/abs/2503.14476) §3.2 | 2025-03 | Reactive: oversample, discard zero-variance groups. **Argues the discarded rollouts are largely free**, since synchronous generation time is dominated by long-tail sequence length. |
| [Online Difficulty Filtering](https://arxiv.org/abs/2504.03380) | 2025-04 | Proves expected policy improvement is lower-bounded by variance of task success probabilities — the theory this idea rests on. |
| [GVM-RAFT](https://arxiv.org/abs/2505.02391) | 2025-05 | Prompt-specific dynamic allocation minimising gradient variance under a compute budget. |
| **[GRESO](https://arxiv.org/abs/2506.02177)** | **2025-06** | **This idea, exactly.** Pre-rollout skip predicted from reward training dynamics; 2.4x rollout speedup, 2.0x total, framed in wall-clock. |
| [MoPPS](https://arxiv.org/abs/2507.04632) | 2025-07 | Bayesian bandit predicting difficulty with no rollouts; ~78% fewer rollouts. |
| [RL-ZVP](https://arxiv.org/abs/2509.21880) | 2025-09 | **Contests the premise** — extracts gradient *from* zero-variance prompts, +8.61 accuracy. |
| [Reinforce-Ada](https://arxiv.org/abs/2510.04996) | 2025-10 | Adaptive sampling that costs **1.59–2.8x more wall-clock per step**. |
| [VADE](https://arxiv.org/abs/2511.18902) | 2025-11 | Beta posterior + Thompson sampling on `p(1-p)^2`; ~3x fewer inferences. |

Dynamic sampling is also a shipped feature: verl (`algorithm.filter_groups`), OpenRLHF
(`--dynamic_filtering`), NeMo-RL (`use_dynamic_sampling`), ms-swift (`--dynamic_sample`),
slime. TRL declined it ([#4764](https://github.com/huggingface/trl/issues/4764), closed
as not planned).

## The idea is not just prior art — it is actively contested and losing

The null measured here is not a small-scale artefact. Several papers report that
filtering zero-variance groups **underperforms doing nothing at all**.

**[RL-ZVP](https://arxiv.org/abs/2509.21880) (2025-09) is the sharpest.** At matched
rollout budget, both filtering families lose to plain GRPO — not marginally:

| Acc@8 | MATH500 | AIME24 | Olympiad |
|---|---|---|---|
| GRPO | 83.00 | 28.33 | 49.59 |
| GRPO + DAPO dynamic sampling | 68.20 | 12.08 | 43.88 |
| GRESO | 67.40 | 12.08 | 44.92 |

Gradient-matched instead of rollout-matched, DAPO-DS needs 2.45–5.29x and GRESO
1.58–3.99x more rollouts, and both still lose. RL-ZVP's own method goes the opposite
direction — it extracts gradient *from* zero-variance groups via entropy-guided advantage
shaping, gaining up to +8.61 accuracy.

Corroborating:

- **[Comparative analysis of PPO/GRPO/DAPO](https://arxiv.org/html/2512.07611v1)** (2025-12): dynamic sampling reaches a *higher surrogate objective* with no true-metric gain, and after the accuracy peak around step 75 it actively hinders improvement. Costs >25% extra per step.
- **[P²O](https://arxiv.org/html/2603.21877v3)** (2026-05): DAPO underperforms plain GRPO by 3.2 points on DeepMath-5K.
- **[KGPS](https://arxiv.org/html/2607.27610)** (2026-07): GRESO assumes stationary difficulty, which RL is not; "persistently high estimation error throughout training", worst early when the policy moves fastest. 28.85 vs 31.70 at identical rollout budget.
- **[GPS](https://arxiv.org/html/2602.01970v2)** (2026-05): GRESO below the no-filtering baseline on AIME24 (31.5 vs 32.4).
- **[DARS](https://arxiv.org/html/2508.13755v8)**: group advantage `2Nu(1-u)` peaks at u=0.5, so the optimiser already over-weights medium difficulty; filtering compounds a bias rather than correcting one. Allocating *more* to hard prompts gains +2.0 Pass@128.

## The premise itself was wrong

The deepest problem is not the allocator. It is the assumption this repo started from —
that degenerate groups waste GPU-seconds proportional to their tokens.

In synchronous RL, a generation step ends when its **slowest sequence** finishes.
Generation length is heavy-tailed, so wall-clock is set by one straggler decoding at
batch size 1 — roughly a tenth of saturated throughput — while everything else waits.
Removing tokens from the middle of that distribution frees capacity that was not the
bottleneck. DAPO says exactly this in §3.2 and it is the reason the paper does not
consider its discarded rollouts expensive.

Practitioner measurement agrees: [UniRL#94](https://github.com/Tencent-Hunyuan/UniRL/issues/94)
concludes dynamic sampling is "a data-efficiency lever, not throughput". Related work
([Beat the Long Tail](https://arxiv.org/abs/2511.13841),
[Straggler-Aware Group Sizing](https://arxiv.org/abs/2606.02218)) confirms rollout is
>70% of runtime and a small fraction of long generations dominates it.

**So "27.7% of groups are degenerate" was never equivalent to "27.7% of GPU-seconds are
recoverable."** The correct framing of this repo's own headline metric is that
`wasted_token_frac` measures wasted *tokens*, and tokens do not convert to wall-clock at
a fixed rate under straggler-bound generation. That is a correction to how the
instrumentation was motivated on day one.

## What remains genuinely open

Not the method. Possibly the evaluation:

**No compute-matched head-to-head exists.** GRESO, MoPPS, VADE, HIVE, SARA, HORA and
VIGOR each use different models, datasets and baselines, and most report **rollout
counts** rather than **GPU-seconds**. DAPO's long-tail argument implies those are not
interchangeable — if generation is gated by the longest sequence, cutting rollout count
need not cut wall-clock, and Reinforce-Ada is a documented case where adaptivity made
wall-clock worse.

Measuring where rollout-count savings do and do not convert into GPU-second savings, on
one stack under an identical budget, would be a useful contribution. It is a benchmark
contribution rather than a method one, and it needs more hardware than a single 24GB
card.
