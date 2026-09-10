# Adaptive allocation: completed proxy investigation, no demonstrated training gain

**Status, 2026-09-10:** historical notes describe a completed frozen-policy allocation
simulation. Original pass-count archives are unavailable. Earlier numerical results are
withdrawn; this note retains the reasoning and the decision not to add a training scheduler.
The allocation script is exploratory source, not evidence of a positive training result.

## The proxy problem

For independent binary outcomes with fixed pass probability `p`, a group of `G` samples
is informative when it contains both a success and a failure:

```text
P(informative | p, G) = 1 - p**G - (1-p)**G
```

`scripts/simulate_allocation.py` estimates prompt-level pass probabilities on earlier
rounds and evaluates allocation on held-out rounds. With at least one sample per prompt,
the informative-group objective has decreasing marginal gains, enabling a greedy
allocation under integer bounds and a fixed completion budget. The script's default
minimum is two samples; infeasible budgets are rejected.

A second proxy, expected summed absolute centered advantage, is proportional to
`2 * (G-1) * p * (1-p)`. It can favor concentrating the budget on fewer prompts. These
objectives optimize different quantities, and neither is held-out learning improvement.
Estimated probabilities introduce further error; a frozen-policy result does not establish
behavior when the policy changes during training.

The historical notes reported disagreement between the proxies and did not justify
building an adaptive training feature. Without the original archive, that remains an
unverified experimental observation rather than a reproduced numerical result.

## Correction to the original motivation

`wasted_token_frac` counts generated tokens in groups with zero relative advantage.
It does **not** measure wasted GPU time. Runtime depends on batch occupancy, sequence
lengths, scheduling and synchronization; deleting some tokens may leave the critical
path unchanged. A token-count reduction cannot be converted into a wall-clock speedup
without measuring it. This correction also applies to the harness's degenerate-group metric.

No novelty claim or assertion that nobody has evaluated this problem is made here.
This repository does not contain a compute-matched survey, a successful scheduler, or
recovered raw simulation inputs. A future experiment would need immutable inputs,
training evaluations, explicit allocation constraints and synchronized timing under
matched budgets. Bigger models and adaptive scheduling are outside current scope.

See the [sampling/positive-control limits](sampling-config-inheritance.md),
[synthetic replay manifest](../evidence/MANIFEST.md), and [reproduction index](reproduction.md).
