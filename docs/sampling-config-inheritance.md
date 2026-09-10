# Sampling configuration: an investigation with unresolved learning impact

**Status, 2026-09-10:** implementation and CPU regressions are inspectable. Original GPU
run records were not recovered. Earlier numerical tables and broad claims about novelty,
framework vulnerability and the wider literature have been removed. This note records
the mechanism investigated and the experimental limits; it is not a current framework audit.

## Mechanism and implementation

A generation API can obtain settings from a checkpoint's generation configuration.
The HF documentation describes that loading priority; vLLM documents its own
`generation_config="vllm"` opt-out. Defaults depend on the library version and API path.
See [HF generation configuration](https://huggingface.co/docs/transformers/main_classes/text_generation)
and [vLLM's explicit configuration example](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/quickstart.md).
These references explain configuration behavior, not results measured by this repository.

If rollout uses a repetition penalty but training scores unprocessed logits, they are
scoring different distributions. A penalty transforms logits based on the prefix. A
neutral top-p/top-k setting alone does not undo that transformation. Non-unit temperature
also requires matching logit scaling during scoring.

`src/rlv/rollout.py` replaces HF checkpoint generation defaults with a neutral configuration
and explicitly sets temperature, top-p/top-k and repetition penalty. The vLLM engine uses
`generation_config="vllm"` and explicit `SamplingParams`. Both expose a common padded batch;
prompt attention masks preserve genuine EOS tokens even when EOS is also the pad token.
Training uses explicit position IDs and a shifted completion mask. Positive temperatures
are applied to scoring logits; greedy decoding is evaluation-only. Deliberately nonneutral
penalty experiments require `--allow-off-policy` and are recorded as such.

`scripts/measure_importance_ratio.py` compares the same sampled tokens under raw and
HF repetition-processed logits. Its printed interpretation is calculated from the tokens
actually supplied, including the neutral case. It retains the first sampled EOS and masks
only padding. The metric alone says nothing about learning improvement.

## What a comparison can establish

The report in `scripts/compare_backends.py` separates configuration and prompt alignment,
exact text match, first-100-character match, exact token match when available, and reward
differences. Its descriptive standard error uses paired **prompt-level** reward differences,
not independent-completion binomial errors. The normal interval is a rough descriptive
summary, especially unreliable with few prompts or boundary rates. No equivalence margin,
power calculation or acceptance threshold was predeclared; the script never issues an
unconditional equivalence verdict.

The [offline fixture](../evidence/comparison/report.json) deliberately matches aggregate
rewards while diverging on every output. It demonstrates a reporting safeguard, not backend
performance. Even identical finite outputs would not prove distributional equivalence.

## Positive control and stop decision

The historical investigation notes describe an attempted neutral-sampling learning-rate
control that did not show a convincing held-out gain. Original evaluation logs and exact
checkpoint/environment records are unavailable, so this observation is **unverified history**.
There is no retained accuracy table and no claim that the model cannot learn.

Without a successful control, an A/B learning-impact conclusion would be unsupported.
Future work would require archived baseline evaluations, fixed data/checkpoint revisions,
multiple independent seeds, a predeclared uncertainty treatment and a compute-matched
comparison. That experiment is outside the current readiness work.

CPU tests cover masks, controlled-logit scoring, temperature and loss reduction, and the
comparison/diagnostic reporting rules. They do not validate HF/vLLM numerical parity,
weight synchronization on a GPU, a stable importance correction, or successful learning.
See [reproduction and environment limits](reproduction.md).
