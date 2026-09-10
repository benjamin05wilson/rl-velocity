"""How many popular instruct checkpoints ship non-neutral sampling defaults?

The inheritance bug only matters in proportion to how often checkpoints carry values
worth inheriting. If Qwen were unusual this would be a footnote; if it is typical, then
every HF-`generate` rollout against an instruct checkpoint is sampling from a distribution
the trainer does not know about.

Downloads only generation_config.json -- a few hundred bytes per model, no weights.
"""

from __future__ import annotations

import json
from pathlib import Path

# Fields that alter the sampled distribution. A checkpoint setting any of these to a
# non-neutral value hands it to HF `generate` for any field the caller omits.
NEUTRAL = {
    "repetition_penalty": 1.0,
    "top_k": (0, None),          # 0 or None both mean "disabled"
    "top_p": 1.0,
    "temperature": 1.0,
    "no_repeat_ngram_size": (0, None),
    "length_penalty": 1.0,
}

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/deepseek-math-7b-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/phi-4",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "01-ai/Yi-1.5-6B-Chat",
    "tiiuae/Falcon3-7B-Instruct",
]


def is_neutral(field: str, value) -> bool:
    expected = NEUTRAL[field]
    if isinstance(expected, tuple):
        return value in expected
    return value == expected


def main() -> int:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    rows, failed = [], []
    for repo in MODELS:
        try:
            path = hf_hub_download(repo, "generation_config.json")
            cfg = json.loads(Path(path).read_text())
        except EntryNotFoundError:
            rows.append((repo, {}, []))  # no file at all -> nothing to inherit
            continue
        except Exception as exc:  # gated, offline, renamed
            failed.append((repo, type(exc).__name__))
            continue

        offenders = [
            (f, cfg[f]) for f in NEUTRAL if f in cfg and not is_neutral(f, cfg[f])
        ]
        rows.append((repo, cfg, offenders))

    print(f"{'checkpoint':<45}{'non-neutral sampling fields shipped'}")
    print("-" * 100)
    n_bad = 0
    for repo, cfg, offenders in rows:
        if not cfg:
            print(f"{repo:<45}(no generation_config.json)")
            continue
        if offenders:
            n_bad += 1
            desc = "  ".join(f"{f}={v}" for f, v in offenders)
        else:
            desc = "-- neutral --"
        print(f"{repo:<45}{desc}")

    checked = len(rows)
    print()
    print(f"{n_bad}/{checked} checkpoints ship at least one non-neutral sampling default")

    # Per-field tally: which parameter is the common offender?
    tally: dict[str, int] = {}
    for _, _, offenders in rows:
        for f, _ in offenders:
            tally[f] = tally.get(f, 0) + 1
    if tally:
        print("\nby field:")
        for f, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {f:<24} {n}/{checked}")

    # repetition_penalty is the one that breaks the on-policy assumption, because
    # unlike top_k/top_p it reshapes probabilities based on generated history.
    rp = sum(1 for _, cfg, _ in rows if cfg.get("repetition_penalty", 1.0) not in (1.0, None))
    print(f"\n{rp}/{checked} ship a repetition_penalty != 1.0")

    if failed:
        print(f"\nnot retrieved ({len(failed)}): " + ", ".join(f"{r} [{e}]" for r, e in failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
