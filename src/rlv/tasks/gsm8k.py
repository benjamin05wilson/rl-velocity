"""GSM8K with a verifiable reward.

Grade-school maths is the right first task here for one reason: the reward is a string
comparison against a known answer, so there is no reward model to train, host, or keep
in GPU memory. On 24GB that is not a stylistic preference -- a reward model would not fit
alongside the policy and the inference engine.

It also makes the waste signal legible. Reward is binary, so a group is degenerate
exactly when its completions are all right or all wrong, with no ambiguity about what
counts as zero advantage.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# GSM8K gold answers are terminated by a #### marker.
_GOLD = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
# Accept \boxed{...} first -- if the model has been asked for it, honouring the format
# rewards instruction-following rather than luck.
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")

PROMPT_TEMPLATE = (
    "Solve the problem step by step, then give the final answer "
    "in \\boxed{{}}.\n\nProblem: {question}"
)


def normalise(text: str) -> str | None:
    """Reduce a numeric answer to a comparable canonical form."""
    if text is None:
        return None
    t = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not t:
        return None
    try:
        val = float(t)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    # Integers and integral floats must compare equal: "72" == "72.0" == "72.00".
    return str(int(val)) if val == int(val) else str(val)


def extract_gold(answer_field: str) -> str | None:
    m = _GOLD.search(answer_field)
    return normalise(m.group(1)) if m else None


def extract_prediction(completion: str) -> str | None:
    """Prefer the boxed answer; fall back to the last number mentioned.

    The fallback is deliberate but not free -- it will occasionally credit a model that
    stumbled onto the right number without following the format. `format_ok` is recorded
    separately so that effect stays visible instead of hiding inside the reward.
    """
    boxed = _BOXED.findall(completion)
    if boxed:
        norm = normalise(boxed[-1])
        if norm is not None:
            return norm
    numbers = _NUMBER.findall(completion)
    return normalise(numbers[-1]) if numbers else None


@dataclass
class Grade:
    reward: float
    correct: bool
    format_ok: bool
    predicted: str | None
    gold: str | None


def grade(completion: str, gold: str | None) -> Grade:
    pred = extract_prediction(completion)
    boxed = _BOXED.findall(completion)
    fmt = bool(boxed) and normalise(boxed[-1]) is not None
    correct = pred is not None and gold is not None and pred == gold
    return Grade(
        reward=1.0 if correct else 0.0,
        correct=correct,
        format_ok=fmt,
        predicted=pred,
        gold=gold,
    )


def load(split: str = "train", limit: int | None = None, revision: str | None = None) -> list[dict]:
    """Return [{question, gold, prompt}] with unparseable rows dropped."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split, revision=revision)
    rows: list[dict] = []
    for row in ds:
        gold = extract_gold(row["answer"])
        if gold is None:
            continue  # a gold answer we cannot parse would silently score everything wrong
        rows.append({
            "question": row["question"],
            "gold": gold,
            "prompt": PROMPT_TEMPLATE.format(question=row["question"]),
        })
        if limit and len(rows) >= limit:
            break
    return rows
