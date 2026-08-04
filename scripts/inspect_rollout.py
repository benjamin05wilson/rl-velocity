"""Print raw completions and how they graded.

A reward of exactly zero for every sample is far more often a harness bug than a
genuinely incapable model, and the only way to tell the two apart is to read what the
model actually emitted.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlv.tasks import gsm8k
from rlv.train import build_prompt, rollout

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    data = gsm8k.load("train", limit=2)
    prompts = [build_prompt(tok, r["question"]) for r in data]

    print("=" * 70)
    print("PROMPT AS THE MODEL SEES IT")
    print("=" * 70)
    print(repr(prompts[0]))

    for max_new in (256, 512):
        seqs, plen = rollout(model, tok, prompts, group_size=2, max_new=max_new, temperature=0.7)
        texts = tok.batch_decode(seqs[:, plen:], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print(f"max_new_tokens={max_new}")
        print("=" * 70)
        for i, t in enumerate(texts[:2]):
            gold = data[i // 2]["gold"]
            g = gsm8k.grade(t, gold)
            ntok = int((seqs[i, plen:] != tok.pad_token_id).sum())
            print(f"\n--- completion {i}  ({ntok} tokens, hit_limit={ntok >= max_new}) ---")
            print(t[-600:] if len(t) > 600 else t)
            print(f"  -> gold={gold} predicted={g.predicted} format_ok={g.format_ok} reward={g.reward}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
