"""Isolate why the training loop scores zero when direct generation does not.

Two differences between the working path and the training path: temperature (0.7 vs
1.0), and the model being in train mode with gradient checkpointing enabled during
rollout. Test them separately rather than changing both and declaring victory.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlv.tasks import gsm8k
from rlv.train import build_prompt, rollout

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def trial(model, tok, prompts, data, label: str, temperature: float, max_new: int = 256):
    seqs, plen = rollout(model, tok, prompts, group_size=4, max_new=max_new, temperature=temperature)
    texts = tok.batch_decode(seqs[:, plen:], skip_special_tokens=True)
    n_ok = n_fmt = n_capped = 0
    for i, t in enumerate(texts):
        g = gsm8k.grade(t, data[i // 4]["gold"])
        n_ok += int(g.correct)
        n_fmt += int(g.format_ok)
        n_capped += int(int((seqs[i, plen:] != tok.pad_token_id).sum()) >= max_new)
    n = len(texts)
    print(f"{label:38s} correct={n_ok}/{n}  boxed={n_fmt}/{n}  hit_token_cap={n_capped}/{n}")
    return texts


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

    data = gsm8k.load("train", limit=2)
    prompts = [build_prompt(tok, r["question"]) for r in data]

    torch.manual_seed(0)
    model.eval()
    trial(model, tok, prompts, data, "eval, temp=0.7", 0.7)
    trial(model, tok, prompts, data, "eval, temp=1.0", 1.0)

    model.gradient_checkpointing_enable()
    model.train()
    texts = trial(model, tok, prompts, data, "train + grad-ckpt, temp=1.0", 1.0)

    model.gradient_checkpointing_disable()
    model.eval()
    trial(model, tok, prompts, data, "eval again (checkpointing off), temp=1.0", 1.0)

    print("\n--- sample from train + grad-ckpt path ---")
    print(repr(texts[0][:400]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
