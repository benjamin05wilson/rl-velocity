"""Print raw completions and how they graded, for either backend.

A reward of exactly zero for every sample is far more often a harness bug than a
genuinely incapable model, and the only way to tell the two apart is to read what the
model actually emitted. Kept as a standing tool rather than a one-off: every future
reward anomaly gets checked here first.
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlv.tasks import gsm8k
from rlv.train import build_prompt

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = gsm8k.load("train", limit=args.prompts)
    prompts = [build_prompt(tok, r["question"]) for r in data]

    if args.backend == "vllm":
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from rlv.rollout import VLLMRollout

        backend = VLLMRollout(args.model, tok, gpu_frac=0.45, max_model_len=args.max_new_tokens + 512)
    else:
        from rlv.rollout import HFRollout

        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        backend = HFRollout(model, tok)

    print("=" * 70)
    print("PROMPT AS THE MODEL SEES IT")
    print("=" * 70)
    print(repr(prompts[0][:400]))

    rb = backend.generate(prompts, args.group_size, args.max_new_tokens, args.temperature)
    lengths = rb.lengths.reshape(-1)

    n_ok = n_fmt = n_capped = 0
    for i, text in enumerate(rb.texts):
        gold = data[i // args.group_size]["gold"]
        g = gsm8k.grade(text, gold)
        n_ok += int(g.correct)
        n_fmt += int(g.format_ok)
        n_capped += int(int(lengths[i]) >= args.max_new_tokens)
        if i < 2:
            print("\n" + "=" * 70)
            print(f"completion {i}  ({int(lengths[i])} tokens)")
            print("=" * 70)
            print(text[-700:] if len(text) > 700 else text)
            print(f"  -> gold={gold} predicted={g.predicted} boxed={g.format_ok} reward={g.reward}")

    n = len(rb.texts)
    print(f"\n[{args.backend}] correct={n_ok}/{n}  boxed={n_fmt}/{n}  hit_token_cap={n_capped}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
