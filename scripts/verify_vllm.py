"""Second toolchain gate: does vLLM actually generate on Blackwell?

torch working on sm_120 does not imply vLLM does. vLLM ships its own attention and
sampling kernels, and new architectures are exactly where those lag. This is the
gate that decides whether the fast-rollout plan is viable at all, so it runs before
any training code is written rather than after.

Also establishes the baseline that Phase 2 has to beat: generation throughput at a
GPU fraction small enough to leave room for a trainer on the same card, because on
24GB the policy and the inference engine have to cohabit.
"""

from __future__ import annotations

import argparse
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Natalia sold clips to 48 friends in April, then half as many in May. How many clips did she sell altogether?",
    "A robe takes 2 bolts of blue fibre and half that much white fibre. How many bolts total?",
    "Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes. How much did she earn?",
    "Betty is saving for a $100 wallet. She has half, her parents give $15, her grandparents twice that. How much more does she need?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-frac", type=float, default=0.45,
                    help="deliberately low: the trainer needs the rest of the card")
    ap.add_argument("--n", type=int, default=8, help="completions per prompt (GRPO group size)")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    print(f"loading {args.model} at gpu_memory_utilization={args.gpu_frac} ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1024,
        dtype="bfloat16",
        enforce_eager=False,
    )
    load_s = time.perf_counter() - t0
    print(f"engine ready in {load_s:.1f}s")

    sampling = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=0)

    # Warm the engine: the first batch pays for graph capture and allocator growth,
    # and folding that into the measurement would flatter every later comparison.
    llm.generate(PROMPTS[:1], SamplingParams(n=1, max_tokens=16), use_tqdm=False)

    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sampling, use_tqdm=False)
    dt = time.perf_counter() - t0

    gen_tokens = sum(len(c.token_ids) for o in outs for c in o.outputs)
    n_seqs = sum(len(o.outputs) for o in outs)

    print("\n--- baseline ---")
    print(f"  sequences:        {n_seqs}  ({len(PROMPTS)} prompts x n={args.n})")
    print(f"  generated tokens: {gen_tokens}")
    print(f"  wall:             {dt:.2f}s")
    print(f"  throughput:       {gen_tokens / dt:.0f} tok/s")
    print(f"  per-sequence:     {gen_tokens / n_seqs:.0f} tokens avg")

    print("\n--- sample completion ---")
    print(f"  Q: {PROMPTS[0]}")
    print(f"  A: {outs[0].outputs[0].text.strip()[:300]}")

    if gen_tokens == 0:
        print("\nBLOCKED -- engine produced no tokens")
        return 1
    print("\nOK -- vLLM generates on sm_120.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
