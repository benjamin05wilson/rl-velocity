"""Is a degenerate group predictable, or is it a coin flip?

This gates the whole adaptive-allocation idea. In GRPO a prompt whose G completions all
score identically produces zero advantage for every one of them: the tokens were
generated, paid for in GPU-seconds, and taught the model nothing. Measured at 29-37% of
groups in this repo's runs.

Reallocating budget away from those prompts only works if degeneracy is a property of
the *prompt* rather than a property of the *draw*. If each rollout is an independent
coin flip, no predictor can beat random selection and the idea is dead before it is
built. So measure that first, cheaply, before writing an allocator.

Method: freeze the weights, sample the same prompts R independent times, and ask whether
round r predicts round r+1. No training -- this isolates prompt difficulty from policy
drift. That is also the limitation: during real training the policy moves, so
predictability measured here is an upper bound on what an online allocator could exploit.

Reports the honest trade: compute saved against informative groups lost.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompts", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out", default="runs/degeneracy")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from rlv.rollout import VLLMRollout
    from rlv.tasks import gsm8k
    from rlv.train import build_prompt

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = gsm8k.load("train", limit=args.prompts)
    prompts = [build_prompt(tok, r["question"]) for r in data]
    backend = VLLMRollout(args.model, tok, gpu_frac=0.85,
                          max_model_len=args.max_new_tokens + 512)

    G = args.group_size
    # passes[r][i] = number of correct completions for prompt i in round r
    passes: list[list[int]] = []
    tokens: list[list[int]] = []

    for r in range(args.rounds):
        rb = backend.generate(prompts, G, args.max_new_tokens, 1.0)
        lens = rb.lengths.reshape(-1).tolist()
        row_p, row_t = [], []
        for i in range(len(prompts)):
            n_ok = sum(
                gsm8k.grade(rb.texts[i * G + j], data[i]["gold"]).correct for j in range(G)
            )
            row_p.append(n_ok)
            row_t.append(sum(lens[i * G: (i + 1) * G]))
        passes.append(row_p)
        tokens.append(row_t)
        degen = sum(1 for p in row_p if p in (0, G))
        print(f"round {r}: degenerate {degen}/{len(prompts)} ({degen / len(prompts):.1%})  "
              f"mean pass {sum(row_p) / len(row_p) / G:.3f}")

    n = len(prompts)
    R = args.rounds

    def is_degen(r: int, i: int) -> bool:
        return passes[r][i] in (0, G)

    base = sum(is_degen(r, i) for r in range(R) for i in range(n)) / (R * n)

    # --- persistence: does degeneracy carry from one round to the next? ---
    dd = dn = nd = nn = 0
    for r in range(R - 1):
        for i in range(n):
            a, b = is_degen(r, i), is_degen(r + 1, i)
            if a and b:
                dd += 1
            elif a and not b:
                dn += 1
            elif not a and b:
                nd += 1
            else:
                nn += 1
    p_d_given_d = dd / (dd + dn) if (dd + dn) else 0.0
    p_d_given_n = nd / (nd + nn) if (nd + nn) else 0.0

    # --- how stable is the pass count itself? ---
    import statistics
    per_prompt_sd = [statistics.pstdev([passes[r][i] for r in range(R)]) for i in range(n)]
    always_degen = sum(1 for i in range(n) if all(is_degen(r, i) for r in range(R)))
    never_degen = sum(1 for i in range(n) if not any(is_degen(r, i) for r in range(R)))

    # --- what could an allocator actually save? ---
    # Oracle: skip every group that turns out degenerate. Unattainable, gives the ceiling.
    # History predictor: skip round r+1 if round r was degenerate. Attainable online.
    tot_tok = sum(sum(tokens[r]) for r in range(R))
    oracle_saved = sum(tokens[r][i] for r in range(R) for i in range(n) if is_degen(r, i))

    pred_saved = pred_lost_groups = pred_skipped = 0
    for r in range(1, R):
        for i in range(n):
            if is_degen(r - 1, i):
                                # predictor says skip
                pred_skipped += 1
                if is_degen(r, i):
                    pred_saved += tokens[r][i]     # correctly avoided waste
                else:
                    pred_lost_groups += 1          # wrongly dropped an informative group
    later_tok = sum(sum(tokens[r]) for r in range(1, R))
    informative_later = sum(1 for r in range(1, R) for i in range(n) if not is_degen(r, i))

    print("\n" + "=" * 66)
    print("DEGENERACY")
    print("=" * 66)
    print(f"  base rate                       {base:.1%} of groups")
    print(f"  always degenerate (all {R} rounds) {always_degen}/{n} ({always_degen / n:.1%})")
    print(f"  never degenerate                {never_degen}/{n} ({never_degen / n:.1%})")
    print(f"  mean per-prompt sd of pass count {sum(per_prompt_sd) / n:.2f} (of {G})")

    print("\n" + "=" * 66)
    print("PREDICTABILITY  (does last round predict this round?)")
    print("=" * 66)
    print(f"  P(degenerate)                    {base:.1%}   <- base rate to beat")
    print(f"  P(degenerate | degen last round) {p_d_given_d:.1%}")
    print(f"  P(degenerate | informative last) {p_d_given_n:.1%}")
    lift = p_d_given_d / base if base else 0.0
    print(f"  lift over base rate              {lift:.2f}x")

    print("\n" + "=" * 66)
    print("WHAT AN ALLOCATOR COULD SAVE")
    print("=" * 66)
    print(f"  oracle ceiling (skip all degenerate)  {oracle_saved / tot_tok:.1%} of generation")
    print("  1-round-history predictor:")
    print(f"    generation avoided                  {pred_saved / later_tok:.1%}")
    print(f"    informative groups wrongly dropped  {pred_lost_groups}/{informative_later} "
          f"({pred_lost_groups / informative_later:.1%})")
    print(f"    precision of the skip decision      "
          f"{pred_saved and (pred_skipped - pred_lost_groups) / pred_skipped:.1%}")

    print()
    if lift < 1.15:
        print("VERDICT: degeneracy is close to a coin flip. Prompt history carries little")
        print("signal, so an allocator keyed on it cannot beat random. Idea does not survive.")
    else:
        print("VERDICT: degeneracy persists across rounds, so it is a property of the prompt")
        print("and not only of the draw. An allocator has something real to exploit -- the")
        print("open question is whether dropping informative groups costs more than the")
        print("generation it saves.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "passes.json").write_text(json.dumps({"passes": passes, "tokens": tokens, "G": G}))
    print(f"\nraw counts -> {out / 'passes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
