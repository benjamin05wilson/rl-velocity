"""Are the two rollout backends actually sampling from the same policy?

A 4x speedup is only a speedup if the fast path produces the same distribution. If vLLM
quietly generates better completions -- different sampler, different numerics, different
prompt handling -- then the comparison measures two things at once and the throughput
claim is unusable.

Two checks, because they fail differently:

  greedy  -- temperature 0 removes sampling noise entirely. Any divergence here is a
             real implementation difference (tokenisation, numerics, prompt handling),
             not chance. This is the sharp test.
  sampled -- at temperature 1 over many prompts, reward rates should agree within
             sampling error. This catches distributional skew that greedy decoding
             would hide.

Run the backends in separate processes; both want most of the card.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlv.tasks import gsm8k
from rlv.train import build_prompt

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def run(backend_name: str, args) -> dict:
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = gsm8k.load("train", limit=args.prompts)
    prompts = [build_prompt(tok, r["question"]) for r in data]

    if backend_name == "vllm":
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from rlv.rollout import VLLMRollout

        backend = VLLMRollout(args.model, tok, gpu_frac=0.55, max_model_len=args.max_new_tokens + 512)
    else:
        from rlv.rollout import HFRollout

        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        backend = HFRollout(model, tok)

    out: dict = {"backend": backend_name}

    # --- greedy ---------------------------------------------------------------
    # temperature <= 0 selects argmax in both backends. Approximating greedy with a
    # tiny temperature instead would divide the logits by ~1e-7 and overflow, which
    # produces garbage and looks exactly like a real divergence.
    torch.manual_seed(0)
    rb = backend.generate(prompts, 1, args.max_new_tokens, 0.0)
    out["greedy_texts"] = rb.texts
    out["greedy_rewards"] = [
        gsm8k.grade(t, data[i]["gold"]).reward for i, t in enumerate(rb.texts)
    ]

    # --- sampled --------------------------------------------------------------
    torch.manual_seed(0)
    rb = backend.generate(prompts, args.group_size, args.max_new_tokens, 1.0)
    rewards, fmt = [], 0
    for i, t in enumerate(rb.texts):
        g = gsm8k.grade(t, data[i // args.group_size]["gold"])
        rewards.append(g.reward)
        fmt += int(g.format_ok)
    out["sampled_rewards"] = rewards
    out["sampled_format_ok"] = fmt
    out["sampled_lengths"] = rb.lengths.reshape(-1).tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hf", "vllm", "report"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out-dir", default="runs/backend-compare")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend != "report":
        res = run(args.backend, args)
        (out_dir / f"{args.backend}.json").write_text(json.dumps(res))
        print(f"wrote {out_dir / f'{args.backend}.json'}")
        return 0

    hf = json.loads((out_dir / "hf.json").read_text())
    vl = json.loads((out_dir / "vllm.json").read_text())

    n = len(hf["greedy_texts"])
    exact = sum(h == v for h, v in zip(hf["greedy_texts"], vl["greedy_texts"], strict=True))
    prefix = sum(
        h[:100] == v[:100] for h, v in zip(hf["greedy_texts"], vl["greedy_texts"], strict=True)
    )
    gr_hf = sum(hf["greedy_rewards"]) / n
    gr_vl = sum(vl["greedy_rewards"]) / n

    print("=" * 62)
    print("GREEDY (temperature 0) -- divergence here is implementation, not chance")
    print("=" * 62)
    print(f"  exact text match:    {exact}/{n} ({exact / n:.0%})")
    print(f"  first-100-char match:{prefix}/{n} ({prefix / n:.0%})")
    print(f"  greedy accuracy:     hf={gr_hf:.3f}  vllm={gr_vl:.3f}  delta={gr_vl - gr_hf:+.3f}")

    sh, sv = hf["sampled_rewards"], vl["sampled_rewards"]
    mh, mv = sum(sh) / len(sh), sum(sv) / len(sv)
    # Binomial standard error on the difference of two proportions.
    se = ((mh * (1 - mh) / len(sh)) + (mv * (1 - mv) / len(sv))) ** 0.5
    z = (mv - mh) / se if se > 0 else 0.0
    lh = sum(hf["sampled_lengths"]) / len(hf["sampled_lengths"])
    lv = sum(vl["sampled_lengths"]) / len(vl["sampled_lengths"])

    print()
    print("=" * 62)
    print(f"SAMPLED (temperature 1, n={len(sh)} completions per backend)")
    print("=" * 62)
    print(f"  mean reward:    hf={mh:.3f}  vllm={mv:.3f}  delta={mv - mh:+.3f}  (SE {se:.3f}, z={z:+.2f})")
    print(f"  boxed format:   hf={hf['sampled_format_ok']}/{len(sh)}  vllm={vl['sampled_format_ok']}/{len(sv)}")
    print(f"  mean length:    hf={lh:.0f}  vllm={lv:.0f} tokens")
    print()
    if abs(z) < 2:
        print(f"  VERDICT: reward difference is within sampling noise (|z|={abs(z):.2f} < 2).")
        print("           The throughput comparison is measuring one thing.")
    else:
        print(f"  VERDICT: reward differs beyond sampling noise (|z|={abs(z):.2f} >= 2).")
        print("           The backends are NOT equivalent -- speedup claim is confounded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
