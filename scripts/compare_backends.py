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

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def run(backend_name: str, args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rlv.tasks import gsm8k
    from rlv.train import build_prompt

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = gsm8k.load("train", limit=args.prompts, revision=args.dataset_revision)
    prompts = [build_prompt(tok, r["question"]) for r in data]

    if backend_name == "vllm":
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from rlv.rollout import VLLMRollout

        backend = VLLMRollout(args.model, tok, gpu_frac=0.55, max_model_len=args.max_new_tokens + 512, revision=args.model_revision)
    else:
        from rlv.rollout import HFRollout

        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        backend = HFRollout(model, tok)

    out: dict = {"backend": backend_name, "config": {k: v for k, v in vars(args).items() if k not in ("backend", "out_dir")},
                 "prompt_ids": prompts, "group_size": args.group_size}

    # --- greedy ---------------------------------------------------------------
    # temperature <= 0 selects argmax in both backends. Approximating greedy with a
    # tiny temperature instead would divide the logits by ~1e-7 and overflow, which
    # produces garbage and looks exactly like a real divergence.
    torch.manual_seed(0)
    rb = backend.generate(prompts, 1, args.max_new_tokens, 0.0)
    out["greedy_texts"] = rb.texts
    out["greedy_tokens"] = [row[mask].tolist() for row, mask in zip(rb.sequences, rb.completion_mask, strict=True)]
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


def compare(hf, vl):
    """Descriptive checks, never a statistical equivalence gate.

    Reward uncertainty uses paired prompt-level differences (clusters of completions),
    with a descriptive normal interval. No predeclared equivalence margin exists.
    """
    import math
    import statistics

    ht, vt = hf["greedy_texts"], vl["greedy_texts"]
    if not ht or len(ht) != len(vt):
        raise ValueError("nonempty aligned greedy outputs required")
    result = {"configuration_match": bool(hf.get("config")) and hf.get("config") == vl.get("config"),
              "prompts_match": bool(hf.get("prompt_ids")) and hf.get("prompt_ids") == vl.get("prompt_ids"),
              "exact_text_fraction": sum(h == vt[i] for i, h in enumerate(ht)) / len(ht),
              "prefix_100_character_fraction": sum(h[:100] == vt[i][:100] for i, h in enumerate(ht)) / len(ht),
              "exact_token_fraction": None,
              "equivalence_established": False,
              "conclusion": "Descriptive diagnostics only; matching rewards or outputs do not establish distributional equivalence."}
    if hf.get("greedy_tokens") is not None and vl.get("greedy_tokens") is not None:
        if len(hf["greedy_tokens"]) != len(vl["greedy_tokens"]):
            raise ValueError("token row counts differ")
        pairs = [(h, vl["greedy_tokens"][i]) for i, h in enumerate(hf["greedy_tokens"])]
        if len(pairs) != len(ht):
            raise ValueError("token/text row counts differ")
        result["exact_token_fraction"] = sum(h == v for h, v in pairs) / len(pairs)
    sh, sv = hf['sampled_rewards'], vl['sampled_rewards']
    if not sh or not sv:
        raise ValueError('nonempty reward observations required')
    result['reward_delta_vllm_minus_hf'] = statistics.mean(sv) - statistics.mean(sh)
    result['paired_prompt_se'] = None
    result['descriptive_95_interval'] = None
    g = hf.get('group_size')
    if result['prompts_match'] and g and g == vl.get('group_size'):
        if len(sh) != len(sv) or len(sh) != len(ht) * g:
            raise ValueError('sample groups do not align with prompts')
        deltas = [statistics.mean(sv[i:i+g]) - statistics.mean(sh[i:i+g]) for i in range(0, len(sh), g)]
        if len(deltas) > 1:
            se = statistics.stdev(deltas) / math.sqrt(len(deltas))
            delta = statistics.mean(deltas)
            result['paired_prompt_se'] = se
            result['descriptive_95_interval'] = [delta - 1.96 * se, delta + 1.96 * se]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hf", "vllm", "report"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-revision")
    ap.add_argument("--dataset-revision")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out-dir", default="runs/backend-compare")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend != "report":
        res = run(args.backend, args)
        with (out_dir / f"{args.backend}.json").open("x", encoding="utf-8", newline="\n") as f:
            json.dump(res, f)
        print(f"wrote {out_dir / f'{args.backend}.json'}")
        return 0

    hf = json.loads((out_dir / "hf.json").read_text(encoding="utf-8"))
    vl = json.loads((out_dir / "vllm.json").read_text(encoding="utf-8"))

    print(json.dumps(compare(hf, vl), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
