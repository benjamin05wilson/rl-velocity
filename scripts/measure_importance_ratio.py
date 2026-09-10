"""How wrong is the policy gradient when rollout sampling carries an inherited penalty?

GRPO and PPO assume that for freshly sampled tokens the importance ratio
pi_policy(t) / pi_sampling(t) is exactly 1, and drop it. That holds only if the
distribution you sampled from is the distribution you differentiate.

When HF `generate` inherits repetition_penalty from the checkpoint's
generation_config.json, the sampler applies a history-dependent reshaping that the
training forward pass knows nothing about. The ratio is then not 1. This measures how
far from 1, in the units that matter.

Method: sample with the penalty active, then for the *same* tokens compute

  logp_policy   -- plain log_softmax of the logits (what the trainer differentiates)
  logp_sampling -- log_softmax after HF's own RepetitionPenaltyLogitsProcessor
                   (what actually drew the token)

using the library's processor rather than a reimplementation, so the sampling
distribution is exactly the one `generate` used. Everything else is held neutral
(temperature 1, top_k off, top_p off) so the penalty is the only difference.
"""

from __future__ import annotations

import argparse

import torch

from rlv.tasks import gsm8k
from rlv.train import build_prompt

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def interpretation(log_ratios):
    if log_ratios.numel() == 0:
        return "No completion tokens measured; no ratio conclusion available."
    ratios = log_ratios.exp()
    fraction = ((ratios - 1).abs() > 0.10).float().mean().item()
    p1 = torch.quantile(ratios.float(), 0.01).item()
    geo = log_ratios.mean().exp().item()
    return (f"Measured {fraction:.1%} of tokens more than 10% from ratio 1; "
            f"p1={p1:.4f}; geometric mean={geo:.4f}. "
            "These token diagnostics alone do not establish learning impact or a correction estimator.")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--penalty", type=float, default=1.1, help="the value Qwen2.5 ships")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    data = gsm8k.load("train", limit=args.prompts)
    prompts = [build_prompt(tok, r["question"]) for r in data]
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    plen = enc.input_ids.shape[1]

    # Sample exactly as an unguarded HF rollout would: penalty inherited, everything
    # else neutral, so the penalty is the sole source of mismatch.
    with torch.no_grad():
        seqs = model.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=args.penalty,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.group_size,
            pad_token_id=tok.pad_token_id,
        )

    with torch.no_grad():
        logits = model(seqs, attention_mask=(seqs != tok.pad_token_id).long()).logits.float()

    proc = RepetitionPenaltyLogitsProcessor(penalty=args.penalty)

    log_ratios: list[torch.Tensor] = []
    n_seq = seqs.shape[0]
    seq_logratio = torch.zeros(n_seq, device=seqs.device)
    seq_ntok = torch.zeros(n_seq, device=seqs.device)

    # Position t predicts token t+1. Walk the completion span, applying the processor
    # to the same prefix `generate` would have had at that step.
    for t in range(plen - 1, seqs.shape[1] - 1):
        step_logits = logits[:, t, :]
        chosen = seqs[:, t + 1]
        alive = chosen != tok.pad_token_id
        if not alive.any():
            continue

        logp_policy = torch.log_softmax(step_logits, dim=-1)
        penalised = proc(seqs[:, : t + 1], step_logits.clone())
        logp_sampling = torch.log_softmax(penalised, dim=-1)

        idx = chosen.unsqueeze(-1)
        lr = logp_policy.gather(-1, idx).squeeze(-1) - logp_sampling.gather(-1, idx).squeeze(-1)
        lr = torch.where(alive, lr, torch.zeros_like(lr))
        log_ratios.append(lr[alive].detach())
        seq_logratio += lr
        seq_ntok += alive.float()

    if not log_ratios:
        print("No completion tokens measured")
        return 1
    lr_all = torch.cat(log_ratios)
    ratios = lr_all.exp()

    print(f"\nmodel={args.model}  inherited repetition_penalty={args.penalty}")
    print(f"sequences={n_seq}  completion tokens measured={lr_all.numel()}")

    print("\n--- token-level importance ratio  pi_policy / pi_sampling ---")
    q = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=ratios.device)
    qs = torch.quantile(ratios, q)
    print(f"  mean            {ratios.mean():.4f}")
    print(f"  median          {qs[2]:.4f}")
    print(f"  p1 / p99        {qs[0]:.4f} / {qs[4]:.4f}")
    print(f"  min / max       {ratios.min():.4f} / {ratios.max():.4f}")
    exact = (ratios - 1).abs() < 1e-4
    print(f"  ratio == 1      {exact.float().mean():.1%} of tokens (the assumed value)")
    print(f"  |ratio-1| > 1%  {((ratios - 1).abs() > 0.01).float().mean():.1%} of tokens")
    print(f"  |ratio-1| > 10% {((ratios - 1).abs() > 0.10).float().mean():.1%} of tokens")

    # Geometric mean is the honest per-token summary. The arithmetic mean of a ratio
    # sits near 1 by construction whenever a thin right tail offsets a fat left one,
    # so quoting it alone would understate the distortion.
    geo = lr_all.mean().exp()
    print(f"  geometric mean  {geo:.4f}   <- per-token summary; 1.0 would be correct")

    print("\n--- sequence-level weight  prod over tokens ---")
    valid = seq_ntok > 0
    slr = seq_logratio[valid]
    print(f"  mean log-weight  {slr.mean():+.2f}   (0 would be correct)")
    print(f"  min / max        {slr.min():+.2f} / {slr.max():+.2f}")
    print(f"  mean tokens/seq  {seq_ntok[valid].mean():.0f}")

    print("\n" + interpretation(lr_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
