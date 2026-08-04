"""GRPO training loop, instrumented.

Rollouts here use HuggingFace `generate`, not vLLM, and that is deliberate.

The question this repo exists to answer is where RL training compute goes. Starting
from the naive loop makes the answer legible and produces an honest before-number that
the vLLM integration in the next phase has to beat. Wiring the fast path in first would
mean never measuring the thing being fixed, and "we made it faster" is not a result
without a baseline that was measured rather than assumed.

The vLLM gate (scripts/verify_vllm.py) already confirmed the fast path is available on
this hardware, so this is a sequencing choice, not a fallback.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlv.algo.advantages import group_advantages, wasted_generation_tokens
from rlv.instrument import MemoryProbe, PhaseTimer, Recorder, StepAccount
from rlv.tasks import gsm8k


def build_prompt(tok, question: str) -> str:
    """Apply the chat template.

    Skipping this is a quiet correctness bug: an instruct-tuned model fed a raw string
    is off-distribution, rambles past the token limit, and produces a reward curve that
    looks like a learning-rate problem when it is actually a formatting one.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": gsm8k.PROMPT_TEMPLATE.format(question=question)}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def rollout(model, tok, prompts: list[str], group_size: int, max_new: int, temperature: float):
    """Sample `group_size` completions for each prompt.

    Returns padded sequences plus the prompt length, so the loss can mask prompt tokens
    later. Advantage applies to what the policy *chose*, never to what it was given.

    The model MUST be in eval mode with gradient checkpointing off. Checkpointing forces
    `use_cache=False`, and generating in that configuration silently produces incoherent
    output -- measured on Qwen2.5-0.5B: 0/8 correct and 7/8 running to the token cap,
    against 6/8 correct in eval mode at the same temperature. Nothing raises. Reward
    simply pins to zero, which reads exactly like a learning-rate problem and will
    happily consume days.
    """
    if model.training:
        raise RuntimeError("rollout() requires eval mode; generating in train mode yields garbage")
    if getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("rollout() requires gradient checkpointing disabled (it forces use_cache=False)")

    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    prompt_len = enc.input_ids.shape[1]

    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        max_new_tokens=max_new,
        num_return_sequences=group_size,
        pad_token_id=tok.pad_token_id,
    )
    return out, prompt_len


def completion_logprobs(model, sequences, prompt_len: int, pad_id: int):
    """Per-token logprobs of the sampled completions under the current policy.

    Recomputed rather than captured during generation: `generate` runs under no_grad,
    and the loss needs a differentiable path. This forward pass is a real cost and is
    timed separately so it shows up in the accounting instead of hiding inside "train".
    """
    logits = model(sequences).logits[:, :-1]          # predict token t+1 from t
    targets = sequences[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Mask to completion tokens only, and drop padding.
    mask = torch.zeros_like(token_logp, dtype=torch.bool)
    mask[:, prompt_len - 1:] = True
    mask &= targets != pad_id
    return token_logp, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--normalise", default="none", choices=["none", "std"])
    ap.add_argument("--run-name", default="grpo-baseline")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data = gsm8k.load("train", limit=512)
    print(f"loaded {len(data)} gsm8k problems")

    rec = Recorder(args.runs_dir, args.run_name, config=vars(args))
    print(f"logging to {rec.run_dir}")

    cursor = 0
    with rec:
        for step in range(args.steps):
            timer = PhaseTimer()
            MemoryProbe.start()
            acct = StepAccount(step=step)

            batch = [data[(cursor + i) % len(data)] for i in range(args.prompts_per_step)]
            cursor += args.prompts_per_step
            prompts = [build_prompt(tok, r["question"]) for r in batch]

            # Generation and training want opposite configurations: generation needs the
            # KV cache and no dropout, the update needs checkpointing to fit in memory.
            # Toggling per phase is not incidental tidiness -- leaving the training
            # configuration on during rollout is the silent-garbage failure guarded
            # against in rollout().
            with timer("rollout"):
                model.gradient_checkpointing_disable()
                model.eval()
                seqs, prompt_len = rollout(
                    model, tok, prompts, args.group_size, args.max_new_tokens, args.temperature
                )
                model.gradient_checkpointing_enable()
                model.train()

            # --- grade ------------------------------------------------------
            with timer("grade"):
                texts = tok.batch_decode(seqs[:, prompt_len:], skip_special_tokens=True)
                # generate() returns groups contiguously per prompt.
                rewards = torch.zeros(args.prompts_per_step, args.group_size)
                lengths = torch.zeros(args.prompts_per_step, args.group_size, dtype=torch.long)
                n_format_ok = 0
                for i in range(args.prompts_per_step):
                    for j in range(args.group_size):
                        idx = i * args.group_size + j
                        g = gsm8k.grade(texts[idx], batch[i]["gold"])
                        rewards[i, j] = g.reward
                        n_format_ok += int(g.format_ok)
                        lengths[i, j] = int((seqs[idx, prompt_len:] != tok.pad_token_id).sum())

            adv = group_advantages(rewards, normalise=args.normalise)
            flat_adv = adv.advantages.reshape(-1).to(model.device)

            # --- policy update ----------------------------------------------
            opt.zero_grad(set_to_none=True)
            total_seqs = seqs.shape[0]
            n_micro = math.ceil(total_seqs / args.micro_batch)
            loss_sum = 0.0
            trained_tokens = 0

            for m in range(n_micro):
                lo, hi = m * args.micro_batch, min((m + 1) * args.micro_batch, total_seqs)
                with timer("forward"):
                    token_logp, mask = completion_logprobs(model, seqs[lo:hi], prompt_len, tok.pad_token_id)
                    a = flat_adv[lo:hi].unsqueeze(1)
                    # Token-mean within the microbatch, then scaled by its share of the
                    # step. Summing per-sequence instead would let long completions
                    # dominate the update purely by being long.
                    denom = mask.sum().clamp(min=1)
                    loss = -((token_logp * a * mask).sum() / denom) * ((hi - lo) / total_seqs)
                with timer("backward"):
                    loss.backward()
                loss_sum += float(loss.item())
                trained_tokens += int(mask.sum().item())

            with timer("optim"):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            timer.resolve()
            snap = timer.snapshot()
            mem = MemoryProbe.read()

            acct.wall_s = round(sum(p["wall_s"] for p in snap.values()), 3)
            acct.device_s = {k: v["device_s"] for k, v in snap.items()}
            acct.idle_s = {k: v["idle_s"] for k, v in snap.items()}
            acct.tokens_prompt = prompt_len * args.prompts_per_step
            acct.tokens_generated = int(lengths.sum().item())
            acct.tokens_trained = trained_tokens
            acct.n_groups = adv.n_groups
            acct.n_groups_degenerate = adv.n_degenerate
            acct.tokens_generated_wasted = wasted_generation_tokens(lengths, rewards)
            acct.reward_mean = adv.reward_mean
            acct.reward_std = adv.reward_std
            acct.advantage_abs_mean = adv.advantage_abs_mean
            acct.grad_norm = float(grad_norm)
            acct.loss = loss_sum
            acct.mem_peak_alloc_gb = mem.peak_alloc_gb
            acct.mem_frag_gb = mem.frag_gb
            rec.step(acct)

            gen_s = acct.device_s.get("rollout", 0.0)
            print(
                f"step {step}  reward={acct.reward_mean:.3f}  "
                f"degenerate={acct.degenerate_frac:.0%}  wasted_tok={acct.wasted_token_frac:.0%}  "
                f"rollout={gen_s:.1f}s ({gen_s / acct.wall_s:.0%} of step)  "
                f"gen={acct.generated_tokens_per_s:.0f} tok/s  "
                f"mem={acct.mem_peak_alloc_gb:.1f}GB  fmt_ok={n_format_ok}/{total_seqs}"
            )

    print(f"\nrun complete -> {rec.run_dir}/events.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
