"""GRPO training loop, instrumented, with a switchable rollout backend.

`--rollout-backend hf` is the naive baseline; `--rollout-backend vllm` is the fast path.
Everything downstream of rollout is identical between them by construction, so the
difference in step time is attributable to the rollout path and nothing else.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlv.algo.advantages import group_advantages, wasted_generation_tokens
from rlv.instrument import MemoryProbe, PhaseTimer, Recorder, StepAccount
from rlv.rollout import HFRollout, VLLMRollout
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


def completion_logprobs(model, sequences, attention_mask, completion_mask):
    """Per-token logprobs of the sampled completions under the current policy.

    Recomputed rather than captured during generation: generation runs under no_grad and
    the loss needs a differentiable path. The attention mask is not optional -- without
    it the model attends to padding and scores the completions under a context the
    sampler never saw, which trains fine and is quietly wrong.
    """
    logits = model(sequences, attention_mask=attention_mask).logits[:, :-1]
    targets = sequences[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp, completion_mask[:, 1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rollout-backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--normalise", default="none", choices=["none", "std"])
    ap.add_argument("--vllm-gpu-frac", type=float, default=0.32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="grpo")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.rollout_backend == "vllm":
        # Must be set before vLLM is imported. In-process workers are what make weight
        # sync a tensor handoff instead of an IPC problem.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        backend = VLLMRollout(
            args.model, tok,
            gpu_frac=args.vllm_gpu_frac,
            max_model_len=args.max_new_tokens + 512,
            seed=args.seed,
        )
    else:
        backend = HFRollout(model, tok)

    data = gsm8k.load("train", limit=512)
    print(f"loaded {len(data)} gsm8k problems, rollout backend = {backend.name}")

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

            # The engine's copy of the weights is stale the moment the optimiser steps,
            # so this is on the critical path and is timed as such.
            if backend.needs_weight_sync():
                with timer("weight_sync"):
                    backend.sync_weights(model)

            # Generation and training want opposite configurations: generation needs the
            # KV cache and no dropout, the update needs checkpointing to fit in memory.
            with timer("rollout"):
                model.gradient_checkpointing_disable()
                model.eval()
                rb = backend.generate(prompts, args.group_size, args.max_new_tokens, args.temperature)
                model.gradient_checkpointing_enable()
                model.train()

            with timer("grade"):
                rewards = torch.zeros(args.prompts_per_step, args.group_size)
                n_format_ok = 0
                for i in range(args.prompts_per_step):
                    for j in range(args.group_size):
                        g = gsm8k.grade(rb.texts[i * args.group_size + j], batch[i]["gold"])
                        rewards[i, j] = g.reward
                        n_format_ok += int(g.format_ok)

            adv = group_advantages(rewards, normalise=args.normalise)
            flat_adv = adv.advantages.reshape(-1).to(model.device)

            opt.zero_grad(set_to_none=True)
            total_seqs = rb.sequences.shape[0]
            n_micro = math.ceil(total_seqs / args.micro_batch)
            loss_sum = 0.0
            trained_tokens = 0

            for m in range(n_micro):
                lo, hi = m * args.micro_batch, min((m + 1) * args.micro_batch, total_seqs)
                with timer("forward"):
                    token_logp, mask = completion_logprobs(
                        model, rb.sequences[lo:hi], rb.attention_mask[lo:hi], rb.completion_mask[lo:hi]
                    )
                    a = flat_adv[lo:hi].unsqueeze(1)
                    # Token-mean within the microbatch, scaled by its share of the step.
                    # Summing per-sequence would let long completions dominate the
                    # update purely by being long.
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
            acct.tokens_generated = int(rb.lengths.sum().item())
            acct.tokens_trained = trained_tokens
            acct.n_groups = adv.n_groups
            acct.n_groups_degenerate = adv.n_degenerate
            acct.tokens_generated_wasted = wasted_generation_tokens(rb.lengths, rewards)
            acct.reward_mean = adv.reward_mean
            acct.reward_std = adv.reward_std
            acct.advantage_abs_mean = adv.advantage_abs_mean
            acct.grad_norm = float(grad_norm)
            acct.loss = loss_sum
            acct.mem_peak_alloc_gb = mem.peak_alloc_gb
            acct.mem_frag_gb = mem.frag_gb
            rec.event("step_extra", step=step, backend=backend.name, format_ok=n_format_ok)
            rec.step(acct)

            roll_w = snap.get("rollout", {}).get("wall_s", 0.0)
            sync_w = snap.get("weight_sync", {}).get("wall_s", 0.0)
            print(
                f"step {step}  reward={acct.reward_mean:.3f}  "
                f"degen={acct.degenerate_frac:.0%}  wasted_tok={acct.wasted_token_frac:.0%}  "
                f"step={acct.wall_s:.1f}s  rollout={roll_w:.1f}s ({roll_w / acct.wall_s:.0%})  "
                f"sync={sync_w:.2f}s  gen={acct.tokens_generated / roll_w if roll_w else 0:.0f} tok/s  "
                f"mem={acct.mem_peak_alloc_gb:.1f}GB  fmt={n_format_ok}/{total_seqs}"
            )

    print(f"\nrun complete -> {rec.run_dir}/events.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
