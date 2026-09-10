"""GRPO training loop, instrumented, with a switchable rollout backend.

`--rollout-backend hf` is the naive baseline; `--rollout-backend vllm` is the fast path.
Shared update code controls one source of variation; it does not prove sampler equivalence.
This is a simplified group-relative REINFORCE loss, without PPO clipping or KL.
"""

from __future__ import annotations

import argparse
import math
import os

import torch

from rlv.algo.advantages import group_advantages, wasted_generation_tokens
from rlv.instrument import MemoryProbe, PhaseTimer, Recorder, StepAccount, StepTimer
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


@torch.no_grad()
def evaluate(backend, model, tok, eval_rows, batch: int = 64) -> dict:
    """Greedy accuracy on a held-out split.

    Training reward is a moving target -- the prompts change every step and the sampling
    is stochastic, so a rising train reward can reflect easier prompts as much as a
    better policy. A fixed held-out set decoded greedily is the number worth comparing
    between conditions.
    """
    model.gradient_checkpointing_disable()
    model.eval()
    if backend.needs_weight_sync():
        backend.sync_weights(model)

    n_correct = n_fmt = 0
    for lo in range(0, len(eval_rows), batch):
        chunk = eval_rows[lo:lo + batch]
        prompts = [build_prompt(tok, r["question"]) for r in chunk]
        rb = backend.generate(prompts, 1, 512, 0.0)   # temperature 0 -> argmax
        for text, row in zip(rb.texts, chunk, strict=True):
            g = gsm8k.grade(text, row["gold"])
            n_correct += int(g.correct)
            n_fmt += int(g.format_ok)

    model.gradient_checkpointing_enable()
    model.train()
    n = len(eval_rows)
    return {"eval_n": n, "eval_accuracy": n_correct / n, "eval_format_ok": n_fmt / n}


def completion_logprobs(model, sequences, attention_mask, completion_mask, temperature=1.0):
    """Per-token logprobs of the sampled completions under the current policy.

    Recomputed rather than captured during generation: generation runs under no_grad and
    the loss needs a differentiable path. The attention mask is not optional -- without
    it the model attends to padding and scores the completions under a context the
    sampler never saw, which trains fine and is quietly wrong.
    """
    validate_temperature(temperature)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    logits = model(sequences, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits[:, :-1]
    targets = sequences[:, 1:]
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp, completion_mask[:, 1:]


def validate_temperature(temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("training temperature must be finite and positive; greedy is evaluation only")


def policy_loss(token_logp, mask, advantages, total_sequences):
    """Sum per-sequence token means / full-step sequence count (including empty rows).

    Every microbatch contributes to the same global denominator. Empty completions
    contribute zero. Partitioning ragged sequences cannot change this reduction.
    """
    if total_sequences <= 0:
        raise ValueError("total_sequences must be positive")
    seq_means = (token_logp * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return -(seq_means * advantages).sum() / total_sequences


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-revision", required=True, help="immutable checkpoint commit SHA")
    ap.add_argument("--dataset-revision", required=True, help="immutable openai/gsm8k commit SHA")
    ap.add_argument("--allow-off-policy", action="store_true", help="explicit diagnostic penalty experiment")
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
    ap.add_argument(
        "--rollout-repetition-penalty", type=float, default=1.0,
        help="1.0 is correct. Set 1.1 to reproduce the value Qwen2.5 leaks via "
             "generation_config.json, for the A/B on whether that bias changes learning.",
    )
    ap.add_argument("--eval-every", type=int, default=0, help="0 disables held-out eval")
    ap.add_argument("--eval-prompts", type=int, default=200)
    ap.add_argument("--train-prompts", type=int, default=2048)
    ap.add_argument("--run-name", default="grpo")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    validate_temperature(args.temperature)
    for key in ("steps", "prompts_per_step", "group_size", "max_new_tokens", "micro_batch", "train_prompts", "eval_prompts"):
        if getattr(args, key) <= 0:
            ap.error(f"{key} must be positive")
    import re
    for revision in (args.model_revision, args.dataset_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            ap.error("revisions must be immutable 40-character commit SHAs")
    if args.rollout_repetition_penalty != 1.0 and not args.allow_off_policy:
        ap.error("nonneutral penalty requires --allow-off-policy; this is a diagnostic, not on-policy training")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Reserve before any expensive model/data load. Failed setup keeps its provenance.
    rec = Recorder(args.runs_dir, args.run_name, config=vars(args))
    try:
        torch.manual_seed(args.seed)
    
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision, dtype=torch.bfloat16, device_map="cuda")
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
                revision=args.model_revision,
                repetition_penalty=args.rollout_repetition_penalty,
            )
        else:
            backend = HFRollout(model, tok, repetition_penalty=args.rollout_repetition_penalty)
    
        data = gsm8k.load("train", limit=args.train_prompts, revision=args.dataset_revision)
        eval_rows = gsm8k.load("test", limit=args.eval_prompts, revision=args.dataset_revision) if args.eval_every else []
        if not data or (args.eval_every and not eval_rows):
            raise ValueError("no parseable rows in requested dataset split")
        print(f"loaded {len(data)} train / {len(eval_rows)} eval problems, "
              f"backend={backend.name} rollout_penalty={args.rollout_repetition_penalty}")
    
        # Each seed sees a different prompt order, so a difference between conditions
        # cannot be an artefact of one ordering.
        import random
        random.Random(args.seed).shuffle(data)
    
        print(f"logging to {rec.run_dir}")
    
    except BaseException as exc:
        rec.event("run_error", error_type=type(exc).__name__, error=str(exc))
        rec.close(status="failed", stage="setup")
        raise

    cursor = 0
    with rec:
        for step in range(args.steps):
            # Evaluate before the update, so step 0 records the untrained baseline
            # every condition starts from.
            if args.eval_every and step % args.eval_every == 0:
                ev = evaluate(backend, model, tok, eval_rows)
                rec.event("eval", step=step, **ev)
                print(f"  [eval] step {step}  accuracy={ev['eval_accuracy']:.3f}  "
                      f"format_ok={ev['eval_format_ok']:.3f}  (n={ev['eval_n']})")

            step_timer = StepTimer()
            step_timer.start()
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
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.eval()  # retain checkpointing while disabling dropout

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
                        model, rb.sequences[lo:hi], rb.attention_mask[lo:hi], rb.completion_mask[lo:hi],
                        temperature=args.temperature,
                    )
                    loss = policy_loss(token_logp, mask, flat_adv[lo:hi], total_seqs)
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

            acct.phase_host_s = {k: v["wall_s"] for k, v in snap.items()}
            acct.device_s = {k: v["device_s"] for k, v in snap.items()}
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
            acct.wall_s = step_timer.stop()
            rec.event("step_extra", step=step, backend=backend.name, format_ok=n_format_ok)
            rec.step(acct)

            roll_w = snap.get("rollout", {}).get("wall_s", 0.0)
            sync_w = snap.get("weight_sync", {}).get("wall_s", 0.0)
            print(
                f"step {step}  reward={acct.reward_mean:.3f}  "
                f"degen={acct.degenerate_frac:.0%}  wasted_tok={acct.wasted_token_frac:.0%}  "
                f"step={acct.wall_s:.1f}s  rollout_host={roll_w:.1f}s  "
                f"sync_host={sync_w:.2f}s  gen/step={acct.generated_tokens_per_s:.0f} tok/s  "
                f"mem={acct.mem_peak_alloc_gb:.1f}GB  fmt={n_format_ok}/{total_seqs}"
            )

        if args.eval_every:
            ev = evaluate(backend, model, tok, eval_rows)
            rec.event("eval", step=args.steps, final=True, **ev)
            print(f"  [eval] final  accuracy={ev['eval_accuracy']:.3f}  "
                  f"format_ok={ev['eval_format_ok']:.3f}")

    print(f"\nrun complete -> {rec.run_dir}/events.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
