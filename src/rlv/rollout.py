"""Rollout backends behind one interface.

Both backends return the same thing, so `train.py` can switch between them with a flag
and everything downstream -- grading, advantages, loss, optimiser -- stays byte-for-byte
identical. A speedup measured across two different scripts is not a measurement; it is
two anecdotes with different bugs.

The vLLM backend's weight-sync cost is timed as its own phase and counted in the step.
An optimisation that moves work somewhere you stopped looking is not an optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Both backends pin these rather than inheriting the model's generation_config.json.
#
# Qwen2.5-0.5B-Instruct ships temperature=0.7, top_p=0.8, top_k=20 and
# repetition_penalty=1.1 in that file. HF `generate` applies all four unless each is
# overridden individually; vLLM's SamplingParams defaults to top_k=0 and
# repetition_penalty=1.0. Overriding only temperature and top_p therefore leaves the
# two backends sampling from measurably different distributions -- 0.262 vs 0.406 mean
# reward, z=3.51, which is more than enough to swamp any throughput comparison.
#
# The deeper problem is not the benchmark. On-policy RL requires that the distribution
# you sample from is the distribution you compute logprobs under. A repetition penalty
# applied during rollout but absent from the training forward pass makes the two differ,
# so the policy gradient is silently computed against the wrong distribution -- and on
# maths a repetition penalty is actively harmful, because correct arithmetic means
# re-emitting digits the penalty is suppressing.
NEUTRAL_SAMPLING = {"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}


@dataclass
class RolloutBatch:
    """Padded sequences plus everything the loss needs to mask correctly."""

    sequences: torch.Tensor        # [B*G, T] left-padded prompt then completion
    attention_mask: torch.Tensor   # [B*G, T] 0 on padding
    completion_mask: torch.Tensor  # [B*G, T] 1 only on sampled completion tokens
    texts: list[str]               # decoded completions, group-contiguous per prompt
    lengths: torch.Tensor          # [B, G] completion token counts


def assemble(
    prompt_ids: list[list[int]],
    completion_ids: list[list[int]],
    texts: list[str],
    n_prompts: int,
    group_size: int,
    pad_id: int,
    device: str | torch.device,
) -> RolloutBatch:
    """Pack ragged prompt/completion pairs into padded tensors.

    Prompts are left-padded and completions right-padded, so every sequence has its
    prompt/completion boundary at the same index and the loss can mask by column. The
    attention mask is built here rather than left to the caller because omitting it is
    a silent correctness bug: the forward pass would attend to pad tokens and score the
    completions under a context the sampler never saw.
    """
    max_p = max(len(p) for p in prompt_ids)
    max_c = max(len(c) for c in completion_ids)
    total, width = len(prompt_ids), max_p + max_c

    seqs = torch.full((total, width), pad_id, dtype=torch.long)
    attn = torch.zeros((total, width), dtype=torch.long)
    comp = torch.zeros((total, width), dtype=torch.bool)

    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids, strict=True)):
        lo = max_p - len(p)                     # left pad the prompt
        seqs[i, lo:max_p] = torch.tensor(p, dtype=torch.long)
        attn[i, lo:max_p] = 1
        if c:
            seqs[i, max_p:max_p + len(c)] = torch.tensor(c, dtype=torch.long)
            attn[i, max_p:max_p + len(c)] = 1
            comp[i, max_p:max_p + len(c)] = True

    lengths = torch.tensor(
        [len(c) for c in completion_ids], dtype=torch.long
    ).view(n_prompts, group_size)

    return RolloutBatch(
        sequences=seqs.to(device),
        attention_mask=attn.to(device),
        completion_mask=comp.to(device),
        texts=texts,
        lengths=lengths,
    )


class HFRollout:
    """Baseline: sample with `model.generate`.

    Deliberately the reference implementation. It is the obvious thing to write, which
    is what makes it the honest baseline to beat.
    """

    name = "hf"

    def __init__(self, model, tok, repetition_penalty: float = 1.0):
        self.model = model
        self.tok = tok
        # Exposed so the inherited-penalty condition can be reproduced deliberately
        # in an A/B, rather than only avoided.
        self.repetition_penalty = repetition_penalty

    def needs_weight_sync(self) -> bool:
        return False  # the trainer's own weights are what generated

    @torch.no_grad()
    def generate(self, prompts, group_size, max_new, temperature) -> RolloutBatch:
        model, tok = self.model, self.tok
        if model.training:
            raise RuntimeError("rollout requires eval mode; generating in train mode yields garbage")
        if getattr(model, "is_gradient_checkpointing", False):
            raise RuntimeError("rollout requires gradient checkpointing off (it forces use_cache=False)")

        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        plen = enc.input_ids.shape[1]
        greedy = temperature <= 0
        out = model.generate(
            **enc,
            do_sample=not greedy,
            # Every sampling parameter is pinned explicitly. `generate` otherwise
            # inherits the model's generation_config.json, and Qwen2.5 ships
            # top_k=20 / repetition_penalty=1.1 there. See NEUTRAL_SAMPLING below.
            temperature=None if greedy else temperature,
            top_p=None if greedy else 1.0,
            top_k=None if greedy else 0,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=max_new,
            num_return_sequences=group_size,
            pad_token_id=tok.pad_token_id,
        )

        # Strip padding back out so both backends hand `assemble` the same ragged form.
        prompt_ids, completion_ids, texts = [], [], []
        for i in range(out.shape[0]):
            p = [t for t in out[i, :plen].tolist() if t != tok.pad_token_id]
            c = out[i, plen:].tolist()
            while c and c[-1] == tok.pad_token_id:
                c.pop()
            prompt_ids.append(p or [tok.pad_token_id])
            completion_ids.append(c)
            texts.append(tok.decode(c, skip_special_tokens=True))

        n_prompts = len(prompts)
        return assemble(
            prompt_ids, completion_ids, texts, n_prompts, group_size,
            tok.pad_token_id, model.device,
        )


class VLLMRollout:
    """Fast path: sample with vLLM, syncing policy weights in before each rollout.

    The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so weights can be
    handed over as live CUDA tensors. Across processes this would need NCCL or CUDA IPC,
    which is the right design at cluster scale and pure overhead on one card.
    """

    name = "vllm"

    def __init__(self, model_name: str, tok, gpu_frac: float, max_model_len: int, seed: int = 0,
                 repetition_penalty: float = 1.0):
        from vllm import LLM

        self.tok = tok
        self.repetition_penalty = repetition_penalty
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_frac,
            max_model_len=max_model_len,
            dtype="bfloat16",
            enforce_eager=False,
            seed=seed,
        )

    def needs_weight_sync(self) -> bool:
        return True  # the engine holds its own copy, stale the moment the optimiser steps

    def sync_weights(self, model) -> None:
        """Push trainer weights into the engine.

        vLLM's `load_weights` takes HuggingFace-format names and handles its own fusions
        (q/k/v into qkv_proj, gate/up into gate_up_proj) internally, so the trainer's
        `named_parameters()` can go straight across.
        """
        named = [(n, p.detach()) for n, p in model.named_parameters()]

        def _load(vllm_model):
            vllm_model.load_weights(iter(named))

        self.llm.apply_model(_load)

    def generate(self, prompts, group_size, max_new, temperature) -> RolloutBatch:
        from vllm import SamplingParams

        outs = self.llm.generate(
            prompts,
            SamplingParams(
                n=group_size,
                temperature=max(temperature, 0.0),
                top_p=1.0,
                top_k=0,               # 0 disables in vLLM
                repetition_penalty=self.repetition_penalty,
                max_tokens=max_new,
            ),
            use_tqdm=False,
        )

        prompt_ids, completion_ids, texts = [], [], []
        for o in outs:                      # vLLM preserves input order
            for c in o.outputs:             # group-contiguous per prompt, matching HF
                prompt_ids.append(list(o.prompt_token_ids))
                completion_ids.append(list(c.token_ids))
                texts.append(c.text)

        return assemble(
            prompt_ids, completion_ids, texts, len(prompts), group_size,
            self.tok.pad_token_id, "cuda",
        )
