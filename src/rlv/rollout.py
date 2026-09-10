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

# Explicit sampler settings avoid checkpoint-dependent defaults. Configuration
# alignment alone does not establish output distribution or learning equivalence.
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
    if not prompt_ids or any(not p for p in prompt_ids):
        raise ValueError("each sequence needs a nonempty prompt")
    if not (len(prompt_ids) == len(completion_ids) == len(texts) == n_prompts * group_size):
        raise ValueError("ragged batch dimensions disagree")
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


def trim_completion(tokens, eos_id, pad_id):
    """Keep the first EOS and remove its generation padding suffix.

    A PAD token before EOS can itself be sampled; retain it. Without EOS all
    generated positions are real. Prompt padding uses the input attention mask.
    """
    eos_ids = set(eos_id if isinstance(eos_id, (list, tuple)) else [eos_id])
    for i, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:i + 1]
    return tokens


class HFRollout:
    """Baseline: sample with `model.generate`.

    Deliberately the reference implementation. It is the obvious thing to write, which
    is what makes it the honest baseline to beat.
    """

    name = "hf"

    def __init__(self, model, tok, repetition_penalty: float = 1.0):
        from transformers import GenerationConfig

        self.model = model
        self.tok = tok
        # Replace checkpoint generation defaults as well as passing explicit knobs.
        # This covers inherited processors beyond top-k/top-p/repetition penalty.
        model.generation_config = GenerationConfig(
            bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
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
            # Start from the neutral config installed above and override the knobs.
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
            row = i // group_size
            p = enc.input_ids[row][enc.attention_mask[row].bool()].tolist()
            c = trim_completion(out[i, plen:].tolist(), tok.eos_token_id, tok.pad_token_id)
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
                 repetition_penalty: float = 1.0, revision: str | None = None):
        from vllm import LLM

        self.tok = tok
        self.repetition_penalty = repetition_penalty
        self.llm = LLM(
            model=model_name,
            revision=revision,
            tokenizer_revision=revision,
            generation_config="vllm",
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
