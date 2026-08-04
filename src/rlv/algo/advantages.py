"""Group-relative advantages, and the accounting of what they cost.

GRPO replaces PPO's learned value function with a within-group baseline: sample G
completions per prompt, and score each against its siblings. That removes the critic --
which is why it fits on one 24GB card at all, since a value model would have to be
resident alongside the policy and the inference engine.

The consequence worth measuring is that a group whose completions all score the same
produces an advantage of exactly zero for every member. With a binary reward that means
every all-correct or all-wrong group is generation that was paid for and learned from
nothing. This module returns that count alongside the advantages rather than discarding
it, because it is the quantity the whole repo exists to study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Normalisation = Literal["std", "none"]


@dataclass
class AdvantageResult:
    advantages: torch.Tensor      # [n_groups, group_size]
    n_groups: int
    n_degenerate: int             # groups with zero reward variance
    reward_mean: float
    reward_std: float
    advantage_abs_mean: float

    @property
    def degenerate_frac(self) -> float:
        return self.n_degenerate / self.n_groups if self.n_groups else 0.0


def group_advantages(
    rewards: torch.Tensor,
    normalise: Normalisation = "none",
    eps: float = 1e-6,
) -> AdvantageResult:
    """Compute group-relative advantages from a [n_groups, group_size] reward tensor.

    `normalise` controls whether the centred reward is divided by the group's standard
    deviation:

      "std"  -- the original GRPO formulation. Dividing by a per-group std reweights
                each group by its own difficulty: a group where one of eight completions
                succeeded gets a much larger gradient than a group split four-four, for
                no principled reason. Dr. GRPO identifies this as a bias.
      "none" -- centre only. The default here, so the baseline is not carrying a known
                artefact into every comparison we later make against it.

    Both are kept because "which of these actually matters, and by how much on this
    hardware" is a measurable question, and answering it from our own logs is cheap.
    """
    if rewards.ndim != 2:
        raise ValueError(f"expected [n_groups, group_size], got {tuple(rewards.shape)}")

    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    centred = rewards - mean

    # Unbiased std is undefined for group_size == 1 and returns NaN; groups of one carry
    # no relative signal anyway, so clamp to zero advantage rather than propagating NaN.
    if rewards.shape[1] < 2:
        std = torch.zeros_like(mean)
    else:
        std = rewards.std(dim=1, keepdim=True, unbiased=True)

    advantages = centred / (std + eps) if normalise == "std" else centred

    # A degenerate group is one with no within-group spread. Detected on the raw reward
    # rather than on the advantage, so the count means the same thing under either
    # normalisation and stays comparable across runs.
    degenerate = (rewards.max(dim=1).values - rewards.min(dim=1).values) == 0

    return AdvantageResult(
        advantages=advantages,
        n_groups=rewards.shape[0],
        n_degenerate=int(degenerate.sum().item()),
        reward_mean=float(rewards.mean().item()),
        reward_std=float(rewards.std().item()) if rewards.numel() > 1 else 0.0,
        advantage_abs_mean=float(advantages.abs().mean().item()),
    )


def wasted_generation_tokens(
    completion_lengths: torch.Tensor,
    rewards: torch.Tensor,
) -> int:
    """Generated tokens that sat inside zero-advantage groups.

    This is the waste figure in the unit that actually bills: tokens, which convert
    directly to GPU-seconds and therefore to money. Reporting waste as a fraction of
    groups understates it whenever degenerate groups run long -- and all-wrong groups
    routinely do, because a model that cannot solve a problem tends to ramble at the
    token limit rather than stop.
    """
    if completion_lengths.shape != rewards.shape:
        raise ValueError(
            f"length/reward shape mismatch: {tuple(completion_lengths.shape)} vs {tuple(rewards.shape)}"
        )
    degenerate = (rewards.max(dim=1).values - rewards.min(dim=1).values) == 0
    return int(completion_lengths[degenerate].sum().item())
