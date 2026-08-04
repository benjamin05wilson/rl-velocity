"""Tests for the advantage computation.

The degeneracy count is the number this repo will make claims about, so it is worth
pinning down precisely -- including the awkward cases where a naive implementation
quietly gets it wrong.
"""

from __future__ import annotations

import torch

from rlv.algo.advantages import group_advantages, wasted_generation_tokens


def test_centring_only_by_default():
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = group_advantages(r)
    assert torch.allclose(out.advantages, torch.tensor([[0.75, -0.25, -0.25, -0.25]]))
    assert out.advantages.sum().abs() < 1e-6  # a baseline must not shift the mean


def test_all_correct_group_is_degenerate():
    out = group_advantages(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    assert out.n_degenerate == 1
    assert out.advantages.abs().max() == 0.0


def test_all_wrong_group_is_degenerate():
    # The failure mode people forget: an all-zero-reward group is just as wasteful as
    # an all-correct one, and early in training it is the common case.
    out = group_advantages(torch.tensor([[0.0, 0.0, 0.0, 0.0]]))
    assert out.n_degenerate == 1
    assert out.advantages.abs().max() == 0.0


def test_mixed_group_is_not_degenerate():
    out = group_advantages(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    assert out.n_degenerate == 0
    assert out.advantages.abs().max() > 0


def test_degenerate_fraction_across_batch():
    r = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],   # degenerate
        [0.0, 0.0, 0.0, 0.0],   # degenerate
        [1.0, 0.0, 0.0, 0.0],   # informative
        [1.0, 1.0, 0.0, 0.0],   # informative
    ])
    out = group_advantages(r)
    assert out.n_groups == 4
    assert out.n_degenerate == 2
    assert out.degenerate_frac == 0.5


def test_std_normalisation_reweights_by_difficulty():
    """The bias that motivates defaulting to 'none'.

    Two groups, one solved once out of four and one solved twice out of four. Under
    centring the successful completions carry 0.75 and 0.5. Dividing by each group's
    own std inflates the rarer success relative to the commoner one, which is a
    difficulty reweighting nobody asked for.
    """
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    plain = group_advantages(r, normalise="none").advantages
    scaled = group_advantages(r, normalise="std").advantages

    assert plain[0, 0] > plain[1, 0]           # 0.75 > 0.5 under centring
    ratio_plain = plain[0, 0] / plain[1, 0]
    ratio_scaled = scaled[0, 0] / scaled[1, 0]
    assert ratio_scaled > ratio_plain          # std division widens the gap further


def test_singleton_group_yields_no_signal_not_nan():
    # Unbiased std of one sample is NaN. Silent NaN here would poison the whole update.
    out = group_advantages(torch.tensor([[1.0]]), normalise="std")
    assert torch.isfinite(out.advantages).all()
    assert out.advantages.abs().max() == 0.0


def test_wasted_tokens_counts_only_degenerate_groups():
    rewards = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    lengths = torch.tensor([[100, 120], [80, 90]])
    assert wasted_generation_tokens(lengths, rewards) == 220


def test_wasted_tokens_weights_long_failures_heavily():
    """Why waste is reported in tokens, not in groups.

    An all-wrong group that rambles to the token limit costs far more than a short
    all-correct one, so a group-count metric understates the real bill.
    """
    rewards = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    lengths = torch.tensor([[512, 512], [40, 45]])
    total = wasted_generation_tokens(lengths, rewards)
    assert total == 1109
    # Both groups are degenerate, but 92% of the wasted tokens come from one of them.
    assert 1024 / total > 0.9
