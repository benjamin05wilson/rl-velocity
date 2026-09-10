"""Does variance-weighted rollout allocation beat uniform, under a fixed budget?

The measured lift (1.91x) says degeneracy is a property of the prompt, so budget can in
principle be steered. But the obvious policy -- skip prompts predicted degenerate -- is a
bad trade: it avoids 16% of generation while discarding 18.4% of informative groups.

That policy is wrong because the decision is not binary. For a prompt with true pass
rate p, a group of size G is informative unless every completion lands on the same side:

    P(informative | p, G) = 1 - p^G - (1-p)^G

Two consequences. First, shrinking G raises the chance of degeneracy rather than
avoiding it, so "skip" and "keep" are the endpoints of a continuum that should be
optimised over. Second, the marginal value of the (G+1)th completion,

    -p^G ln p - (1-p)^G ln(1-p)

is positive and decreasing in G -- the objective is concave and separable, so greedily
handing each completion to whichever prompt gains most is optimal. No learned predictor
is needed; only an estimate of p per prompt.

Evaluated on the counts from measure_degeneracy_predictability.py. Crucially p is
estimated on early rounds and scored on held-out later rounds, because estimating and
evaluating on the same data would manufacture a result out of estimation noise.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path


def p_informative(p: float, g: int) -> float:
    if g <= 1:
        return 0.0
    return 1.0 - p**g - (1.0 - p) ** g


def marginal_gain(p: float, g: int) -> float:
    """Increase in P(informative) from adding one more completion."""
    return p_informative(p, g + 1) - p_informative(p, g)


def greedy_allocate(ps: list[float], budget: int, g_min: int, g_max: int) -> list[int]:
    """Maximise expected informative groups subject to sum(G_i) == budget.

    Concave separable objective, so greedy by marginal gain is optimal.
    """
    n = len(ps)
    alloc = [g_min] * n
    spent = g_min * n
    if spent > budget:
        raise ValueError("g_min * n exceeds budget")

    heap = [(-marginal_gain(ps[i], g_min), i) for i in range(n) if g_min < g_max]
    heapq.heapify(heap)
    while spent < budget and heap:
        neg, i = heapq.heappop(heap)
        alloc[i] += 1
        spent += 1
        if alloc[i] < g_max:
            heapq.heappush(heap, (-marginal_gain(ps[i], alloc[i]), i))
    return alloc


def expected_informative(ps: list[float], alloc: list[int]) -> float:
    return sum(p_informative(p, g) for p, g in zip(ps, alloc, strict=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/degeneracy/passes.json")
    ap.add_argument("--fit-rounds", type=int, default=3, help="rounds used to estimate p")
    ap.add_argument("--g-min", type=int, default=2)
    ap.add_argument("--g-max", type=int, default=32)
    args = ap.parse_args()

    blob = json.loads(Path(args.data).read_text())
    passes, G = blob["passes"], blob["G"]
    R, n = len(passes), len(passes[0])
    fit_R = min(args.fit_rounds, R - 1)

    # p estimated on early rounds only, with Laplace smoothing so a prompt that went
    # 0/24 is not assigned p=0 (which would claim it can never be informative).
    est = []
    for i in range(n):
        ok = sum(passes[r][i] for r in range(fit_R))
        tot = fit_R * G
        est.append((ok + 1) / (tot + 2))

    # Ground truth from held-out rounds -- what the allocator is scored against.
    true = []
    for i in range(n):
        ok = sum(passes[r][i] for r in range(fit_R, R))
        tot = (R - fit_R) * G
        true.append(ok / tot)

    budget = G * n     # identical spend for every policy
    uniform = [G] * n
    alloc = greedy_allocate(est, budget, args.g_min, args.g_max)

    u_true = expected_informative(true, uniform)
    a_true = expected_informative(true, alloc)
    # Ceiling: allocate using the held-out truth itself. Unreachable, bounds the gain.
    oracle = greedy_allocate(true, budget, args.g_min, args.g_max)
    o_true = expected_informative(true, oracle)

    print(f"prompts={n}  rounds={R}  G={G}  budget={budget} completions")
    print(f"p estimated on rounds 0-{fit_R - 1}, scored on rounds {fit_R}-{R - 1}\n")

    print("=" * 64)
    print("EXPECTED INFORMATIVE GROUPS PER ROUND (same total spend)")
    print("=" * 64)
    print(f"  uniform G={G}                {u_true:7.1f}  ({u_true / n:.1%} of prompts)")
    print(f"  variance-weighted greedy     {a_true:7.1f}  ({a_true / n:.1%})")
    print(f"  oracle (knows held-out p)    {o_true:7.1f}  ({o_true / n:.1%})")
    print()
    print(f"  gain over uniform            {a_true / u_true:.3f}x")
    print(f"  oracle ceiling               {o_true / u_true:.3f}x")
    frac = (a_true - u_true) / (o_true - u_true) if o_true > u_true else 0.0
    print(f"  fraction of ceiling captured {frac:.1%}")

    # Where did the budget go?
    buckets = {"skipped to g_min": 0, "reduced": 0, "unchanged": 0, "increased": 0}
    for g in alloc:
        if g == args.g_min:
            buckets["skipped to g_min"] += 1
        elif g < G:
            buckets["reduced"] += 1
        elif g == G:
            buckets["unchanged"] += 1
        else:
            buckets["increased"] += 1
    print("\n" + "=" * 64)
    print("ALLOCATION SHAPE")
    print("=" * 64)
    for k, v in buckets.items():
        print(f"  {k:<22} {v:4d} prompts ({v / n:.1%})")
    print(f"  max group size assigned  {max(alloc)}")

    # Sanity: how much headroom exists at all? If most prompts sit near p=0.5,
    # uniform is already near-optimal and there is nothing to win.
    near_half = sum(1 for p in true if 0.3 <= p <= 0.7)
    extreme = sum(1 for p in true if p <= 0.05 or p >= 0.95)
    print(f"\n  prompts with held-out p in [0.3,0.7]  {near_half}/{n} ({near_half / n:.1%})")
    print(f"  prompts with p<=0.05 or p>=0.95       {extreme}/{n} ({extreme / n:.1%})")

    # --- sensitivity: a less crude objective -------------------------------
    # "Number of informative groups" is binary and ignores that a 4/8 split carries
    # more gradient than a 1/8 split. With binary rewards and centred advantages, the
    # total advantage magnitude in a group of size G is 2k(G-k)/G for k correct, and
    # taking the expectation over k ~ Binomial(G, p) gives exactly 2(G-1)p(1-p).
    #
    # That is LINEAR in G, so per completion the signal is ~2p(1-p) regardless of group
    # size. Under this objective the allocator's job is purely to favour prompts with
    # high p(1-p) -- and because it is linear rather than concave, the unconstrained
    # optimum is degenerate (spend everything on one prompt), which is why g_max is
    # doing real work here and why this objective cannot be used naively.
    def signal(p: float, g: int) -> float:
        return 2.0 * (g - 1) * p * (1.0 - p)

    def greedy_signal(ps: list[float], budget: int) -> list[int]:
        order = sorted(range(n), key=lambda i: -ps[i] * (1 - ps[i]))
        alloc_s = [args.g_min] * n
        spent_s = args.g_min * n
        for i in order:
            take = min(args.g_max - args.g_min, budget - spent_s)
            alloc_s[i] += take
            spent_s += take
            if spent_s >= budget:
                break
        return alloc_s

    alloc_s = greedy_signal(est, budget)
    u_sig = sum(signal(true[i], G) for i in range(n))
    a_sig = sum(signal(true[i], alloc_s[i]) for i in range(n))
    o_sig = sum(signal(true[i], g) for i, g in enumerate(greedy_signal(true, budget)))

    print("\n" + "=" * 64)
    print("SENSITIVITY: total advantage magnitude instead of group count")
    print("=" * 64)
    print(f"  uniform                      {u_sig:8.1f}")
    print(f"  p(1-p)-weighted (estimated)  {a_sig:8.1f}   {a_sig / u_sig:.3f}x")
    print(f"  oracle                       {o_sig:8.1f}   {o_sig / u_sig:.3f}x")

    # The shape matters more than the number. A linear objective has a bang-bang
    # optimum, so this "gain" may just be the objective rewarding concentration --
    # something it values and learning does not.
    at_max = sum(1 for g in alloc_s if g >= args.g_max)
    at_min = sum(1 for g in alloc_s if g <= args.g_min)
    covered = sum(1 for g in alloc_s if g >= G)
    print(f"\n  allocation shape: {at_max} prompts at g_max={args.g_max}, "
          f"{at_min} at g_min={args.g_min}")
    print(f"  prompts still receiving a full group (>= {G}): {covered}/{n} ({covered / n:.1%})")
    print(f"  -> effective prompt diversity per step falls to {covered / n:.0%} of uniform")

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    gain, sig_gain = a_true / u_true, a_sig / u_sig
    print(f"  informative-group objective   {gain:.3f}x   (oracle ceiling {o_true / u_true:.3f}x)")
    print(f"  advantage-magnitude objective {sig_gain:.3f}x   (oracle ceiling {o_sig / u_sig:.3f}x)")
    print()
    if gain < 1.02 and sig_gain > 1.2:
        print("  The two objectives disagree, and that IS the result.")
        print()
        print("  Counting informative groups shows no gain, and barely any to be had:")
        print(f"  even a perfect allocator reaches only {o_true / u_true:.3f}x, because")
        print(f"  1 - p^G - (1-p)^G is flat across most of the p range at G={G}. Large")
        print("  groups are already robust to prompt difficulty; only genuinely extreme")
        print(f"  prompts degenerate, and those are {extreme / n:.0%} of the set.")
        print()
        print("  Total advantage magnitude shows a large gain, but it is an artefact.")
        print("  That objective is linear in G, so its optimum is bang-bang, and the")
        print("  allocation it produces collapses prompt diversity to a fifth of uniform.")
        print("  It also assumes signal grows linearly with group size, which is false --")
        print("  advantages are centred within a group, so extra samples of the same")
        print("  prompt are correlated and give diminishing returns the model ignores.")
        print()
        print("  Neither proxy is trustworthy, and they bracket the answer rather than")
        print("  settle it. Resolving this needs an actual training comparison -- which")
        print("  is the experiment whose positive control failed earlier today.")
        print()
        print("  Conclusion: NOT SUPPORTED on this evidence, and not refutable either.")
        print("  Do not build the allocator on the strength of the 1.44x.")
    elif gain < 1.02:
        print("  No material gain. Uniform allocation is already close to optimal for")
        print("  this prompt-difficulty distribution; the idea is not worth building.")
    elif frac < 0.3:
        print("  Real but small, and most of the ceiling is lost to estimation error.")
        print("  A better estimator of p matters more than a better allocator.")
    else:
        print("  Worth building online: the gain survives held-out evaluation and")
        print("  captures a decent share of the oracle ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
