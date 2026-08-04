"""Summarise runs from their event logs.

Reads only runs/<name>/events.jsonl, so every number here is reconstructible from the
on-disk record without rerunning anything. That is the point of an append-only log: a
claim you cannot regenerate from the artefact is a claim you are asking to be believed.

Step 0 is excluded from timing by default -- it absorbs allocator growth, autotuning and
cache warmup, and including it flatters whichever backend has the slower startup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((run_dir / "meta.json").read_text())
    steps = []
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") == "step":
            steps.append(rec)
    return meta, steps


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarise(run_dir: Path, skip_first: bool) -> dict | None:
    meta, steps = load(run_dir)
    if not steps:
        return None
    used = steps[1:] if skip_first and len(steps) > 1 else steps
    if not used:
        return None

    cfg = meta.get("config", {})
    phases: dict[str, float] = {}
    for s in used:
        for phase, secs in (s.get("device_s") or {}).items():
            phases[phase] = phases.get(phase, 0.0) + secs
    for k in phases:
        phases[k] /= len(used)

    gen_tokens = mean([s["tokens_generated"] for s in used])
    roll_dev = phases.get("rollout", 0.0)

    return {
        "run": run_dir.name,
        "backend": cfg.get("rollout_backend", "?"),
        "steps": len(used),
        "step_s": mean([s["wall_s"] for s in used]),
        "rollout_dev_s": roll_dev,
        "rollout_share": roll_dev / mean([s["wall_s"] for s in used]) if used else 0.0,
        "sync_s": phases.get("weight_sync", 0.0),
        "gen_tok_s": gen_tokens / roll_dev if roll_dev else 0.0,
        "degenerate_frac": mean([s["degenerate_frac"] for s in used]),
        "wasted_token_frac": mean([s["wasted_token_frac"] for s in used]),
        "reward": mean([s["reward_mean"] for s in used]),
        "mem_gb": max(s["mem_peak_alloc_gb"] for s in used),
        "gpu": meta.get("environment", {}).get("gpu", "?"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--include-first-step", action="store_true")
    args = ap.parse_args()

    root = Path(args.runs_dir)
    dirs = [root / r for r in args.runs] if args.runs else sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "events.jsonl").exists()
    )

    rows = [r for d in dirs if (r := summarise(d, not args.include_first_step))]
    if not rows:
        print("no runs with step events found")
        return 1

    hdr = f"{'run':<16}{'backend':<9}{'n':>3}{'step_s':>9}{'rollout_s':>11}{'share':>7}{'sync_s':>8}{'tok/s':>9}{'degen':>7}{'waste':>7}{'mem_GB':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['run']:<16}{r['backend']:<9}{r['steps']:>3}{r['step_s']:>9.2f}"
            f"{r['rollout_dev_s']:>11.2f}{r['rollout_share']:>6.0%}{r['sync_s']:>8.3f}"
            f"{r['gen_tok_s']:>9.0f}{r['degenerate_frac']:>7.0%}{r['wasted_token_frac']:>7.0%}"
            f"{r['mem_gb']:>8.1f}"
        )

    hf = [r for r in rows if r["backend"] == "hf"]
    vl = [r for r in rows if r["backend"] == "vllm"]
    if hf and vl:
        h, v = hf[-1], vl[-1]
        print()
        print(f"step time:      {h['step_s']:.2f}s -> {v['step_s']:.2f}s   ({h['step_s'] / v['step_s']:.2f}x)")
        print(f"rollout device: {h['rollout_dev_s']:.2f}s -> {v['rollout_dev_s']:.2f}s   ({h['rollout_dev_s'] / v['rollout_dev_s']:.2f}x)")
        print(f"generation:     {h['gen_tok_s']:.0f} -> {v['gen_tok_s']:.0f} tok/s   ({v['gen_tok_s'] / h['gen_tok_s']:.2f}x)")
        print(f"rollout share:  {h['rollout_share']:.0%} -> {v['rollout_share']:.0%} of step")
        print()
        # The speedup is bounded by how much of the step was rollout to begin with.
        # Stating the ceiling stops the next optimisation being aimed at the wrong phase.
        ceiling = 1 / (1 - h["rollout_share"]) if h["rollout_share"] < 1 else float("inf")
        print(f"Amdahl ceiling from removing rollout entirely: {ceiling:.1f}x")
        print(f"achieved: {h['step_s'] / v['step_s']:.2f}x -- remaining step time is now "
              f"{1 - v['rollout_share']:.0%} non-rollout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
