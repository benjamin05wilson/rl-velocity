"""Standard-library replay. Never infer utilization, equivalence or Amdahl bounds."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load(run_dir: Path):
    meta = json.loads((run_dir / 'meta.json').read_text())
    steps = [json.loads(line) for line in (run_dir / 'events.jsonl').read_text().splitlines() if line.strip()]
    return meta, [s for s in steps if s.get('kind') == 'step']


def mean(values):
    return sum(values) / len(values) if values else None


def summarise(run_dir: Path, skip_first: bool = True):
    meta, steps = load(run_dir)
    # Exclude actual step zero, even in a one-step or interrupted run.
    used = [s for s in steps if not skip_first or s['step'] != 0]
    if not used:
        return None
    schemas = {s.get('timing_schema', 'legacy_phase_sum') for s in used}
    if len(schemas) != 1:
        raise ValueError('mixed timing schemas in one run')
    schema = schemas.pop()
    walls = [s.get('wall_s') for s in used]
    if any(v is None or v <= 0 for v in walls):
        raise ValueError('positive wall_s required for every included step')
    hosts = {}
    devices = {}
    for source, target in [('phase_host_s', hosts), ('device_s', devices)]:
        phases = set().union(*(s.get(source, {}) for s in used))
        for phase in phases:
            vals = [s.get(source, {}).get(phase) for s in used]
            target[phase] = mean(vals) if all(v is not None for v in vals) else None
    return dict(run=run_dir.name, provenance=meta.get('provenance', 'unverified run'),
                backend=meta.get('config', {}).get('rollout_backend', '?'), steps=len(used),
                timing_schema=schema, step_s=mean(walls), phase_host_s=hosts, device_s=devices,
                gen_tok_s=sum(s['tokens_generated'] for s in used) / sum(walls),
                reward=mean([s['reward_mean'] for s in used]))


def render_svg(rows):
    """Deterministic standalone evidence figure; separate bars avoid overlap assumptions."""
    metrics = [(r, 'enclosing step' if r['timing_schema'] == 'synchronized_step_v2' else 'legacy phase sum', r['step_s']) for r in rows]
    for r in rows:
        metrics.extend((r, f'{p} host', v) for p, v in sorted(r['phase_host_s'].items()) if v is not None)
    scale = 430 / max(v for _, _, v in metrics)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{100 + 38 * len(metrics)}" role="img">',
             '<title>Recorded host durations; provenance shown per run</title>',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<g font-family="sans-serif" font-size="14" fill="#0f172a">',
             '<text x="24" y="30">Fixture replay — synthetic values, NOT GPU benchmark evidence</text>',
             '<text x="24" y="54">Separate host-duration bars; no utilization or additive phase claim.</text>']
    for i, (r, name, v) in enumerate(metrics):
        y = 88 + i * 38
        label = html.escape(f"{r['run']}: {name}")
        lines.extend([f'<text x="24" y="{y + 16}">{label}</text>',
                      f'<rect x="340" y="{y}" width="{v * scale:.2f}" height="23" fill="#2563eb"/>',
                      f'<text x="{350 + v * scale:.2f}" y="{y + 16}">{v:.2f} s</text>'])
    return '\n'.join(lines + ['</g></svg>']) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--runs-dir', default='runs')
    ap.add_argument('--include-first-step', action='store_true')
    ap.add_argument('--svg', type=Path, help='fixture-only plot destination')
    args = ap.parse_args()
    root = Path(args.runs_dir)
    dirs = [root / r for r in args.runs] if args.runs else sorted(d for d in root.iterdir() if (d / 'meta.json').exists())
    rows = [r for d in dirs if (r := summarise(d, not args.include_first_step))]
    if not rows:
        print('no steps after warmup exclusion')
        return 1
    print('Step 0 excluded by default. tok/s denominator = same recorded wall_s column.')
    print('Legacy rows are partial phase sums, NOT end-to-end measurements. No automatic speedup comparison.')
    print('run | provenance | timing schema | n | recorded seconds | generated tok/s | reward')
    for r in rows:
        print(f"{r['run']} | {r['provenance']} | {r['timing_schema']} | {r['steps']} | {r['step_s']:.3f} | {r['gen_tok_s']:.3f} | {r['reward']:.3f}")
    if args.svg:
        if any(r['provenance'] != 'synthetic fixture' for r in rows):
            ap.error('--svg is reserved for the labeled synthetic fixture')
        args.svg.write_text(render_svg(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
