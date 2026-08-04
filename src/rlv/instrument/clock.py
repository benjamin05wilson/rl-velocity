"""Phase timing that does not lie.

Two traps this exists to avoid:

1. `time.perf_counter()` around CUDA work measures kernel *launch* time, not
   execution. Launches are async, so an unsynchronised timer reports microseconds
   for work that takes half a second. Most homegrown RL profiling is wrong this way.

2. The naive fix -- `torch.cuda.synchronize()` around every phase -- is also wrong.
   It serialises the pipeline you are trying to measure, so the act of measuring
   destroys the overlap you care about. Measured throughput drops and you optimise
   against a distorted picture.

The way out is CUDA events: the device timestamps itself in-stream, costing
essentially nothing, and we only pay a sync once per step when we read the numbers
back. Wall time and device time are both recorded, because their *divergence* is
the signal -- a phase whose wall time exceeds its device time is a phase where the
GPU sat idle waiting on Python, and in RL that gap is usually where the wins are.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import time

import torch


@dataclass
class PhaseStats:
    wall_s: float = 0.0
    device_s: float = 0.0
    calls: int = 0

    @property
    def idle_s(self) -> float:
        """Wall time not covered by device work -- the GPU-starvation gap."""
        return max(0.0, self.wall_s - self.device_s)


class PhaseTimer:
    """Accumulates per-phase wall and device time across a training step.

    Usage:
        timer = PhaseTimer()
        with timer("rollout"):
            ...
        timer.resolve()          # single sync, reads device timings back
        timer.stats["rollout"].device_s
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self.stats: dict[str, PhaseStats] = defaultdict(PhaseStats)
        # (phase, start_event, end_event) queued until resolve()
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._resolved = True

    @contextmanager
    def __call__(self, phase: str):
        if not self.enabled:
            t0 = time.perf_counter()
            yield
            self.stats[phase].wall_s += time.perf_counter() - t0
            self.stats[phase].calls += 1
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - t0
            end.record()
            self._pending.append((phase, start, end))
            self._resolved = False
            self.stats[phase].wall_s += wall
            self.stats[phase].calls += 1

    def resolve(self) -> None:
        """Sync once and drain queued device timings. Call at step boundary."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for phase, start, end in self._pending:
            self.stats[phase].device_s += start.elapsed_time(end) / 1000.0
        self._pending.clear()
        self._resolved = True

    def snapshot(self) -> dict[str, dict[str, float]]:
        if not self._resolved:
            self.resolve()
        return {
            phase: {
                "wall_s": round(s.wall_s, 6),
                "device_s": round(s.device_s, 6),
                "idle_s": round(s.idle_s, 6),
                "calls": s.calls,
            }
            for phase, s in self.stats.items()
        }

    def reset(self) -> None:
        if not self._resolved:
            self.resolve()
        self.stats.clear()
        self._pending.clear()


@dataclass
class MemoryProbe:
    """Peak memory per step. Reset each step or the peak is meaningless."""

    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    # Reserved-but-unallocated is fragmentation. On a 24GB card that gap is
    # frequently what stands between you and a larger batch.
    frag_gb: float = 0.0

    @staticmethod
    def start() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @classmethod
    def read(cls) -> MemoryProbe:
        if not torch.cuda.is_available():
            return cls()
        alloc = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        return cls(
            peak_alloc_gb=round(alloc, 3),
            peak_reserved_gb=round(reserved, 3),
            frag_gb=round(reserved - alloc, 3),
        )
