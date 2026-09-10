"""Host phase durations and CUDA events on the current stream.

Event spans include stream waits; they are not SM busy time. Host/event differences
cannot diagnose GPU idleness. A separate enclosing timer synchronizes the current
CUDA device at both boundaries; other processes/devices are outside its scope.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class PhaseStats:
    wall_s: float = 0.0
    device_s: float = 0.0
    calls: int = 0



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
            try:
                yield
            finally:
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
                "device_s": round(s.device_s, 6) if self.enabled else None,
                "calls": s.calls,
            }
            for phase, s in self.stats.items()
        }

    def reset(self) -> None:
        if not self._resolved:
            self.resolve()
        self.stats.clear()
        self._pending.clear()


class StepTimer:
    """Enclosing elapsed time including final device drain, excluding initial drain."""

    def __init__(self):
        self.started = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()

    def stop(self):
        if self.started is None:
            raise RuntimeError("step timer has not started")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.started = None
        return elapsed


@dataclass
class MemoryProbe:
    """Peak memory per step. Reset each step or the peak is meaningless."""

    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    # Difference of allocator peaks, not a direct measurement of fragmentation.
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
