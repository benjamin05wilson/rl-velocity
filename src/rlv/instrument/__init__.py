"""Measurement layer. Deliberately the first thing built, not the last.

The research question this repo exists to answer -- where does RL training compute
actually go, and how much of it buys no learning -- is unanswerable without timing
that survives CUDA's async execution model and a log that survives a crashed run.
Everything else in the repo is downstream of these two files.
"""

from rlv.instrument.clock import MemoryProbe, PhaseStats, PhaseTimer
from rlv.instrument.recorder import Recorder, StepAccount, environment_fingerprint

__all__ = [
    "MemoryProbe",
    "PhaseStats",
    "PhaseTimer",
    "Recorder",
    "StepAccount",
    "environment_fingerprint",
]
