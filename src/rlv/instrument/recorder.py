"""Append-only flight recorder for training runs.

RL runs fail hours in, and they fail in ways a loss curve does not explain: entropy
collapses, KL detonates, the reward hacks itself, generation quietly starts emitting
truncated garbage. The usual response is to squint at a dashboard and guess, because
whatever state would have explained it was never written down.

So: every event is appended to JSONL and flushed immediately. A run killed by OOM at
step 4000 still leaves 3999 readable steps on disk. Nothing is buffered in the hope
of a clean shutdown, because there rarely is one.

JSONL over a metrics service on purpose -- it diffs, greps, replays offline, and
survives having no network. Ship to wandb *as well* if you like, never instead.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Everything needed to argue a result is real months later.

    A throughput number without the arch list and driver behind it is an anecdote.
    """
    fp: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    fp["packages"] = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    keys = ("CUDA_HOME", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_WSL2_ENABLE_PIN_MEMORY",
            "VLLM_USE_FLASHINFER_SAMPLER", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")
    fp["runtime_settings"] = {k: os.environ.get(k) for k in keys}
    fp["driver"] = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            fp["driver"] = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        import torch

        fp["torch"] = torch.__version__
        fp["cuda_built"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            fp["gpu"] = props.name
            fp["gpu_capability"] = f"sm_{props.major}{props.minor}"
            fp["gpu_total_gb"] = round(props.total_memory / 1e9, 2)
            fp["gpu_sm_count"] = props.multi_processor_count
            fp["torch_arch_list"] = torch.cuda.get_arch_list()
    except Exception as exc:  # noqa: BLE001
        fp["torch_probe_error"] = str(exc)
    return fp


@dataclass
class StepAccount:
    """Step timing and token accounting; zero-advantage tokens are not GPU savings."""

    step: int
    wall_s: float | None = None
    timing_schema: str = "synchronized_step_v2"
    phase_host_s: dict[str, float] = field(default_factory=dict)

    # --- compute accounting -------------------------------------------------
    tokens_prompt: int | None = None
    tokens_generated: int = 0
    tokens_trained: int = 0
    device_s: dict[str, float | None] = field(default_factory=dict)   # current-stream event span

    # --- the waste signal ---------------------------------------------------
    n_groups: int = 0
    n_groups_degenerate: int = 0      # every completion scored the same -> zero advantage
    tokens_generated_wasted: int = 0  # generation spent inside those groups

    # --- learning signal ----------------------------------------------------
    reward_mean: float = 0.0
    reward_std: float = 0.0
    advantage_abs_mean: float = 0.0
    kl: float | None = None
    entropy: float | None = None
    grad_norm: float = 0.0
    loss: float = 0.0

    # --- memory -------------------------------------------------------------
    mem_peak_alloc_gb: float = 0.0
    mem_frag_gb: float = 0.0

    @property
    def degenerate_frac(self) -> float:
        return self.n_groups_degenerate / self.n_groups if self.n_groups else 0.0

    @property
    def wasted_token_frac(self) -> float:
        return self.tokens_generated_wasted / self.tokens_generated if self.tokens_generated else 0.0

    @property
    def generated_tokens_per_s(self) -> float:
        return self.tokens_generated / self.wall_s if self.wall_s else None


class Recorder:
    """One run, one directory, one append-only event log."""

    def __init__(self, root: str | Path, run_name: str, config: dict[str, Any] | None = None) -> None:
        if not run_name or run_name in (".", "..") or Path(run_name).name != run_name or "\\" in run_name:
            raise ValueError("run_name must be a single directory name")
        self.run_dir = Path(root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.run_name = run_name
        self.t0 = time.time()

        meta = {
            "run_name": run_name,
            "started_unix": self.t0,
            "config": config or {},
            "environment": environment_fingerprint(),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        # Exclusive create in a newly reserved directory; resume is unsupported.
        self._fh = open(self.run_dir / "events.jsonl", "x", buffering=1, encoding="utf-8")  # noqa: SIM115 - closed by Recorder context

    def event(self, kind: str, **payload: Any) -> None:
        rec = {"t": round(time.time() - self.t0, 4), "kind": kind, **payload}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()  # crash at step N+1 must not lose step N

    def step(self, acct: StepAccount) -> None:
        d = asdict(acct)
        # Derived fields are stored, not recomputed at read time, so the log stays
        # self-describing when analysed by something that is not this code.
        d["degenerate_frac"] = round(acct.degenerate_frac, 4)
        d["wasted_token_frac"] = round(acct.wasted_token_frac, 4)
        d["generated_tokens_per_s"] = round(acct.generated_tokens_per_s, 2) if acct.generated_tokens_per_s is not None else None
        self.event("step", **d)

    def note(self, message: str, **payload: Any) -> None:
        """Human-readable marker: config changes, anomalies, manual interventions."""
        self.event("note", message=message, **payload)

    def close(self, status: str = "completed", **payload: Any) -> None:
        self.event("run_end", status=status, elapsed_s=round(time.time() - self.t0, 2), **payload)
        self._fh.close()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # The failure path is the one that matters. Record it in-band.
            self.event("run_error", error_type=exc_type.__name__, error=str(exc))
            self.close(status="failed")
        else:
            self.close(status="completed")
