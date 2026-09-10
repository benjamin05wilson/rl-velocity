# Reproduction, metrics and limitations

## Evidence index

- [Synthetic manifest and raw inputs](../evidence/MANIFEST.md): runnable with the Python
  3.9+ standard library, offline and without installation. These are fixture values only.
- [Replay transcript](../evidence/replay.txt), [phase figure](../evidence/phase-times.svg),
  [comparison diagnostics](../evidence/comparison/report.json): deterministic derived outputs.
- [CPU regressions](../tests/test_readiness.py) and [advantage tests](../tests/test_advantages.py):
  controlled tensors/stubs; no downloaded model or dataset. Torch and pytest required.
- [Sampling investigation](sampling-config-inheritance.md) and
  [allocation decision](adaptive-allocation-negative.md): historical observations without
  recovered raw receipts, numerical results withdrawn.

Artifact files are written as UTF-8 with LF line endings. `.gitattributes` keeps checkouts
consistent. The SVG writer has a regression for explicit encoding/newlines: a first Windows
CI run passed CPU tests but exposed a locale-encoded em dash in the generated plot. The
byte-stability fix retains the CI evidence-drift gate. This is a portability repair, not a
change to any fixture values.

## Measurement contract (new events only)

`timing_schema=synchronized_step_v2` identifies **new** enclosing timing:

- Synchronize the current CUDA device **before** starting the host clock, and synchronize
  again **before** stopping it. Include prompt construction, weight sync, generation,
  grading, advantages, loss/gradient computation, optimizer update, scalar extraction,
  phase resolution and memory/metric collection inside those boundaries.
- Exclude model/tokenizer/data load, initial synchronization, held-out evaluation, and
  recorder/console I/O after the measured step. This is step compute latency, not whole-job
  duration; `run_end.elapsed_s` includes setup and logging since recorder creation.
- `wall_s` is that enclosing host duration. `generated_tokens_per_s` divides generated
  tokens by **enclosing step time**. Offline analysis uses sum(tokens)/sum(wall_s) for
  the included steps, not an unweighted mean of per-step rates.
- `phase_host_s` records selected host intervals without per-phase synchronization.
  `device_s` records CUDA event spans on the **current stream**; cross-stream engine
  work may not be bounded by those events. The enclosing device synchronization drains
  local-device work, but does not measure work in another process or another device.
  Engine integration and actual stream behavior still need hardware validation.
- Host and CUDA-event times are not an additive partition. Their difference is not SM
  utilization, busy time, idle time or starvation. There is no automatic Amdahl calculation.
- CPU event/device memory telemetry is null. KL, entropy and prompt-token telemetry are
  null when not measured. The legacy `mem_frag_gb` field is a difference of allocator
  peaks, **not** a direct fragmentation measurement. Tokens from degenerate groups are
  zero-advantage tokens, not a measure of recoverable compute.

Legacy events without the schema are labeled `legacy_phase_sum`; old `wall_s` is not
reinterpreted as synchronized end-to-end time. Mixed schemas inside a run are rejected.
Step zero is excluded by identifier, including single-step runs; `--include-first-step`
opts in. Missing per-phase values stay unavailable. No automatic backend speedup is emitted.
New measurements cannot validate the withdrawn historical headline figures retroactively.

## CPU environment

Locally checked: macOS ARM64, Python 3.12.13, Torch 2.14.0, pytest 9.1.1, Ruff 0.16.6.
[Exact installed versions](../requirements/cpu-tested.txt) constrain CPU checks. The lean
install emits a PyTorch warning about optional NumPy being absent; these tests do not use
the NumPy bridge. No Transformers, datasets or vLLM installation is required by CPU tests.
The base distribution has no ML runtime dependencies. `cpu`, `train`, `vllm` and `dev`
extras separate the responsibilities and declare bounded version ranges.

[CI](../.github/workflows/cpu.yml) runs tests, lint, replay and checked-in artifact drift
checks on Linux, Windows and macOS for pull requests and main pushes. CI does not establish
GPU compatibility or successful training. Exact remote results are in the PR checks;
local execution cannot substitute for a Windows or Linux runtime result.

## GPU candidate recipe

**Unvalidated candidate, not a known-working reconstruction.** Original full dependency
and driver records were not recovered. The older note named Transformers 5.14.1,
vLLM 0.26.0 and Blackwell sm_120; historical `env.sh` mentioned WSL2, FlashInfer 0.6.14,
a CUDA-13 PyTorch build and a separately supplied nvcc. Those recollections do not identify
a reproducible CUDA toolkit/driver pair.

For an explicit future starting point, dependency metadata was resolved on 2026-09-10 for
**Linux x86-64, glibc 2.39 (e.g. Ubuntu 24.04), CPython 3.12** into
[`gpu-linux-candidate.txt`](../requirements/gpu-linux-candidate.txt). It pins the full resolved
package set. vLLM 0.26.0's [PyPI metadata](https://pypi.org/pypi/vllm/0.26.0/json) requires
Torch 2.11.0 and FlashInfer 0.6.14; the candidate is consistent with those requirements.
It was **not installed**, and CUDA extensions were not imported or executed. Wheel-only
resolution for glibc 2.28 failed on llguidance wheels; that older target is not claimed.

Prerequisites: a compatible NVIDIA GPU with enough free VRAM for the selected batch/model
(the historical target was one 24 GB card), a driver compatible with the selected CUDA
wheels, and any toolkit/compiler needed by their JIT kernels. Verify these using the
versioned [vLLM installation guidance](https://docs.vllm.ai/en/v0.26.0/getting_started/installation/)
and actual installed wheel metadata. Windows-native and macOS GPU training are not supported
by this candidate; WSL is a Linux environment with additional driver/kernel constraints.

Commands below are for **bash from the repository root** and may download large packages,
weights and dataset files. They are not part of the offline demo or readiness validation:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv --python 3.12 .venv-gpu
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv-gpu/bin/python -r requirements/gpu-linux-candidate.txt
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv-gpu/bin/python --no-deps -e .
export RLV_PYTHON="$PWD/.venv-gpu/bin/python"
source ./env.sh
"$RLV_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
"$RLV_PYTHON" -m rlv.train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision "$RLV_MODEL_REVISION" \
  --dataset-revision "$RLV_DATASET_REVISION" \
  --rollout-backend hf --steps 2 --train-prompts 4 \
  --prompts-per-step 1 --group-size 2 --max-new-tokens 32 \
  --micro-batch 1 --temperature 1 --run-name hf-smoke-001
```

Expected success signal: `run complete -> runs/hf-smoke-001/events.jsonl` and a final
`run_end` event with `status=completed`. Inspect `meta.json` and both steps; replay excludes
step zero. Generation length and step count are bounded, but initialization/downloads may
still take time. Reusing the name raises a collision error; choose a new name for each attempt.
No checkpoint resume is implemented. Setup/training failures leave their status in the log.

For a separate vLLM smoke attempt, use the same command with `--rollout-backend vllm`
and `--run-name vllm-smoke-001`. `env.sh` sets in-process workers and an explicit sampler
choice; the current-device synchronized timer assumes that topology. A successful smoke
would still not prove sampler equivalence or a speedup. Nonneutral repetition-penalty
experiments require `--allow-off-policy`; matching temperature scoring does not correct
an intentionally nonneutral penalty.

The [artifact revision manifest](../requirements/artifact-revisions.json) records model
and GSM8K commit SHAs resolved from metadata on 2026-09-10. They are **new candidate pins**,
not the missing historical revisions. Model, tokenizer and vLLM use the same model revision;
the dataset revision is separately passed and stored in run config.

`env.sh` uses repo-local caches/interpreter paths and does not guess a CUDA installation.
Set `RLV_CUDA_HOME` only to a verified installed toolkit if needed. For WSL, opt into the
historical pinned-memory workaround with `RLV_WSL_WORKAROUNDS=1` **before sourcing**, after
checking kernel/driver support. The sampler flag defaults to disabling FlashInfer sampling;
this choice is recorded and must stay matched across comparisons. Shell sweep scripts are
Linux research utilities requiring GNU `timeout`; they are not the bounded first demo.

`scripts/verify_gpu.py` is an optional **sm_120-specific** GPU probe. Its architecture
expectations are not a generic CUDA requirement, and its matmul-rate heuristic cannot prove
tensor-core utilization. Do not run it on a CPU or interpret its success as training validation.

## Outstanding external requirements

Original run provenance remains unavailable; historical figures stay withdrawn. Any stronger
performance/learning claim requires new archived GPU runs, working CUDA/driver/JIT validation,
backend sampling checks, and a successful positive control. Owner licence selection is pending;
no licence terms were changed. There is no package release or multi-GPU/adaptive product promise.
