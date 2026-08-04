# source env.sh
#
# WSL2-specific settings. Keep these in one file rather than scattered through
# scripts, so a run's environment is reproducible and reviewable.

# vLLM disables pinned host memory under WSL by default and its GPU worker then
# fails outright with "UVA is not available", because UvaBuffer requires it.
# Pinned memory has worked on WSL2 kernels since 4.19.121; vLLM still gates it
# behind this opt-in. Ours is 6.18.33, so enable it.
#
# Without this, vLLM does not start at all on WSL -- it is not a tuning knob.
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# FlashInfer resolves the CUDA version by running $CUDA_HOME/bin/nvcc and only falls
# back to torch.version.cuda if that fails. This box has a stale /usr/local/cuda at
# 12.6, so FlashInfer concluded CUDA < 12.9 and refused to emit sm_120 code --
# "SM 12.x requires CUDA >= 12.9" -- while torch was perfectly happy on CUDA 13.
# Point it at the pip-installed nvcc 13.3 instead.
#
# Worth noting the failure shape: torch reporting CUDA 13.0 told us nothing about what
# FlashInfer would decide, because they disagree about where the version comes from.
export CUDA_HOME=/root/rl-velocity/.venv/lib/python3.12/site-packages/nvidia/cu13

# FlashInfer JIT-compiles its sm_120 kernels at engine start and shells out to ninja,
# so ninja must be on PATH -- not merely installed in the venv. Same for nvcc.
export PATH="/root/rl-velocity/.venv/bin:$CUDA_HOME/bin:$PATH"

# FlashInfer 0.6.14 bundles CCCL headers that reject nvcc 13.3 outright --
# "CUDA compiler and CUDA toolkit headers are incompatible" -- but only its sampling
# kernels (sampling.cu, renorm.cu) are JIT-built at startup. Attention compiles and
# captures CUDA graphs without complaint, so disabling just the sampler keeps the
# fast attention path and falls back to the PyTorch top-k/top-p sampler.
#
# This is a real deviation from a stock vLLM install and must be declared in any
# throughput comparison: sampling is not the dominant cost, but a benchmark that
# hides a changed sampler is not a benchmark.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Fragmentation matters more than usual when the trainer and the inference engine
# share one 24GB card; expandable segments reduce the reserved-but-unusable gap.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep HF caches on the WSL ext4 filesystem. Anything under /mnt/c crosses the
# 9p filesystem boundary and turns model loading into a multi-minute affair.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
