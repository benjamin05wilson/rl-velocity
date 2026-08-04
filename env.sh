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

# Fragmentation matters more than usual when the trainer and the inference engine
# share one 24GB card; expandable segments reduce the reserved-but-unusable gap.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep HF caches on the WSL ext4 filesystem. Anything under /mnt/c crosses the
# 9p filesystem boundary and turns model loading into a multi-minute affair.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
