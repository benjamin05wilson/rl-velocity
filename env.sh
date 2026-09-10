# Source from bash: source ./env.sh
# Repository-local defaults. GPU compatibility must still be validated separately.
if [ -z "${BASH_VERSION:-}" ]; then
  echo 'env.sh requires bash' >&2
  return 1
fi
export RLV_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RLV_PYTHON="${RLV_PYTHON:-$RLV_REPO_ROOT/.venv/bin/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RLV_REPO_ROOT/.cache/uv}"
export HF_HOME="${HF_HOME:-$RLV_REPO_ROOT/.cache/huggingface}"
export PATH="$(dirname -- "$RLV_PYTHON"):$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Pin the sampler choice and record it in meta.json. This was a historical workaround;
# it is not a claim that current FlashInfer attention kernels work on every GPU.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Optional WSL workaround, only after checking your kernel/driver supports pinned memory.
if [ "${RLV_WSL_WORKAROUNDS:-0}" = 1 ]; then
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
fi
# An explicitly supplied toolkit overrides CUDA_HOME; never guess a site-packages path.
if [ -n "${RLV_CUDA_HOME:-}" ]; then
  export CUDA_HOME="$RLV_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
# Metadata resolved on 2026-09-10, not recovered historical experiment revisions.
export RLV_MODEL_REVISION="${RLV_MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
export RLV_DATASET_REVISION="${RLV_DATASET_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
