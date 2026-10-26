#!/usr/bin/env bash
# Dual A800 80GB profile; all train.sh overrides remain available.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export MSC_PYTHON="${MSC_PYTHON:-$PWD/.runs/envs/omg/bin/python}"
export MSC_CONFIG="${MSC_CONFIG:-configs/msc_v3_2xa800.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export SAMPLING="${SAMPLING:-episode_interleaved}"
export ACTIVE_EPISODES="${ACTIVE_EPISODES:-64}"
export WINDOWS_PER_EPISODE="${WINDOWS_PER_EPISODE:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-.runs/msc-v3-flow-only-2xa800-b2048}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export NUM_THREADS="${NUM_THREADS:-4}"
export NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH:-.runs/msc-v3-normalization-stats.json}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-.runs/data-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.runs/compiler-cache-a800}"
# This container's A800 PCIe P2P was verified with standalone AllReduce.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="$(dirname "$MSC_PYTHON"):$PATH"
if [[ ! -x "$MSC_PYTHON" ]]; then
  printf 'Missing Python environment: %s\n' "$MSC_PYTHON" >&2
  exit 1
fi
exec bash train.sh "$@"
