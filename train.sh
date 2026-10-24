#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Edit the YAML for batch size, steps, data roots and statistics paths.
PYTHON="${MSC_PYTHON:-.venv/bin/python}"
CONFIG="configs/msc_v3_2xa6000.yaml"
NUM_GPUS=2
MASTER_PORT="${MASTER_PORT:-29563}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Direct P2P stalled on this host; shared-memory NCCL passed the dual-GPU test.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.runs/msc-v3-compiler-cache}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Reuse allocation space when text and audio batch shapes alternate.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  --nproc_per_node="$NUM_GPUS" -m msc.train --config "$CONFIG" "$@"
