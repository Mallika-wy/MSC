#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 100 条 Text-Motion 过拟合实验；从 train.sh 复制。
# 修改下列默认值，或用环境变量临时覆盖；末尾的 section.key=value 优先级最高。
# bash train.sh --stats 使用同一套参数生成动力学统计量，不启动训练。

# ---------- Python 与 GPU ----------
PYTHON="${MSC_PYTHON:-.venv/bin/python}"
CONFIG="${MSC_CONFIG:-configs/msc_v3_2xa6000.yaml}" # 共享模型配置；GPU 数量由下面的设置决定
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#GPU_LIST[@]}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29563}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"

# ---------- 输出、训练长度与优化器 ----------
OUTPUT_DIR="${OUTPUT_DIR:-.runs/msc-v3-overfit-text100}"          # 输出目录
RESUME="${RESUME:-null}"                                                  # null 或 checkpoint 路径
SEED="${SEED:-42}"
STEPS="${STEPS:-10000}"                         # optimizer steps
BATCH_SIZE="${BATCH_SIZE:-1536}"               # 每卡 microbatch；A6000 实测峰值约 41.3 GiB
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-0.000001}" # cosine 衰减的最终下限
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
SCHEDULE_STEPS="${SCHEDULE_STEPS:-$STEPS}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
PRECISION="${PRECISION:-bf16}"                   # fp32 / bf16 / fp16
DETERMINISTIC="${DETERMINISTIC:-false}"
COMPILE_EXPERT="${COMPILE_EXPERT:-true}"
BF16_GRADIENT_REDUCE="${BF16_GRADIENT_REDUCE:-true}"
NUM_THREADS="${NUM_THREADS:-4}"

# ---------- 日志、验证与保存 ----------
LOG_EVERY="${LOG_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"      # 保存 checkpoints/step_XXX.pt，保留历次文件
VALIDATION_EVERY="${VALIDATION_EVERY:-0}"       # 0 关闭验证
VALIDATION_BATCHES="${VALIDATION_BATCHES:-2}"
VALIDATION_MOTIONS_PER_TASK="${VALIDATION_MOTIONS_PER_TASK:-128}" # null 表示不限制
WANDB_PROJECT="${WANDB_PROJECT:-msc-v3-overfit-text100}"                       # null 关闭；启用时始终 offline
WANDB_NAME="${WANDB_NAME:-null}"                 # null 使用输出目录名称

# ---------- 数据：列表使用 YAML 格式；修改训练池后重新运行 --stats ----------
# 从原 Release 按清单读取 100 条动作，包含各动作的全部选定有效滑窗。
DATA_ROOTS="${DATA_ROOTS:-[data/processed/omg-text/release]}"
TASKS="${TASKS:-[text]}"
MOTION_MANIFEST="${MOTION_MANIFEST:-configs/overfit_text100_manifest.json}"
DATA_SPLIT="${DATA_SPLIT:-train}"
LABEL_POLICY="${LABEL_POLICY:-train_default}"
ALIGNMENT_POLICY="${ALIGNMENT_POLICY:-allow_estimated}" # allow_estimated / verified_only
NUM_WORKERS="${NUM_WORKERS:-4}"                  # 每卡 DataLoader workers
CACHE_MB="${CACHE_MB:-512}"                      # 每个 worker 的缓存 MiB
AUDIO_CONTEXT_SEC="${AUDIO_CONTEXT_SEC:-0.25}"
VIDEO_CONTEXT_SEC="${VIDEO_CONTEXT_SEC:-0.25}"
MAX_MOTIONS_PER_TASK="${MAX_MOTIONS_PER_TASK:-null}"
MAX_WINDOWS_PER_MOTION="${MAX_WINDOWS_PER_MOTION:-null}"
STATS_WORKERS="${STATS_WORKERS:-8}"              # --stats 的 CPU 进程数

# ---------- 模型：G1 125D / 30 FPS / 5+30 帧为固定契约 ----------
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
NUM_LAYERS="${NUM_LAYERS:-12}"
NUM_HEADS="${NUM_HEADS:-16}"
FFN_DIM="${FFN_DIM:-3072}"
MODEL_DROPOUT="${MODEL_DROPOUT:-0.0}"
TEMPORAL_BIAS="${TEMPORAL_BIAS:-relative_distance}" # relative_distance / none
T5_PATH="${T5_PATH:-model/t5-base-local}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-50}"
TEXT_ENABLED="${TEXT_ENABLED:-true}"
AUDIO_ENABLED="${AUDIO_ENABLED:-true}"
VIDEO_ENABLED="${VIDEO_ENABLED:-false}"
TEXT_DROPOUT="${TEXT_DROPOUT:-0.0}"
AUDIO_DROPOUT="${AUDIO_DROPOUT:-0.0}"
VIDEO_DROPOUT="${VIDEO_DROPOUT:-0.0}"
VIDEO_DIM="${VIDEO_DIM:-auto}"                    # 从真实特征元数据解析
VIDEO_SPATIAL_TOKENS="${VIDEO_SPATIAL_TOKENS:-}"   # 留空时从真实元数据解析
NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH:-null}" # null 使用内置 125D 统计量
RTC="${RTC:-false}"
MAX_DELAY="${MAX_DELAY:-8}"

# ---------- 动力学辅助监督 ----------
DYNAMICS_STATS_PATH="${DYNAMICS_STATS_PATH:-.runs/msc-v3-dynamics-stats.json}"
JOINT_VELOCITY_WEIGHT="${JOINT_VELOCITY_WEIGHT:-0.0}"
JOINT_ACCELERATION_WEIGHT="${JOINT_ACCELERATION_WEIGHT:-0.0}"
DYNAMIC_WARMUP_STEPS="${DYNAMIC_WARMUP_STEPS:-1000}"

# ---------- 通信、编译与显存分配 ----------
# 本机直接 P2P 曾停滞，默认使用已验证的 NCCL 共享内存路径。
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.runs/msc-v3-compiler-cache}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NUM_THREADS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 统一传给配置解析器；使用数组保留路径、名称和 YAML 列表中的空格。
OVERRIDES=(
  "train.device=$TRAIN_DEVICE"
  "train.output_dir=$OUTPUT_DIR"
  "train.resume=$RESUME"
  "train.seed=$SEED"
  "train.steps=$STEPS"
  "train.batch_size=$BATCH_SIZE"
  "train.accumulation_steps=$ACCUMULATION_STEPS"
  "train.learning_rate=$LEARNING_RATE"
  "train.min_learning_rate=$MIN_LEARNING_RATE"
  "train.weight_decay=$WEIGHT_DECAY"
  "train.warmup_steps=$WARMUP_STEPS"
  "train.schedule_steps=$SCHEDULE_STEPS"
  "train.max_grad_norm=$MAX_GRAD_NORM"
  "train.precision=$PRECISION"
  "train.deterministic=$DETERMINISTIC"
  "train.compile_expert=$COMPILE_EXPERT"
  "train.bf16_gradient_reduce=$BF16_GRADIENT_REDUCE"
  "train.num_threads=$NUM_THREADS"
  "train.log_every=$LOG_EVERY"
  "train.checkpoint_every=$CHECKPOINT_EVERY"
  "train.validation_every=$VALIDATION_EVERY"
  "train.validation_batches=$VALIDATION_BATCHES"
  "train.validation_motions_per_task=$VALIDATION_MOTIONS_PER_TASK"
  "train.wandb_project=$WANDB_PROJECT"
  "train.wandb_name=$WANDB_NAME"
  "data.roots=$DATA_ROOTS"
  "data.tasks=$TASKS"
  "data.motion_manifest=$MOTION_MANIFEST"
  "data.split=$DATA_SPLIT"
  "data.label_policy=$LABEL_POLICY"
  "data.alignment_policy=$ALIGNMENT_POLICY"
  "data.num_workers=$NUM_WORKERS"
  "data.cache_bytes=$((CACHE_MB * 1024 * 1024))"
  "data.audio_context_sec=$AUDIO_CONTEXT_SEC"
  "data.video_context_sec=$VIDEO_CONTEXT_SEC"
  "data.max_motions_per_task=$MAX_MOTIONS_PER_TASK"
  "data.max_windows_per_motion=$MAX_WINDOWS_PER_MOTION"
  "model.hidden_dim=$HIDDEN_DIM"
  "model.num_layers=$NUM_LAYERS"
  "model.num_heads=$NUM_HEADS"
  "model.ffn_dim=$FFN_DIM"
  "model.dropout=$MODEL_DROPOUT"
  "model.condition.temporal_bias=$TEMPORAL_BIAS"
  "model.t5_path=$T5_PATH"
  "model.text_max_length=$TEXT_MAX_LENGTH"
  "model.condition.text.enabled=$TEXT_ENABLED"
  "model.condition.audio.enabled=$AUDIO_ENABLED"
  "model.condition.video.enabled=$VIDEO_ENABLED"
  "model.condition.text.dropout=$TEXT_DROPOUT"
  "model.condition.audio.dropout=$AUDIO_DROPOUT"
  "model.condition.video.dropout=$VIDEO_DROPOUT"
  "model.condition.video.dim=$VIDEO_DIM"
  "model.stats_path=$NORMALIZATION_STATS_PATH"
  "model.rtc=$RTC"
  "model.max_delay=$MAX_DELAY"
  "model.dynamic_stats_path=$DYNAMICS_STATS_PATH"
  "model.losses.joint_velocity=$JOINT_VELOCITY_WEIGHT"
  "model.losses.joint_acceleration=$JOINT_ACCELERATION_WEIGHT"
  "model.losses.dynamic_warmup_steps=$DYNAMIC_WARMUP_STEPS"
)
if [[ -n "$VIDEO_SPATIAL_TOKENS" ]]; then
  OVERRIDES+=("model.condition.video.spatial_tokens=$VIDEO_SPATIAL_TOKENS")
fi

if [[ "${1:-}" == --stats ]]; then
  shift
  printf '[MSC] Computing Train dynamics statistics with %s workers.\n' "$STATS_WORKERS"
  # 解析最后的命令行覆盖，确保输出路径与最终训练配置一致。
  STATS_OUTPUT=$("$PYTHON" -c \
    'import sys; from msc.config import load_config; print(load_config(sys.argv[1], sys.argv[2:]).model.dynamic_stats_path)' \
    "$CONFIG" "${OVERRIDES[@]}" "$@")
  exec "$PYTHON" -m msc.data.stats --config "$CONFIG" \
    --workers "$STATS_WORKERS" --output "$STATS_OUTPUT" "${OVERRIDES[@]}" "$@"
fi

printf '[MSC] Python: %s\n[MSC] Config: %s\n[MSC] GPUs: %s; processes: %s\n[MSC] Loading PyTorch and training modules...\n' \
  "$PYTHON" "$CONFIG" "$CUDA_VISIBLE_DEVICES" "$NUM_GPUS"

if (( NUM_GPUS == 1 )); then
  exec "$PYTHON" -u -m msc.train --config "$CONFIG" "${OVERRIDES[@]}" "$@"
fi

exec "$PYTHON" -u -m torch.distributed.run \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$NUM_GPUS" -m msc.train --config "$CONFIG" \
  "${OVERRIDES[@]}" "$@"
