#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 直接修改这里，然后运行：bash train.sh
PYTHON=".venv/bin/python"        # 也可以改成当前环境的 python
CONFIG="configs/trimodal_release.yaml"
DEVICE="cuda:0"
OUTPUT_DIR=".runs/my-train"

# 填写已发布的 Release 路径，保持 YAML 列表格式。
DATA_ROOTS='[data/processed/omg-text/release, data/processed/audio/aistpp/release, data/processed/audio/finedance/release, data/processed/audio/gdance/release]'
T5_PATH="model/t5-base-local"                     # 冻结 T5，训练时按 batch 在线编码
DYNAMIC_STATS=".runs/trimodal-dynamics-stats.json"  # 需先生成的动力学统计 JSON，不是特征缓存

BATCH_SIZE=8                  # 每个微批次的样本数
ACCUMULATION_STEPS=2           # 有效 batch = BATCH_SIZE × ACCUMULATION_STEPS
STEPS=100000                  # 优化器更新总步数；数据按 epoch 打乱遍历，用完一轮再重复
LEARNING_RATE=0.0001
WARMUP_STEPS=1000
PRECISION="bf16"              # bf16 / fp16 / fp32；CPU 使用 fp32
NUM_WORKERS=4
SEED=42
CHECKPOINT_EVERY=1000
VALIDATION_EVERY=100           # 没有 val 数据时设为 0
LOG_EVERY=10
RESUME=null                   # 继续训练时填写 checkpoint 路径，例如 .runs/my-train/last.pt

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m msc.train --config "$CONFIG" \
  "data.roots=$DATA_ROOTS" \
  "data.num_workers=$NUM_WORKERS" \
  "model.t5_path=$T5_PATH" \
  "model.dynamic_stats_path=$DYNAMIC_STATS" \
  "train.device=$DEVICE" \
  "train.output_dir=$OUTPUT_DIR" \
  "train.batch_size=$BATCH_SIZE" \
  "train.accumulation_steps=$ACCUMULATION_STEPS" \
  "train.steps=$STEPS" \
  "train.schedule_steps=$STEPS" \
  "train.learning_rate=$LEARNING_RATE" \
  "train.warmup_steps=$WARMUP_STEPS" \
  "train.precision=$PRECISION" \
  "train.seed=$SEED" \
  "train.checkpoint_every=$CHECKPOINT_EVERY" \
  "train.validation_every=$VALIDATION_EVERY" \
  "train.log_every=$LOG_EVERY" \
  "train.resume=$RESUME" \
  "$@"
