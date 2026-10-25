#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${MSC_PYTHON:-.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/path/to/step_XXX.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# EGL 使用物理 GPU 编号；CUDA_VISIBLE_DEVICES 不负责重编号 EGL。
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"

DATA_ROOT="${DATA_ROOT:-}"                 # 留空：使用 checkpoint 中的数据集
MOTION_MANIFEST="${MOTION_MANIFEST:-}"     # 留空：继承子集；none：整个指定数据集
SPLIT="${SPLIT:-train}"
TASK="${TASK:-text}"                     # text / audio / video
SEED="${SEED:-42}"
NUM_MOTIONS="${NUM_MOTIONS:-10}"
RTC="${RTC:-false}"                      # true / false
FEEDBACK="${FEEDBACK:-generated}"        # generated / sonic
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
CFG_SCALE="${CFG_SCALE:-1.0}"
REPLAN_FRAMES="${REPLAN_FRAMES:-15}"
RTC_ALPHA="${RTC_ALPHA:-0.9}"
RTC_MASK="${RTC_MASK:-exponential}"
DEVICE="${DEVICE:-cuda:0}"
PRECISION="${PRECISION:-bf16}"
VIDEO="${VIDEO:-true}"
PANEL_SIZE="${PANEL_SIZE:-512}"
CAPTION_FONT="${CAPTION_FONT:-}"          # 可选 TTF/TTC 字体；默认优先使用 Noto CJK
SONIC_CONFIG="${SONIC_CONFIG:-configs/data_pipeline/omg_text.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-.runs/msc-full-eval}"

ARGS=(--checkpoint "$CHECKPOINT" --split "$SPLIT" --task "$TASK"
  --seed "$SEED" --num-motions "$NUM_MOTIONS" --feedback "$FEEDBACK"
  --inference-steps "$INFERENCE_STEPS" --cfg-scale "$CFG_SCALE"
  --replan-frames "$REPLAN_FRAMES" --rtc-alpha "$RTC_ALPHA" --rtc-mask "$RTC_MASK"
  --device "$DEVICE" --precision "$PRECISION" --panel-size "$PANEL_SIZE"
  --sonic-config "$SONIC_CONFIG" --output "$OUTPUT_DIR")
[[ -z "$DATA_ROOT" ]] || ARGS+=(--data-root "$DATA_ROOT")
[[ -z "$MOTION_MANIFEST" ]] || ARGS+=(--motion-manifest "$MOTION_MANIFEST")
[[ -z "$CAPTION_FONT" ]] || ARGS+=(--caption-font "$CAPTION_FONT")
case "$RTC" in
  true|1) ARGS+=(--rtc) ;;
  false|0) ARGS+=(--no-rtc) ;;
  *) echo 'RTC must be true or false' >&2; exit 2 ;;
esac
case "$VIDEO" in
  true|1) ARGS+=(--video) ;;
  false|0) ARGS+=(--no-video) ;;
  *) echo 'VIDEO must be true or false' >&2; exit 2 ;;
esac
exec "$PYTHON" -u -m msc.eval_full "${ARGS[@]}" "$@"
