#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

NUM_ENVS="${1:-1024}"
MAX_ITERS="${2:-1000}"
SEED="${3:-1}"

RUN_NAME="p69_himloco_flat_surgical_${MAX_ITERS}"

echo "[$(date)] 启动修正版 Stage 1 纯平地 1000 轮训练 (Envs=${NUM_ENVS}, Iters=${MAX_ITERS}, Seed=${SEED}, RunName=${RUN_NAME})..."

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go2-Flat \
  --agent himloco_rsl_rl_cfg \
  --experiment_name go2_rough \
  --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --headless \
  "${@:4}"

