#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

NUM_ENVS="${1:-1024}"
MAX_ITERS="${2:-1500}"
SEED="${3:-1}"

RUN_NAME="p71_himloco_flat_fast_stride_${MAX_ITERS}"

echo "[$(date)] 启动 Stage 1 大步幅高速直行 (1.5~1.8 m/s, gait_span=0.12) 500 轮强化训练..."
echo "配置: Task=Unitree-Go2-Flat, Envs=${NUM_ENVS}, Iters=${MAX_ITERS}, Seed=${SEED}, Resume=p69_himloco_flat_surgical_1000/model_999.pt, RunName=${RUN_NAME}"

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go2-Flat \
  --agent himloco_rsl_rl_cfg \
  --experiment_name go2_rough \
  --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --resume \
  --load_run ".*p69_himloco_flat_surgical_1000" \
  --checkpoint "model_999.pt" \
  --headless \
  "${@:4}"

