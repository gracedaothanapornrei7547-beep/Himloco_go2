#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

NUM_ENVS="${1:-1024}"
MAX_ITERS="${2:-5200}"
SEED="${3:-1}"

RUN_NAME="p72_himloco_stage2_stairs_assault_3000"

echo "[$(date)] 启动 Stage 2 多障碍与 10cm~15cm 台阶 3000 轮攻坚训练..."
echo "配置: Task=Unitree-Go2-Velocity, Envs=${NUM_ENVS}, Iters=${MAX_ITERS} (2200->5200), Seed=${SEED}, Resume=p71_himloco_flat_fast_stride_1500/model_2200.pt, RunName=${RUN_NAME}"

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go2-Velocity \
  --agent himloco_rsl_rl_cfg \
  --experiment_name go2_rough \
  --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --resume \
  --load_run ".*p71_himloco_flat_fast_stride_1500" \
  --checkpoint "model_2200.pt" \
  --reset_terrain \
  --headless \
  "${@:4}"
