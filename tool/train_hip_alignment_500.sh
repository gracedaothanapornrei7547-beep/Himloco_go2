#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

NUM_ENVS="${1:-1024}"
MAX_ITERS="${2:-500}"
SEED="${3:-1}"

RUN_NAME="p73_himloco_hip_posture_alignment_500"

echo "[$(date)] 启动 500 轮髋关节构型纠偏（根除平地内八与楼梯外八）训练..."
echo "配置: Task=Unitree-Go2-Velocity, Envs=${NUM_ENVS}, Iters=${MAX_ITERS} (7400->7900), Seed=${SEED}, Resume=p72_himloco_stage2_stairs_assault_3000/model_7399.pt, RunName=${RUN_NAME}"

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go2-Velocity \
  --agent himloco_rsl_rl_cfg \
  --experiment_name go2_rough \
  --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --resume \
  --load_run ".*p72_himloco_stage2_stairs_assault_3000" \
  --checkpoint "model_7399.pt" \
  --reset_terrain \
  --headless \
  "${@:4}"

