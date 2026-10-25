#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

LOG_FILE="/home/rum/himloco_lab/logs/p66_project_robo_cold_5000.log"

echo "[$(date)] Starting p66 training: 5000 iterations, 1024 envs, seed 1..."

export PYTHONUNBUFFERED=1
nohup /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go2-Velocity \
  --agent himloco_rsl_rl_cfg \
  --experiment_name go2_rough \
  --run_name p66_project_robo_cold_5000 \
  --num_envs 1024 \
  --max_iterations 5000 \
  --seed 1 \
  --headless \
  > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "[$(date)] Launched successfully! PID=${PID}, Log=${LOG_FILE}"
