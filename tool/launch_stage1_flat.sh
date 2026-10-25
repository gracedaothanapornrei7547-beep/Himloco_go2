#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

RUN_NAME="p68_stage1_flat_ground_300"
LOG_FILE="/home/rum/himloco_lab/logs/${RUN_NAME}.log"
NUM_ENVS=1024
MAX_ITERS="${1:-300}"

echo "[$(date)] 准备启动 Stage 1 纯平地基线训练 (300 轮，定型完美站立小跑)..."
echo "配置: Envs=${NUM_ENVS}, MaxIterations=${MAX_ITERS}, Task=Unitree-Go2-Flat"

export PYTHONUNBUFFERED=1

CMD=(
  /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/train.py
  --task Unitree-Go2-Flat
  --agent himloco_rsl_rl_cfg
  --experiment_name go2_rough
  --run_name "${RUN_NAME}"
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERS}"
  --seed 1
  --headless
)

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "[$(date)] 启动成功！PID=${PID}, 日志文件=${LOG_FILE}"

