#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

RUN_NAME="p67_project_robo_min_speed15"
LOG_FILE="/home/rum/himloco_lab/logs/${RUN_NAME}.log"
NUM_ENVS=1024
MAX_ITERS="${1:-5000}"
RESUME_FLAG="${2:-false}"

echo "[$(date)] 准备启动 Phase 67 (Project-Robo Min采样 + 1.5m/s 高速并发探索)..."
echo "配置: Envs=${NUM_ENVS}, MaxIterations=${MAX_ITERS}, Resume=${RESUME_FLAG}"

export PYTHONUNBUFFERED=1

CMD=(
  /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/train.py
  --task Unitree-Go2-Velocity
  --agent himloco_rsl_rl_cfg
  --experiment_name go2_rough
  --run_name "${RUN_NAME}"
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERS}"
  --seed 1
  --headless
)

if [[ "${RESUME_FLAG}" == "true" ]]; then
  CMD+=(
    --resume True
    --load_run p66_project_robo_cold_5000
    --checkpoint model_4999.pt
  )
fi

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "[$(date)] 启动成功！PID=${PID}, 日志文件=${LOG_FILE}"
