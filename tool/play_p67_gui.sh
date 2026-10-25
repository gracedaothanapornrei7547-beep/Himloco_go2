#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

export DISPLAY="${DISPLAY:-:0}"
export HIMLOCO_KEEP_DISPLAY=1

NUM_ENVS="${1:-16}"
LATEST_RUN_DIR="/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-10-13_20-21-32_p67_project_robo_min_speed15"

if [[ -f "${LATEST_RUN_DIR}/model_299.pt" ]]; then
  CHECKPOINT="${LATEST_RUN_DIR}/model_299.pt"
else
  CHECKPOINT=$(ls -v "${LATEST_RUN_DIR}"/model_*.pt 2>/dev/null | tail -n 1)
fi

echo "[$(date)] 启动 Phase 67 Isaac Lab 图形界面可视化 (DISPLAY=${DISPLAY}, Envs=${NUM_ENVS}, Checkpoint=${CHECKPOINT})..."

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go2-Velocity \
  --agent himloco_rsl_rl_cfg \
  --checkpoint "${CHECKPOINT}" \
  --num_envs "${NUM_ENVS}" \
  --real-time \
  --heading_mode \
  --cmd_vx 1.0 \
  "${@:2}"

