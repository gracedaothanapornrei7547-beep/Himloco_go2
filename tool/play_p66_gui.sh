#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

export DISPLAY="${DISPLAY:-:0}"
export HIMLOCO_KEEP_DISPLAY=1

# 默认使用 16 个环境，保证 RTX 4060 (8G) 显存充裕且 3D 渲染帧率流畅
NUM_ENVS="${1:-16}"
CHECKPOINT="/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-10-11_22-36-40_p66_project_robo_cold_5000/model_4999.pt"

echo "[$(date)] 启动 Isaac Lab 图形界面回放 (DISPLAY=${DISPLAY}, Envs=${NUM_ENVS})..."

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go2-Velocity \
  --agent himloco_rsl_rl_cfg \
  --checkpoint "${CHECKPOINT}" \
  --num_envs "${NUM_ENVS}" \
  --real-time \
  --heading_mode \
  --cmd_vx 0.8 \
  "${@:2}"

