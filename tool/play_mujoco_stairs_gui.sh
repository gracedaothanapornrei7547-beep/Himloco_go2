#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

export DISPLAY="${DISPLAY:-:0}"
export HIMLOCO_KEEP_DISPLAY=1
export MUJOCO_GL="glfw"

SCENE="${1:-stairs_5steps_15cm}"
VX="${2:-0.8}"
DURATION="${3:-0}"

LATEST_RUN_DIR=$(ls -td /home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/*p73* /home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/*stage2* 2>/dev/null | head -n 1)

if [[ -z "${LATEST_RUN_DIR:-}" ]]; then
  echo "未找到 stage2 相关训练目录！" >&2
  exit 1
fi

echo "[$(date)] 启动 MuJoCo Sim2Sim 图形界面回放 (DISPLAY=${DISPLAY}, Scene=${SCENE}, Vx=${VX}m/s, ModelDir=${LATEST_RUN_DIR})..."

exec /home/rum/himloco_lab/tool/run_himloco.sh deploy/sim2sim/sim2sim_mujoco.py \
  --model_dir "${LATEST_RUN_DIR}" \
  --scene "${SCENE}" \
  --vx "${VX}" \
  --duration "${DURATION}" \
  --heading_mode \
  "${@:4}"
