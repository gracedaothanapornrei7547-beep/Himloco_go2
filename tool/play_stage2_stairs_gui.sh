#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

export DISPLAY="${DISPLAY:-:0}"
export HIMLOCO_KEEP_DISPLAY=1

NUM_ENVS="${1:-5}"
STEP_HEIGHT="${2:-0.15}"
CMD_VX="${3:-0.9}"
LATEST_RUN_DIR=$(ls -td /home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/*stage2* 2>/dev/null | head -n 1)

if [[ -z "${LATEST_RUN_DIR:-}" ]]; then
  echo "未找到 stage2 相关训练目录！" >&2
  exit 1
fi

CHECKPOINT=$(ls -v "${LATEST_RUN_DIR}"/model_*.pt 2>/dev/null | tail -n 1)

echo "[$(date)] 启动 Stage 2 极限 15cm 爬楼梯图形界面回放 (DISPLAY=${DISPLAY}, Envs=${NUM_ENVS}, StepHeight=${STEP_HEIGHT}m, CmdVx=${CMD_VX}m/s, Checkpoint=${CHECKPOINT})..."

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go2-Velocity-Play \
  --agent himloco_rsl_rl_cfg \
  --checkpoint "${CHECKPOINT}" \
  --num_envs "${NUM_ENVS}" \
  --step_height "${STEP_HEIGHT}" \
  --stair_only up \
  --real-time \
  --heading_mode \
  --cmd_vx "${CMD_VX}" \
  "${@:4}"
