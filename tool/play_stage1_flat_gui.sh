#!/usr/bin/env bash
set -euo pipefail

cd /home/rum/himloco_lab

export DISPLAY="${DISPLAY:-:0}"
export HIMLOCO_KEEP_DISPLAY=1

NUM_ENVS="${1:-16}"
CMD_VX="${2:-1.6}"

LATEST_RUN_DIR=$(ls -td /home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/*flat* 2>/dev/null | head -n 1)

if [[ -z "${LATEST_RUN_DIR:-}" ]]; then
  echo "未找到 flat 相关训练目录！" >&2
  exit 1
fi

if [[ -f "${LATEST_RUN_DIR}/model_2200.pt" ]]; then
  CHECKPOINT="${LATEST_RUN_DIR}/model_2200.pt"
else
  CHECKPOINT=$(ls -v "${LATEST_RUN_DIR}"/model_*.pt 2>/dev/null | tail -n 1)
fi

echo "[$(date)] 启动 Stage 1 大步幅高速平地图形界面回放 (DISPLAY=${DISPLAY}, Envs=${NUM_ENVS}, CmdVx=${CMD_VX}, Checkpoint=${CHECKPOINT})..."

exec /home/rum/himloco_lab/tool/run_himloco.sh scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go2-Flat-Play \
  --agent himloco_rsl_rl_cfg \
  --checkpoint "${CHECKPOINT}" \
  --num_envs "${NUM_ENVS}" \
  --real-time \
  --heading_mode \
  --cmd_vx "${CMD_VX}" \
  "${@:3}"
