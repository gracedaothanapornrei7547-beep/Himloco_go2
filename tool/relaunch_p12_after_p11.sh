#!/bin/bash
# 等 p11 自然跑完后，自动从零重训 p12（Fix #2 时间门控速度课程 + Fix #3 upward 归一化）。
# p12 与 p11 同配置（1024 envs × 20000 iter，seed 1），唯一差别是已经带上本轮改的源码。
set -u

P11_PATTERN="train.py.*p11_stair10_uniform"
P12_NAME="p12_stair10_uniform_timevel"
P12_LOG="/home/rum/himloco_lab/logs/${P12_NAME}_train.log"
MAX_STALL_SEC=1800    # 30 分钟日志没动 = 卡死
MAX_TOTAL_SEC=115200  # 32h 兜底（实测 ~4.6-5.0s/iter → 20000 iter ≈ 25.6-27.8h + 余量）

echo "[relaunch] $(date) waiting for p11 (train.py.*p11_stair10_uniform) to exit..."
while pgrep -f "$P11_PATTERN" >/dev/null 2>&1; do
    sleep 60
done

echo "[relaunch] $(date) p11 exited. Grace 30s for GPU settle, then launching $P12_NAME..."
sleep 30

cd /home/rum/himloco_lab || exit 1
setsid nohup ./run_himloco.sh scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go2-Velocity \
  --agent himloco_rsl_rl_cfg \
  --experiment_name go2_him_direct \
  --run_name "$P12_NAME" \
  --num_envs 1024 \
  --max_iterations 20000 \
  --seed 1 \
  --headless \
  > "$P12_LOG" 2>&1 < /dev/null &
P12_PID=$!
echo "[relaunch] $(date) launched $P12_NAME PID=$P12_PID log=$P12_LOG"

# --- 进入 p12 看门狗：进程死亡 / 日志停更（疑似 PhysX 卡死）即退出并通知 ---
start=$(date +%s)
last_mtime=$(stat -c %Y "$P12_LOG" 2>/dev/null || echo 0)
while true; do
    now=$(date +%s)
    if ! pgrep -f "train.py.*$P12_NAME" >/dev/null 2>&1; then
        echo "[relaunch] WATCHDOG: p12 training process exited (elapsed $(( (now-start)/60 )) min). Inspect log tail."
        exit 0
    fi
    cur_mtime=$(stat -c %Y "$P12_LOG" 2>/dev/null || echo "$last_mtime")
    stale=$(( now - cur_mtime ))
    if [ "$stale" -gt "$MAX_STALL_SEC" ]; then
        echo "[relaunch] WATCHDOG: STALL — p12 log stale ${stale}s while process alive. Likely PhysX hang."
        exit 1
    fi
    if [ $(( now - start )) -gt "$MAX_TOTAL_SEC" ]; then
        echo "[relaunch] WATCHDOG: p12 total timeout reached; training should have completed by now."
        exit 2
    fi
    sleep 60
done
