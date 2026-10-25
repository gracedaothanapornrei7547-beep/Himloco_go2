#!/bin/bash
# p11 训练看门狗：进程死亡 / 日志停更（疑似 PhysX 卡死）即退出并通知。
# p11 = 10cm 台阶 + 均匀地形课程 + 速度课程自动 0→2.0，从零训练。
LOG=/home/rum/himloco_lab/logs/p11_stair10_uniform_train.log
PATTERN="train.py.*p11_stair10_uniform"
MAX_STALL_SEC=1800    # 30 分钟日志没动 = 卡死
MAX_TOTAL_SEC=115200  # 32h 兜底（实测 ~4.6-5.0s/iter → 20000 iter ≈ 25.6-27.8h + 余量）
start=$(date +%s)
last_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
while true; do
    now=$(date +%s)
    if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
        echo "WATCHDOG: training process exited (elapsed $(( (now-start)/60 )) min). Inspect log tail."
        exit 0
    fi
    cur_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$last_mtime")
    stale=$(( now - cur_mtime ))
    if [ "$stale" -gt "$MAX_STALL_SEC" ]; then
        echo "WATCHDOG: STALL — log mtime stale ${stale}s while process alive. Likely PhysX hang."
        exit 1
    fi
    if [ $(( now - start )) -gt "$MAX_TOTAL_SEC" ]; then
        echo "WATCHDOG: total timeout reached; training should have completed by now."
        exit 2
    fi
    sleep 60
done
