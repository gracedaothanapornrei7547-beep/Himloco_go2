#!/bin/bash
# p9 训练看门狗：进程死亡 / 日志停更（疑似 PhysX 卡死）即退出并通知。
LOG=/home/rum/himloco_lab/logs/p9_foot_clearance_onesided_train.log
PATTERN="train.py.*p9_foot_clearance_onesided"
MAX_STALL_SEC=1200   # 20 分钟日志没动 = 卡死
MAX_TOTAL_SEC=16200  # 4.5h 兜底（3.3h ETA + 余量）
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
        echo "WATCHDOG: STALL — log mtime stale ${stale}s while process alive. Likely PhysX hang (cf. p7 iter 41739)."
        exit 1
    fi
    if [ $(( now - start )) -gt "$MAX_TOTAL_SEC" ]; then
        echo "WATCHDOG: total timeout reached; training should have completed by now."
        exit 2
    fi
    sleep 60
done
