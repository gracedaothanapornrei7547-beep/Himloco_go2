#!/bin/bash
# DreamWaQ CENet 训练看门狗: 进程死亡 / 日志停更(疑似 PhysX 卡死) / 总超时
LOG=/home/rum/dreamwaq_lab/logs/dreamwaq_cenet_fresh_train.log
PATTERN="train_dreamwaq.py.*Unitree-Go2-DreamWaQ-Rough"
MAX_STALL_SEC=1200    # 20 分钟日志没动 = 卡死
MAX_TOTAL_SEC=54000   # 15h 兜底 (DreamWaQ ~9h + 余量)
start=$(date +%s)
last_mtime=$start
while true; do
    now=$(date +%s)
    if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
        echo "WATCHDOG: DreamWaQ process exited (elapsed $(( (now-start)/60 )) min). Inspect log tail."
        exit 0
    fi
    cur_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$last_mtime")
    stale=$(( now - cur_mtime ))
    if [ "$stale" -gt "$MAX_STALL_SEC" ]; then
        echo "WATCHDOG: STALL — log mtime stale ${stale}s while process alive. Likely PhysX hang."
        exit 1
    fi
    if [ $(( now - start )) -gt "$MAX_TOTAL_SEC" ]; then
        echo "WATCHDOG: total timeout (15h) reached; DreamWaQ should have completed."
        exit 2
    fi
    last_mtime="$cur_mtime"
    sleep 60
done
