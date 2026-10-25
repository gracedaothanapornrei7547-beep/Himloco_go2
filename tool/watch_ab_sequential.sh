#!/bin/bash
# 顺序 A/B 训练看门狗: HIMLoco(SwAV) -> DreamWaQ(CENet)
# 检测: 进程死亡 / 日志停更(疑似 PhysX 卡死) / 总超时。两阶段日志分开跟踪。
# 用法: ./scripts/watch_ab_sequential.sh

HIMLOCO_PAT="train.py.*Unitree-Go2-Velocity"
DREAMWAQ_PAT="train_dreamwaq.py.*Unitree-Go2-DreamWaQ-Rough"
PARENT_PAT="run_himloco.sh scripts/himloco_rsl_rl/train.py"   # 链式父 shell

HIMLOCO_LOG=/home/rum/himloco_lab/logs/himloco_swav_fresh_train.log
DREAMWAQ_LOG=/home/rum/dreamwaq_lab/logs/dreamwaq_cenet_fresh_train.log

MAX_STALL_SEC=1200    # 20 分钟日志没动 = 卡死
MAX_TOTAL_SEC=172800  # 48h 兜底 (HIMLoco ~19h + DreamWaQ ~? + 余量)

start=$(date +%s)
prev_phase="none"
prev_mtime=0

phase_of() {
    if pgrep -f "$HIMLOCO_PAT" >/dev/null 2>&1; then echo "himloco"
    elif pgrep -f "$DREAMWAQ_PAT" >/dev/null 2>&1; then echo "dreamwaq"
    else echo "none"; fi
}

log_of() { case "$1" in himloco) echo "$HIMLOCO_LOG";; dreamwaq) echo "$DREAMWAQ_LOG";; *) echo "";; esac; }

while true; do
    now=$(date +%s)
    phase=$(phase_of)

    if [ "$phase" = "none" ]; then
        if ! pgrep -f "$PARENT_PAT" >/dev/null 2>&1; then
            echo "WATCHDOG: all training processes exited (elapsed $(( (now-start)/60 )) min). Inspect both logs."
            exit 0
        fi
        # 阶段切换间隙(父 shell 存活但 python 未起): 容忍, 不判卡死
        prev_phase="none"; prev_mtime=0
        sleep 60; continue
    fi

    LOG=$(log_of "$phase")

    # 阶段切换: 重置 mtime 基线
    if [ "$phase" != "$prev_phase" ]; then
        echo "WATCHDOG: phase -> $phase (elapsed $(( (now-start)/60 )) min)"
        prev_phase="$phase"
        prev_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    fi

    cur_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$prev_mtime")
    stale=$(( now - cur_mtime ))
    if [ "$stale" -gt "$MAX_STALL_SEC" ]; then
        echo "WATCHDOG: STALL in $phase — log mtime stale ${stale}s while process alive. Likely PhysX hang."
        exit 1
    fi
    if [ $(( now - start )) -gt "$MAX_TOTAL_SEC" ]; then
        echo "WATCHDOG: total timeout (48h) reached in phase $phase; should have completed by now."
        exit 2
    fi
    prev_mtime="$cur_mtime"
    sleep 60
done
