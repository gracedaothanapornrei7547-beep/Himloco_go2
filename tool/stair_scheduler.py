#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""楼梯分档调度器：5 → 7.5 → 10 cm，每档"训一段 → 评测 → 过则升档"。

设计（见 agent.md §9.7 / §9.13）：
  - **升档由独立多 seed 评测决定**，不由训练内的位移决定 —— 这是 sibling 项目验证过、
    也是唯一被证明能做到 10cm 的做法。
  - **固定台阶高度**，不用难度爬升课程（我们实测那会把均衡压在 ≤1.7cm）。
  - **两级评测**（省 GPU 时间）：
      · 快筛：up × 域随机化关 × 1 个 seed。没通过就不必跑其余 11 个组合。
      · 认证：up/down × 域随机化 开/关 × 3 seed，共 12 个组合，全过才升档。
  - **保持性**：升档后回头重测之前所有更低档，防止"为了 10cm 把 5cm 忘了"。
  - 全程串行（8GB 显存跑不了并行），每步前等 GPU 空出来。
  - 状态落在 out_dir/state.json，可中断续跑。

用法：
    nohup python3 scripts/himloco_rsl_rl/stair_scheduler.py \
        --start-checkpoint logs/himloco_rsl_rl/go2_rough/2026-09-11_19-13-24_p17_him_baseline/model_5999.pt \
        --out-dir logs/stair_curriculum/p18 \
        > logs/p18_scheduler.log 2>&1 &
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGES = [("h05", 0.05), ("h075", 0.075), ("h10", 0.10)]
FAST_SEED = 42
CERT_SEEDS = (42, 7, 2026)
CERT_DIRS = ("up", "down")
CERT_DR = ("0", "1")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0


def wait_gpu_free(threshold=600, timeout_s=900):
    """等 GPU 空出来（别的训练/评测退干净）。8GB 卡上并行会 OOM。"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        u = gpu_used_mib()
        if u < threshold:
            return True
        log(f"  GPU 仍占用 {u} MiB，等待…")
        time.sleep(20)
    log(f"  ⚠️ 等 GPU 超时（{gpu_used_mib()} MiB）")
    return False


def sh(cmd, env_extra=None, timeout_s=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    log("  $ " + " ".join(cmd[:6]) + (" …" if len(cmd) > 6 else ""))
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout_s)
    return r


def latest_checkpoint(run_name: str, since_ts: float = 0.0):
    """训练跑完后，找该 run_name 最新一次 run 目录里编号最大的 checkpoint。

    ★ 坑：`run_name` 会**跨重启复用**（每档每轮的 a01/a02… 是固定名字），而且调度器
      重启后会从同一档同一轮重来一遍 —— 磁盘上于是躺着好几个同名目录。如果本次训练
      因为报错/被 kill 而**没产出 checkpoint**，只按"最新的同名目录"去找，就会静默
      捡到**上一轮那个同名目录里的旧 checkpoint**，评测结果张冠李戴。
      所以只认 mtime 晚于本次尝试开始时刻的目录。
    """
    root = os.path.join(REPO, "logs", "himloco_rsl_rl", "go2_rough")
    cands = [d for d in os.listdir(root) if d.endswith("_" + run_name)] if os.path.isdir(root) else []
    cands = [d for d in cands if os.path.getmtime(os.path.join(root, d)) >= since_ts]
    if not cands:
        return None
    run_dir = os.path.join(root, sorted(cands)[-1])
    cks = [f for f in os.listdir(run_dir) if re.fullmatch(r"model_\d+\.pt", f)]
    if not cks:
        return None
    cks.sort(key=lambda f: int(f[6:-3]))
    return os.path.join(run_dir, cks[-1])


def train_attempt(ckpt, h, iters, run_name, timeout_s=4 * 3600):
    """从 ckpt 续训。

    ★ 坑：`--load_run` 不能用 `.*` —— train.py 走的是 IsaacLab 的 `get_checkpoint_path`，
      而 `.*` 会匹配到**本次 run 刚建的那个目录**（最新），于是它在里面找一个还不存在的
      checkpoint → `FileNotFoundError`。必须显式给 checkpoint 所在的 run 目录名，
      并且 `--checkpoint` 只传**文件名**（get_checkpoint_path 会拿它当 glob 拼在目录后面）。
    """
    wait_gpu_free()
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    ckpt_file = os.path.basename(ckpt)
    cmd = [
        "./run_himloco.sh", "scripts/himloco_rsl_rl/train.py",
        "--task", "Unitree-Go2-Stair-Stage", "--headless",
        "--num_envs", "1024", "--seed", "1",
        "--resume", "--load_run", run_dir, "--checkpoint", ckpt_file,
        "--max_iterations", str(iters),
        "--run_name", run_name,
    ]
    r = sh(cmd, env_extra={"HIMLOCO_STAIR_H": str(h)}, timeout_s=timeout_s)
    return r


def evaluate(ckpt, h, direction, keep_dr, seed, num_steps, climb_steps, timeout_s=1800):
    """跑一个评测组合，返回 dict（从 stdout 里抓 [Eval][JSON] 行）。"""
    wait_gpu_free()
    cmd = [
        "./run_himloco.sh", "scripts/himloco_rsl_rl/eval_stair_stage.py",
        "--task", "Unitree-Go2-Stair-Stage-Play",
        "--checkpoint", ckpt,
        "--direction", direction,
        "--seed", str(seed),
        "--num_envs", "16",
        "--num_steps", str(num_steps),
        "--cmd_vx", "0.5",
        "--climb_steps", str(climb_steps),
    ]
    r = sh(cmd, env_extra={"HIMLOCO_STAIR_H": str(h), "HIMLOCO_EVAL_KEEP_DR": keep_dr}, timeout_s=timeout_s)
    for line in (r.stdout or "").splitlines():
        if line.startswith("[Eval][JSON] "):
            return json.loads(line[len("[Eval][JSON] "):])
    return {"verdict": "ERROR", "stderr_tail": (r.stderr or "")[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters-per-attempt", type=int, default=300)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--num-steps", type=int, default=1400)
    ap.add_argument("--climb-steps", type=int, default=8)
    ap.add_argument("--cert-pass-frac", type=float, default=0.75,
                    help="认证时要求通过的组合比例（1.0 = 12/12 全过）")
    ap.add_argument("--only-stage", type=str, default=None, help="只跑某一档（调试用），如 h10")
    ap.add_argument("--run-prefix", type=str, default="p21", help="实验前缀名（默认 p21）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, "state.json")
    state = {"started": datetime.now().isoformat(), "checkpoint": os.path.abspath(args.start_checkpoint),
             "stages": {}, "history": []}
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        log(f"续跑：已有 state（checkpoint={state.get('checkpoint')}）")

    # ★ 兼容旧 state（两层）：本文件早期版本既不在 history 里记 ckpt，顶层 `checkpoint`
    #   又只在整个阶梯跑完才更新 —— 中途重启会退回最初起点、把已训的几轮全丢掉。
    #   恢复优先级（新→旧）：
    #     1. history 里最后一条带 ckpt 的记录 —— 最新，就是崩溃前那一轮训出来的
    #     2. 已通过档位里最高那一档的 checkpoint —— 旧 state 没记 ckpt 时的**真实断点**，
    #        从它续 = 只丢当前这一轮的训练，不丢整个阶梯
    #     3. 命令行 `--start-checkpoint`
    resumed_from = None
    for e in reversed(state.get("history", [])):
        if e.get("ckpt"):
            resumed_from = e["ckpt"]
            break
    if resumed_from is None:
        for tok, _ in reversed(STAGES):
            st = state.get("stages", {}).get(tok, {})
            if st.get("passed") and st.get("checkpoint"):
                resumed_from = st["checkpoint"]
                break
    if resumed_from and os.path.abspath(resumed_from) != os.path.abspath(state.get("checkpoint", "")):
        log(f"（自愈旧 state：顶层记的 {os.path.basename(state.get('checkpoint', '?'))} 已过期，"
            f"改用 {os.path.basename(resumed_from)}）")
        state["checkpoint"] = resumed_from

    ckpt = state.get("checkpoint", os.path.abspath(args.start_checkpoint))
    passed_stages = []

    for token, h in STAGES:
        if args.only_stage and token != args.only_stage:
            continue
        if state["stages"].get(token, {}).get("passed"):
            log(f"== 档 {token}（{h*100:.1f}cm）已通过，跳过 ==")
            passed_stages.append((token, h))
            continue

        log(f"\n{'='*78}\n== 档 {token}：台阶 {h*100:.1f} cm  起点 {os.path.basename(ckpt)}\n{'='*78}")
        attempt = 0
        best_ckpt = ckpt
        best_climbed_median = -1.0
        while attempt < args.max_attempts:
            attempt += 1
            run_name = f"{args.run_prefix}_{token}_a{attempt:02d}"
            log(f"-- 档 {token} 尝试 {attempt}/{args.max_attempts}：训练 {args.iters_per_attempt} iter (基于 {os.path.basename(ckpt)}) --")
            t0 = time.time()
            tr = train_attempt(ckpt, h, args.iters_per_attempt, run_name)
            new_ckpt = latest_checkpoint(run_name, since_ts=t0)
            if new_ckpt is None:
                log(f"   ⚠️ 训练没产出 checkpoint（returncode={tr.returncode}），重试")
                state["history"].append({"stage": token, "attempt": attempt, "train_rc": tr.returncode,
                                         "error": (tr.stderr or "")[-300:]})
                json.dump(state, open(state_path, "w"), indent=1, ensure_ascii=False)
                continue
            log(f"   新 checkpoint: {os.path.relpath(new_ckpt, REPO)}")

            # ---- 快筛（1 个组合）----
            fast = evaluate(new_ckpt, h, "up", "0", FAST_SEED, args.num_steps, args.climb_steps)
            cur_median = float(fast.get("climbed_median") or 0.0)
            log(f"   快筛 up/dr=off/seed{FAST_SEED}: {fast.get('verdict')} "
                f"({fast.get('passed_envs')}/{fast.get('num_envs')}, 中位 {cur_median:.2f})")
            state["history"].append({"stage": token, "attempt": attempt, "kind": "fast", "ckpt": new_ckpt, **fast})

            # ---- 防退化与历史最佳追踪（修复 §10.9 只前进不回退事故）----
            if cur_median > best_climbed_median:
                best_climbed_median = cur_median
                best_ckpt = new_ckpt
                ckpt = new_ckpt
                log(f"   ★ 刷新本档最佳快筛攀爬中位数: {best_climbed_median:.2f} (ckpt: {os.path.basename(best_ckpt)})")
            elif best_climbed_median > 0 and cur_median < best_climbed_median * 0.7:
                log(f"   ⚠️ 发生严重退化（当前中位 {cur_median:.2f} 级 < 历史最佳 {best_climbed_median:.2f} 级的 70%）")
                log(f"   ↺ 回退到本档历史最佳 checkpoint: {os.path.basename(best_ckpt)}")
                ckpt = best_ckpt
            else:
                ckpt = new_ckpt

            state["checkpoint"] = ckpt
            json.dump(state, open(state_path, "w"), indent=1, ensure_ascii=False)

            if fast.get("verdict") != "PASS":
                log("   快筛未过 → 继续下一轮尝试")
                continue

            # ---- 认证（12 个组合）----
            log(f"   快筛通过 → 跑完整认证（{len(CERT_DIRS)*len(CERT_DR)*len(CERT_SEEDS)} 个组合）")
            cert = []
            for d in CERT_DIRS:
                for dr in CERT_DR:
                    for s in CERT_SEEDS:
                        res = evaluate(ckpt, h, d, dr, s, args.num_steps, args.climb_steps)
                        cert.append(res)
                        log(f"     {d}/dr={dr}/seed{s}: {res.get('verdict')} "
                            f"({res.get('passed_envs')}/{res.get('num_envs')})")
                        state["history"].append({"stage": token, "attempt": attempt, "kind": "cert", **res})
                        json.dump(state, open(state_path, "w"), indent=1, ensure_ascii=False)
            n_pass = sum(1 for c in cert if c.get("verdict") == "PASS")
            frac = n_pass / max(len(cert), 1)
            log(f"   认证结果：{n_pass}/{len(cert)} = {frac:.0%}（门槛 {args.cert_pass_frac:.0%}）")

            # ---- 保持性：回头重测所有更低档 ----
            retain_ok = True
            for ptoken, ph in passed_stages:
                rr = evaluate(ckpt, ph, "up", "0", FAST_SEED, args.num_steps, args.climb_steps)
                log(f"   保持性 {ptoken}（{ph*100:.1f}cm）: {rr.get('verdict')} "
                    f"({rr.get('passed_envs')}/{rr.get('num_envs')})")
                state["history"].append({"stage": token, "attempt": attempt, "kind": "retention",
                                         "retain_stage": ptoken, **rr})
                json.dump(state, open(state_path, "w"), indent=1, ensure_ascii=False)
                if rr.get("verdict") != "PASS":
                    retain_ok = False

            if frac >= args.cert_pass_frac and retain_ok:
                log(f"   ✅ 档 {token} 通过（认证 {frac:.0%}，保持性 OK）")
                state["stages"][token] = {"passed": True, "height_m": h, "checkpoint": ckpt,
                                          "cert_frac": frac}
                passed_stages.append((token, h))
                json.dump(state, open(state_path, "w"), indent=1, ensure_ascii=False)
                break
            else:
                log(f"   ✗ 档 {token} 未过（认证 {frac:.0%}，保持性 {'OK' if retain_ok else '退化'}）"
                    f" → 继续训练（下一轮从当前 checkpoint 续）")
        else:
            log(f"   ⛔ 档 {token} 用尽 {args.max_attempts} 次尝试仍未通过，停在这里")
            state["stages"][token] = {"passed": False, "height_m": h, "checkpoint": ckpt}
            json.dump(state, open(state_path, "w"), indent=1, ensure_ascii=False)
            break

    state["finished"] = datetime.now().isoformat()
    state["checkpoint"] = ckpt
    json.dump(state, open(state_path, "w"), indent=1, ensure_ascii=False)
    log(f"\n调度器结束。最终 checkpoint = {os.path.relpath(ckpt, REPO)}")
    log(f"通过档位: {[t for t, _ in passed_stages] or '无'}")
    log(f"状态文件: {state_path}")


if __name__ == "__main__":
    main()
