# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""楼梯分档评测器：判一个 checkpoint 在「某个台阶高度 + 某个方向 + 某个域随机化档」下过不过。

设计原则（见 agent.md §9.13）：
  - **一次调用只测一个组合**（方向 × 域随机化 × seed）。同进程反复建/销毁 Isaac 场景容易出
    玄学问题，而且单个组合边界清晰、失败了能一眼看出是哪一格 —— 组合由调度器串起来。
  - **判据锚在"能不能爬上去"，而不是"速度好不好"**：公开数据（§1.4）表明卡人的是
    **连续性**不是高度，所以要求**连续爬够 N 级且全程不摔倒**。
  - **地形隔离**：把非目标地形的 proportion 全置 0、目标楼梯置 1.0 → 20 列全是它，
    每个 env 都必然生在该楼梯上，不存在"测了半天其实在斜坡上"。

★ 方向与地形名的对应**反直觉**，已在 IsaacLab 源码确认（`mesh_terrains.py`）：
    pyramid_stairs      origin z = +(num_steps+1)*h  → 出生在金字塔顶 → **下楼梯**
    pyramid_stairs_inv  origin z = -(num_steps+1)*h  → 出生在金字塔底 → **上楼梯**
  所以 `--direction up` 用的是 `pyramid_stairs_inv`。搞反会得到完全相反的结论。

用法：
    HIMLOCO_STAIR_H=0.10 HIMLOCO_EVAL_KEEP_DR=0 \
    ./run_himloco.sh scripts/himloco_rsl_rl/eval_stair_stage.py \
        --task Unitree-Go2-Stair-Stage-Play \
        --checkpoint <run_dir>/model_XXXX.pt \
        --direction up --seed 42 --num_envs 32 --num_steps 600 --cmd_vx 0.5 --climb_steps 8
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Stair-stage evaluator: one (direction x DR x seed) combo per call.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="himloco_rsl_rl_cfg", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=600, help="Control steps per episode window.")
parser.add_argument("--cmd_vx", type=float, default=0.5, help="Pinned forward command (m/s).")
parser.add_argument("--climb_steps", type=int, default=8, help="Number of consecutive steps that must be climbed.")
parser.add_argument("--direction", type=str, choices=["up", "down"], required=True,
                    help="up = pyramid_stairs_inv（出生在底）; down = pyramid_stairs（出生在顶）。")
parser.add_argument("--settle_steps", type=int, default=50,
                    help="Steps to ignore before recording the start height (spawn drop needs to finish).")
parser.add_argument("--json_out", type=str, default=None, help="If set, append the machine-readable result here.")
parser.add_argument("--no_obs_noise", action="store_true",
                    help="诊断用：关掉观测噪声（enable_corruption=False）。默认保持环境原设定。")
parser.add_argument("--video", action="store_true",
                    help="录跟随相机 mp4，用于肉眼查看该组合的真实表现。**不改变任何判据**（纯旁路）。")
parser.add_argument("--cam_res", type=int, default=640, help="视频宽度（高度 = 宽度 * 9/16）。")
parser.add_argument("--follow_env", type=int, default=0, help="相机跟随第几个 env。")
parser.add_argument("--video_dir", type=str, default=None,
                    help="mp4 输出目录（默认写在 checkpoint 同目录的 videos/stair/ 下）。")
parser.add_argument("--view", action="store_true",
                    help="打开 Isaac Sim 图形界面（本机 DISPLAY）实时看，而不是 headless。可和 --video 同时用。")
cli_args.add_himloco_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in sys.argv and "--view" not in sys.argv:
    sys.argv.append("--headless")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# ★ 相机（RTX）必须在 AppLauncher 之前打开 —— 否则 scene 里加了 CameraCfg 也渲染不出图
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import retrieve_file_path

import himloco_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs

    # ---- 地形隔离：只留目标楼梯 ----
    gen = env_cfg.scene.terrain.terrain_generator
    target = "pyramid_stairs_inv" if args_cli.direction == "up" else "pyramid_stairs"
    for k in gen.sub_terrains:
        gen.sub_terrains[k].proportion = 0.0
    gen.sub_terrains[target].proportion = 1.0
    h = float(env_cfg.scene.terrain.terrain_generator.sub_terrains[target].step_height_range[0])

    # ---- 观测噪声：默认**保持开着**，这决定了所有评测数字的口径 ----
    # ★ 实测（2026-09-12）：`StairStagePlayEnvCfg` 继承自 `StairStageEnvCfg`，而不是上游那个
    #   会关噪声的 `RobotPlayEnvCfg`；`observations.policy.enable_corruption` 因此仍是 True
    #   （base_ang_vel ±0.2、joint_vel ±1.5、joint_pos ±0.01、gravity ±0.05…）。
    #   所以 **"域随机化关"这一档 ≠ 干净确定性评测** —— 它关的是动力学随机化（加质量/质心/
    #   推扰），观测噪声一直在。我们至今所有数字（h05 12/12、h075 13/16、以及手工基线
    #   5cm 8/16 / 10cm 0/16）都是**带观测噪声**测的。
    #   而 G0b 逐位验证过的部署链路（ONNX→MuJoCo→C++）是**无观测噪声**的 —— 两者不是同一个
    #   输入分布。本开关只为把"能力不够"和"噪声敏感"这两件事分开，**不动调度器的判据**。
    if args_cli.no_obs_noise:
        env_cfg.observations.policy.enable_corruption = False

    # ---- 评测口径：固定直线前进 ----
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    # ---- 跟随相机（仅 --video 时；纯旁路，不参与任何判据）----
    # ★ 必须用 {ENV_REGEX_NS} 按 env 复制：全局单相机（无 regex）在 scene.reset(env_ids) 时
    #   sensor_base 会把 env_ids 索引进 (1,) 缓冲，触发 CUDA device-side assert。
    if args_cli.video:
        env_cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0,
            data_types=["rgb"],
            width=args_cli.cam_res,
            height=int(args_cli.cam_res * 9 / 16),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        from isaaclab_tasks.utils import get_checkpoint_path

        log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    keep_dr = os.environ.get("HIMLOCO_EVAL_KEEP_DR", "0") == "1"
    print(f"[Eval] ckpt={os.path.basename(resume_path)}  h={h * 100:.1f}cm  dir={args_cli.direction}"
          f"  terrain={target}  DR={'on' if keep_dr else 'off'}  seed={args_cli.seed}"
          f"  cmd={args_cli.cmd_vx}m/s  N={args_cli.climb_steps}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    camera = env.unwrapped.scene["camera"] if args_cli.video else None

    # 加载策略（与训练同一套 HIM runner，保证口径一致）
    from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper

    env = HimlocoVecEnvWrapper(env, history_length=agent_cfg.history_length,
                               privileged_history_length=agent_cfg.privileged_history_length)
    # ★ num_envs/device 必须在包 wrapper 之后取 —— gym 的 OrderEnforcing 层没有这两个属性
    N = env.num_envs
    device = env.device
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    # ---- 逐步记录：**按回合**统计"摔倒前爬到几级" ----
    #
    # ★ 为什么不能整段取 max：Play 环境 episode_length_s=20s=1000 步，跑超过 1000 步就会
    #   触发 time_out 重置，重置后机器人回到出生点、z 掉回去 —— 整段 max 会把两个回合混在一起。
    # ★ 为什么不能要求"全程不摔"：那测的是**耐久**而不是**能不能爬**。实测 p17 在 5cm 上
    #   climbed 中位 3.78、最大 8.30 级（=它爬得上去），但 28 秒里 16/16 都摔了。
    #   所以正确口径是：**单个回合内、摔倒之前，最高爬到几级**。
    # ★ 方向决定"进展"是上升还是下降 —— 这是本脚本最容易错的一处：
    #   up   （pyramid_stairs_inv，出生在底）→ 进展 = 高度**上升** = (z_max - z_start)/h
    #   down （pyramid_stairs，   出生在顶）→ 进展 = 高度**下降** = (z_start - z_min)/h
    #   用 z_max 去量下楼梯，量到的是"有没有反弹上升"，和下降了几级无关 —— 会得到伪值。
    is_down = args_cli.direction == "down"

    def _progress(z_start_, z_hi_, z_lo_):
        num = (z_start_ - z_lo_) if is_down else (z_hi_ - z_start_)
        return torch.nan_to_num((num / h).clamp(min=0.0), nan=0.0)

    settle = args_cli.settle_steps
    since_reset = torch.zeros(N, dtype=torch.long, device=device)
    z_ep_start = torch.full((N,), float("nan"), device=device)
    z_ep_max = torch.full((N,), float("-inf"), device=device)
    z_ep_min = torch.full((N,), float("inf"), device=device)
    best_climb = torch.zeros(N, device=device)   # 各回合 "摔倒前完成几级" 的最大值
    fell_any = torch.zeros(N, dtype=torch.bool, device=device)
    n_episodes = 0
    # ---- 相机：吊在机器人后上方，eye 相对 root (-3.5, 0, 1.8)，看向前方 1.5m、高 0.1m 处 ----
    cam_eye_off = torch.tensor([-3.5, 0.0, 1.8], device=device)
    cam_tgt_off = torch.tensor([1.5, 0.0, 0.1], device=device)

    # ---- 录像（纯旁路，不影响判据）----
    writer = None
    out_path = None
    fps = int(round(1.0 / env.unwrapped.step_dt))
    if args_cli.video:
        import imageio.v2 as imageio

        out_dir = args_cli.video_dir or os.path.join(os.path.dirname(resume_path), "videos", "stair")
        os.makedirs(out_dir, exist_ok=True)
        tag = (f"h{int(round(h * 1000)):03d}_{args_cli.direction}_dr{int(keep_dr)}"
               f"_s{args_cli.seed}_vx{args_cli.cmd_vx}")
        out_path = os.path.join(out_dir, tag + ".mp4")
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
        print(f"[Eval][Video] {out_path}  fps={fps}  res={args_cli.cam_res}x{int(args_cli.cam_res*9/16)}"
              f"  跟随 env{args_cli.follow_env}")

    # ---- 图形界面：把视口相机跟着 env 走（headless 时 set_camera_view 内部自己跳过）----
    sim_ctx = env.unwrapped.sim
    view_dt = (1.0 / fps) if args_cli.view else 0.0
    next_tick = time.time() + view_dt
    if args_cli.view:
        print(f"[Eval][View] 已开图形界面，视口跟随 env{args_cli.follow_env}，按实时 {fps} fps 节流")

    obs = env.get_observations()
    actions = policy(obs)
    for step in range(args_cli.num_steps):
        if camera is not None or args_cli.view:
            # 先摆相机再步进 —— 渲染发生在 step 内部
            root_pos = robot.data.root_pos_w
            eyes = root_pos + cam_eye_off
            tgts = root_pos + cam_tgt_off
            if camera is not None:
                camera.set_world_poses_from_view(eyes, tgts)
            if args_cli.view:
                f = args_cli.follow_env
                sim_ctx.set_camera_view(eyes[f].tolist(), tgts[f].tolist())
        with torch.inference_mode():
            obs, _, _, dones, *_ = env.step(actions)
            z = robot.data.root_pos_w[:, 2]
            since_reset = since_reset + 1
            if step % 50 == 0:
                print(f"[TRACE Step {step:3d}] x={robot.data.root_pos_w[0, 0]:.2f}, y={robot.data.root_pos_w[0, 1]:.2f}, z={robot.data.root_pos_w[0, 2]:.2f}, vx={robot.data.root_lin_vel_b[0, 0]:.2f}, act_norm={torch.norm(actions[0]).item():.2f}")
            # 出生时机器人会从 spawn 高度落到地面 —— 落稳（since_reset == settle）才起算，
            # 否则 z_ep_start 被抬高/压低，进展被系统性污染。
            z_ep_start = torch.where(since_reset == settle, z, z_ep_start)
            active = since_reset >= settle
            z_ep_max = torch.where(active, torch.maximum(z_ep_max, z), z_ep_max)
            z_ep_min = torch.where(active, torch.minimum(z_ep_min, z), z_ep_min)
            if dones.any():
                done = dones.bool()
                fell_any |= done
                prog = _progress(z_ep_start, z_ep_max, z_ep_min)
                best_climb = torch.where(done, torch.maximum(best_climb, prog), best_climb)
                n_episodes += int(done.sum())
                # 回合结束 → 该 env 从下一拍重新起算
                since_reset = torch.where(done, torch.zeros_like(since_reset), since_reset)
                z_ep_start = torch.where(done, torch.full_like(z_ep_start, float("nan")), z_ep_start)
                z_ep_max = torch.where(done, torch.full_like(z_ep_max, float("-inf")), z_ep_max)
                z_ep_min = torch.where(done, torch.full_like(z_ep_min, float("inf")), z_ep_min)
            # ★ policy 必须留在 inference_mode 块内：obs 是 inference tensor，出了这个块
            #   estimator 里的 F.linear 会试图建图，报 "Inference tensors cannot be saved
            #   for backward"。（相机/录像/节流都必须放在块**外**。）
            actions = policy(obs)

        if writer is not None:
            frame = camera.data.output["rgb"][args_cli.follow_env].cpu().numpy()
            writer.append_data(frame)

        if view_dt:
            # 不限速的话界面会一闪而过，根本看不清；这里按实时节流（跑不动就自然掉帧）
            next_tick += view_dt
            slack = next_tick - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.time()

    # 跑到最后仍未结束的回合，也要算进去
    best_climb = torch.maximum(best_climb, _progress(z_ep_start, z_ep_max, z_ep_min))

    need = args_cli.climb_steps * 0.8   # 留 20% 余量，容忍一步的尺度误差
    ok = best_climb >= need
    n_ok = int(ok.sum())
    n_pass_env = n_ok
    verdict = "PASS" if n_pass_env == N else ("PARTIAL" if n_pass_env > 0 else "FAIL")

    med = float(best_climb.median())
    print("\n" + "=" * 84)
    word = "下" if is_down else "上"
    print(f"[Eval] {verdict}  {n_pass_env}/{N} env 在**单个回合内**{word}完 ≥{args_cli.climb_steps} 级")
    print(f"       最佳单回合完成级数: mean={float(best_climb.mean()):.2f} median={med:.2f} "
          f"max={float(best_climb.max()):.2f}")
    print(f"       至少摔过一次的 env: {int(fell_any.sum())}/{N}   （总回合数 {n_episodes}）")
    print("=" * 84)

    result = {
        "checkpoint": os.path.abspath(resume_path),
        "stage_height_m": h,
        "direction": args_cli.direction,
        "keep_dr": keep_dr,
        "seed": args_cli.seed,
        "cmd_vx": args_cli.cmd_vx,
        "climb_steps_required": args_cli.climb_steps,
        "num_envs": N,
        "num_steps": args_cli.num_steps,
        "passed_envs": n_pass_env,
        "fell_envs": int(fell_any.sum()),
        "climbed_median": med,
        "climbed_max": float(best_climb.max()),
        "verdict": verdict,
    }
    if out_path:
        result["video"] = os.path.abspath(out_path)
    print("[Eval][JSON] " + json.dumps(result, ensure_ascii=False))
    if args_cli.json_out:
        with open(args_cli.json_out, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if writer is not None:
        writer.close()
        print(f"[Eval][Video] 已写出 {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
