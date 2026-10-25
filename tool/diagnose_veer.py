# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""诊断左拐根因：策略到底有没有在执行航向修正，还是有机械/力矩不对称。

在无限平地上固定 2.0 m/s 直线（heading_command 部署口径），连续走满距离，逐帧对比：

  - wz_cmd  = 航向控制器输出的偏航速率指令（heading 误差反馈重算）
  - wz_act  = 机器人实际偏航速率（base 系 z 轴）
  - heading = 实际 yaw（base 朝向，world 系）
  - 左右脚净接触力（竖直 + 水平）→ 判断是否有左右加载不对称
  - 由水平接触力 × 脚力臂 合成的绕 z 净力矩 τ_z → 判断是否有一个「持续左转力矩」在推

关键判读：
  - 若 wz_act 与 wz_cmd 强相关（斜率≈1）→ 策略在执行修正，拐是别的问题；
  - 若 wz_act 与 wz_cmd 几乎无关、且 wz_act 恒正 → 策略根本没执行，根因在 reward 或控制裕度；
  - 若 τ_z 持续为负/正 → 机械/接触不对称在产生一个方向性的偏航力矩。

用法（在 himloco_lab 根目录）：
    ./run_himloco.sh scripts/himloco_rsl_rl/diagnose_veer.py \
        --task Unitree-Go2-Velocity-Play \
        --checkpoint <path/to/model_xxx.pt> \
        --cmd_vx 2.0 --distance 50 --num_envs 64 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Diagnose the root cause of the left veer.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--env_spacing", type=float, default=5.0, help="Spawn grid spacing (m).")
parser.add_argument("--distance", type=float, default=50.0, help="Target straight-line distance (m).")
parser.add_argument("--cmd_vx", type=float, default=2.0, help="Pinned forward command (m/s).")
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing

    # 平地无障碍 + 部署口径 heading 指令（同 measure_heading_fall.py）
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.curriculum.terrain_levels = None
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.heading_command = True
    env_cfg.commands.base_velocity.rel_heading_envs = 1.0
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    # ⚠️★ 必须显式恢复 ang_vel_z 到训练值 (-2,2)：RobotPlayEnvCfg.__post_init__ 把它钉成
    #   (-0,0)=(0,0)，而 heading 控制器用它作 clamp 范围 → 修正量被 clamp 死成 0。
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
    env_cfg.terminations.time_out = None

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene["contact_forces"]
    dt = env.unwrapped.step_dt

    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length,
    )
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    # ---- 解析左右脚 ----
    foot_names = list(contact.body_names)
    print(f"[INFO] contact sensor body names: {foot_names}")
    left_idx = [i for i, n in enumerate(foot_names) if n.upper().startswith(("FL", "RL"))]
    right_idx = [i for i, n in enumerate(foot_names) if n.upper().startswith(("FR", "RR"))]
    print(f"[INFO] left feet idx={left_idx}  right feet idx={right_idx}")

    # 映射 foot 名 -> robot body 索引（用于取脚位置算力矩）
    robot_body_names = list(robot.body_names)
    foot_robot_idx = [robot_body_names.index(n) for n in foot_names]

    N = env.num_envs
    device = env.device
    max_steps = int(args_cli.distance / args_cli.cmd_vx / dt) + 250

    # ---- 累加器（torch 向量化，跑完再汇总）----
    cnt = 0
    sum_wz_cmd = 0.0
    sum_wz_act = 0.0
    sum_wz_cmd_sq = 0.0
    sum_wz_act_sq = 0.0
    sum_wz_cmd_act = 0.0
    sum_heading = 0.0
    sum_abs_wz_act = 0.0
    max_abs_wz_act = 0.0
    sum_left_vert = 0.0
    sum_right_vert = 0.0
    sum_left_horiz = 0.0
    sum_right_horiz = 0.0
    sum_tau_z = 0.0
    # 分段时间轴（前/中/后 1/3）看漂移是恒定还是加速
    seg_wz_cmd = [0.0, 0.0, 0.0]
    seg_wz_act = [0.0, 0.0, 0.0]
    seg_cnt = [0, 0, 0]

    obs = env.get_observations()

    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _, _, _ = env.step(actions)

        # 当前拍（reward 同口径）：指令 vs 实际
        wz_cmd = env.unwrapped.command_manager.get_command("base_velocity")[:, 2].clone()
        wz_act = robot.data.root_ang_vel_b[:, 2].clone()
        heading = robot.data.heading_w.clone()

        cnt += N
        sum_wz_cmd += wz_cmd.sum().item()
        sum_wz_act += wz_act.sum().item()
        sum_wz_cmd_sq += (wz_cmd**2).sum().item()
        sum_wz_act_sq += (wz_act**2).sum().item()
        sum_wz_cmd_act += (wz_cmd * wz_act).sum().item()
        sum_heading += heading.sum().item()
        sum_abs_wz_act += wz_act.abs().sum().item()
        max_abs_wz_act = max(max_abs_wz_act, wz_act.abs().max().item())

        # 左右脚净接触力
        nf = contact.data.net_forces_w  # (N, num_feet, 3)
        left_vert = nf[:, left_idx, 2].sum(dim=1) if left_idx else torch.zeros(N, device=device)
        right_vert = nf[:, right_idx, 2].sum(dim=1) if right_idx else torch.zeros(N, device=device)
        left_horiz = nf[:, left_idx, 0:2].norm(dim=-1).sum(dim=1) if left_idx else torch.zeros(N, device=device)
        right_horiz = nf[:, right_idx, 0:2].norm(dim=-1).sum(dim=1) if right_idx else torch.zeros(N, device=device)
        sum_left_vert += left_vert.sum().item()
        sum_right_vert += right_vert.sum().item()
        sum_left_horiz += left_horiz.sum().item()
        sum_right_horiz += right_horiz.sum().item()

        # 绕 z 净力矩 τ_z = Σ (r × F)_z（水平力 × 相对 base 的力臂）
        foot_pos = robot.data.body_pos_w[:, foot_robot_idx, :]  # (N, num_feet, 3)
        r = foot_pos - robot.data.root_pos_w[:, None, :]  # (N, num_feet, 3)
        f_horiz = nf[:, :, 0:2]  # (N, num_feet, 2)
        tau_z = (r[:, :, 0] * f_horiz[:, :, 1] - r[:, :, 1] * f_horiz[:, :, 0]).sum(dim=1)
        sum_tau_z += tau_z.sum().item()

        # 分段（前/中/后 1/3）
        seg = 0 if step < max_steps // 3 else (1 if step < 2 * max_steps // 3 else 2)
        seg_wz_cmd[seg] += wz_cmd.sum().item()
        seg_wz_act[seg] += wz_act.sum().item()
        seg_cnt[seg] += N

    # ---- 汇总 ----
    n = cnt
    mean_wz_cmd = sum_wz_cmd / n
    mean_wz_act = sum_wz_act / n
    var_wz_cmd = sum_wz_cmd_sq / n - mean_wz_cmd**2
    var_wz_act = sum_wz_act_sq / n - mean_wz_act**2
    cov = sum_wz_cmd_act / n - mean_wz_cmd * mean_wz_act
    corr = cov / (torch.sqrt(torch.tensor(max(var_wz_cmd * var_wz_act, 1e-12)))).item()
    slope = cov / var_wz_cmd if var_wz_cmd > 1e-12 else float("nan")

    print("\n================ 左拐根因诊断 ================")
    print(f"checkpoint={resume_path}")
    print(f"num_envs={N}  distance={args_cli.distance}m  cmd_vx={args_cli.cmd_vx}m/s  dt={dt:.4f}s  steps={max_steps}")
    print(f"\n--- 偏航指令 vs 实际（reward 同口径，全走全程平均）---")
    print(f"wz_cmd   mean={mean_wz_cmd:+.4f} rad/s   std={torch.sqrt(torch.tensor(max(var_wz_cmd,0))):.4f}")
    print(f"wz_act   mean={mean_wz_act:+.4f} rad/s   std={torch.sqrt(torch.tensor(max(var_wz_act,0))):.4f}   "
          f"|max|={max_abs_wz_act:.4f}")
    print(f"相关系数 corr(wz_cmd, wz_act) = {corr:+.4f}    回归斜率 slope = {slope:+.4f}")
    print(f"航向 heading   mean={sum_heading/n:+.4f} rad ({sum_heading/n*180/3.14159:+.1f}°)")
    print(f"\n--- 分段（前/中/后 1/3 平均 wz，rad/s）看漂移是恒定还是加速 ---")
    for i, label in enumerate(["前 1/3", "中 1/3", "后 1/3"]):
        c = seg_cnt[i] or 1
        print(f"  {label}: wz_cmd={seg_wz_cmd[i]/c:+.4f}   wz_act={seg_wz_act[i]/c:+.4f}")
    print(f"\n--- 左右脚净接触力（全程时间平均，N）---")
    print(f"竖直力 左={sum_left_vert/n:+.3f}  右={sum_right_vert/n:+.3f}  "
          f"差(左-右)={ (sum_left_vert-sum_right_vert)/n:+.3f}")
    print(f"水平力 左={sum_left_horiz/n:+.3f}  右={sum_right_horiz/n:+.3f}  "
          f"差(左-右)={ (sum_left_horiz-sum_right_horiz)/n:+.3f}")
    print(f"绕z净力矩 τ_z（水平接触力×力臂，+为左转）mean={sum_tau_z/n:+.4f} N·m")

    print("\n--- 判读 ---")
    if abs(corr) > 0.7:
        print("wz_act 与 wz_cmd 强相关 → 策略在执行航向修正，拐是别的原因（指令本身偏/力矩未抵消）。")
    elif abs(corr) < 0.3:
        print("wz_act 与 wz_cmd 几乎无关 → 策略没在执行航向修正（reward 太平 或 控制裕度不够）。")
    else:
        print("wz_act 与 wz_cmd 中等相关 → 策略部分执行修正，但被某个持续扰动抵消。")
    if (sum_left_horiz - sum_right_horiz) / n > 1.0 or (sum_left_horiz - sum_right_horiz) / n < -1.0:
        print("左右脚水平力明显不对称 → 存在机械/步态加载不对称。")
    if abs(sum_tau_z / n) > 0.5:
        print(f"τ_z 持续非零（{sum_tau_z/n:+.3f} N·m）→ 存在方向性偏航力矩在推着转。")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
