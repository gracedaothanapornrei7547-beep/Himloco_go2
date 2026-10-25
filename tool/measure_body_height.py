# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""部署验证：上/下楼梯时机身离地高度（base_height）+ 关节偏离默认位形。

在默认混合地形（30% 上楼梯 + 30% 下楼梯 + 平地/斜坡/障碍）上，命令固定 vx 直线，
连续跑固定步数，每步测量：

  - base_height = 机身中心 Z − base_height_scanner 打到的地形 Z（机身相对脚下地形的净空）
    —— 上楼梯时若「机身太低/下蹲」，这个值会明显低于标称站高（Go2 约 0.30m）。
        min 值即「下蹲最深」时的离地净空，是验证 joint_pos_penalty 的核心指标。
  - joint_dev = ‖joint_pos − default_joint_pos‖₂（12 关节对默认位形的 L2 偏离）
    —— joint_pos_penalty 的直接作用量：权重 -1.0 会把这个值压小（姿势更接近默认站立）。

对比口径（A/B）：
    基线 p5c  model_39989.pt（无 joint_pos_penalty）
    处理 p6   model_42988.pt（resume p5c + joint_pos_penalty=-1.0，其余全同）
    同一地形、同一 --step_height、同一 --cmd_vx、同一 heading 口径跑两次，
    只差 joint_pos_penalty，看 base_height 的 min/p5 是否抬升。

用法（在 himloco_lab 根目录）：
    ./run_himloco.sh scripts/himloco_rsl_rl/measure_body_height.py \
        --task Unitree-Go2-Velocity-Play \
        --checkpoint <path/to/model_xxx.pt> \
        --num_envs 64 --step_height 0.05 --cmd_vx 2.0 \
        --heading_mode --heading_control_stiffness 2.0 --num_steps 2000
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Measure body height + joint deviation over stairs.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of control steps to run.")
parser.add_argument("--cmd_vx", type=float, default=2.0, help="Pinned forward command (m/s).")
parser.add_argument(
    "--step_height",
    type=float,
    default=0.05,
    help="Override staircase step height (m), fixed for all difficulties.",
)
parser.add_argument(
    "--heading_mode",
    action="store_true",
    default=False,
    help="Deployment-faithful heading command (in-distribution for the policy).",
)
parser.add_argument(
    "--heading_control_stiffness",
    type=float,
    default=2.0,
    help="Heading P-gain. Match the value the checkpoint was trained with.",
)
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
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
    """Measure body height above local terrain + joint deviation on stairs."""
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs

    # 固定直线指令（body-frame vy=0；vx 钉死 cmd_vx），并钉死 80% anchor 组采样范围。
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    if args_cli.heading_mode:
        # 部署口径：heading_command=True + rel_heading_envs=1.0 + ang_vel_z=(-2,2)
        env_cfg.commands.base_velocity.heading_command = True
        env_cfg.commands.base_velocity.rel_heading_envs = 1.0
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
    else:
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    if args_cli.heading_control_stiffness is not None:
        env_cfg.commands.base_velocity.heading_control_stiffness = args_cli.heading_control_stiffness
    # 固定楼梯台阶高度（上/下楼梯都改），保证 A/B 两次跑的是同一难度楼梯。
    for key in ("pyramid_stairs", "pyramid_stairs_inv"):
        st = env_cfg.scene.terrain.terrain_generator.sub_terrains.get(key)
        if st is not None:
            st.step_height_range = (args_cli.step_height, args_cli.step_height)

    # resolve checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    scanner = env.unwrapped.scene["base_height_scanner"]
    dt = env.unwrapped.step_dt

    # wrap for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length,
    )

    # create runner and load checkpoint
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    N = env.num_envs
    device = env.device
    print(f"[INFO] control dt={dt:.4f}s  num_envs={N}  num_steps={args_cli.num_steps}  "
          f"cmd_vx={args_cli.cmd_vx}m/s  step_height={args_cli.step_height}m")

    # --- 累积统计 ---
    # base_height 全样本（每步每 env 一个值，非法全 NaN 的 env 该步跳过）
    bh_all = []
    # 每个 env 的 min base_height（下蹲最深）
    bh_min = torch.full((N,), float("inf"), device=device)
    # 每个 env 的 mean base_height
    bh_sum = torch.zeros(N, device=device)
    bh_cnt = torch.zeros(N, device=device)
    # 关节偏离（joint_pos_penalty 作用量）
    jd_sum = torch.zeros(N, device=device)

    def _terrain_z():
        ray_z = scanner.data.ray_hits_w[..., 2]  # (N, num_rays)
        valid = ~torch.isnan(ray_z) & ~torch.isinf(ray_z) & (ray_z.abs() < 1e6)
        masked = torch.where(valid, ray_z, torch.full_like(ray_z, float("nan")))
        return torch.nanmean(masked, dim=1)  # (N,)，全无效的 env 为 NaN

    obs = env.get_observations()
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _, _, _ = env.step(actions)

        # 机身中心 Z − 脚下地形 Z
        base_z = robot.data.root_pos_w[:, 2]
        terrain_z = _terrain_z()
        bh = base_z - terrain_z  # (N,)
        jd = torch.linalg.norm(robot.data.joint_pos - robot.data.default_joint_pos, dim=1)  # (N,)

        valid_bh = ~torch.isnan(bh)
        bh_all.append(bh[valid_bh].cpu())
        bh_min = torch.where(valid_bh & (bh < bh_min), bh, bh_min)
        bh_sum = torch.where(valid_bh, bh_sum + bh, bh_sum)
        bh_cnt = bh_cnt + valid_bh.float()
        jd_sum += jd

    # --- 汇总 ---
    bh_all = torch.cat(bh_all) if bh_all else torch.zeros(0)
    bh_mean_per_env = bh_sum / bh_cnt.clamp(min=1.0)
    jd_mean_per_env = jd_sum / args_cli.num_steps

    def _stats(t: torch.Tensor, name: str, unit: str = ""):
        valid = t[~torch.isnan(t) & torch.isfinite(t)]
        if valid.numel() == 0:
            print(f"[{name}] no valid samples")
            return
        q05 = torch.quantile(valid, 0.05).item()
        print(
            f"[{name}] n={valid.numel()}  mean={valid.mean().item():.4f}{unit}  "
            f"median={valid.median().item():.4f}  p5={q05:.4f}  min={valid.min().item():.4f}  "
            f"max={valid.max().item():.4f}"
        )

    print("\n================ 机身离地高度验证结果 ================")
    print(f"num_envs={N}  num_steps={args_cli.num_steps}  cmd_vx={args_cli.cmd_vx}m/s  "
          f"step_height={args_cli.step_height}m  dt={dt:.4f}s")
    print("\n--- base_height = 机身中心 Z − 脚下地形 Z（离地净空，标称站高约 0.30m）---")
    _stats(bh_all, "base_height 全样本", " m")
    _stats(bh_min, "base_height 每 env 最小值（下蹲最深）", " m")
    _stats(bh_mean_per_env, "base_height 每 env 均值", " m")
    print("\n--- joint_dev = ‖joint_pos − default‖₂（12 关节，joint_pos_penalty 作用量）---")
    _stats(jd_mean_per_env, "joint_dev 每 env 均值", " rad")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
