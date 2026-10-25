# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""实测前进速度：平地 + 固定 cmd_vx 直线，看策略实际能达到多快。

输出：稳态 body-frame 前进速度 vx 的分布（mean/median/p90/max）+ 跟踪误差。
用于诊断「提速训练到底有没有把 top speed 抬上去」——比「到没到 50m」更直接。

用法（在 himloco_lab 根目录）：
    ./run_himloco.sh scripts/himloco_rsl_rl/measure_achieved_speed.py \
        --task Unitree-Go2-Velocity-Play \
        --checkpoint <path/to/model_xxx.pt> \
        --cmd_vx 1.5 --num_envs 64 --headless
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Measure achieved forward speed at a pinned command.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--env_spacing", type=float, default=5.0, help="Spawn grid spacing (m).")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of control steps to measure.")
parser.add_argument("--warmup", type=int, default=150, help="Initial steps to discard as transient.")
parser.add_argument("--cmd_vx", type=float, default=1.5, help="Pinned forward command (m/s).")
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
    """Measure achieved forward speed at pinned command."""
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing

    # 平地 + 关地形课程 + 固定直线指令
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.curriculum.terrain_levels = None
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    # 关键：同时钉死 80% anchor 组的采样范围，否则只有 20% frontier 组(env_id<0.2N)被钉在 cmd_vx，
    # 其余 51 个 env 仍从默认 (-1,1) 采样 → 测出来的是混合速度，不是"能不能跑到 cmd_vx"。
    env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
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
    dt = env.unwrapped.step_dt

    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length,
    )
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    print(f"[INFO] control dt={dt:.4f}s  num_envs={args_cli.num_envs}  cmd_vx={args_cli.cmd_vx}m/s  "
          f"num_steps={args_cli.num_steps}  warmup={args_cli.warmup}")

    obs = env.get_observations()
    vx_steps = []
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            vx_steps.append(robot.data.root_lin_vel_b[:, 0].clone())
            actions = policy(obs)
            obs, *_ = env.step(actions)

    vx = torch.stack(vx_steps, dim=0)  # (num_steps, num_envs)
    vx_ss = vx[args_cli.warmup:]  # steady state

    flat = vx_ss.flatten().float()
    print("\n================ 实测前进速度（平地 · 固定 %.1f m/s）================" % args_cli.cmd_vx)
    print(f"num_envs={vx.shape[1]}  num_steps={vx.shape[0]}  warmup={args_cli.warmup}")
    print(f"稳态 vx: mean={flat.mean().item():.4f}  std={flat.std().item():.4f}  "
          f"median={flat.median().item():.4f}  p90={torch.quantile(flat, 0.90).item():.4f}  "
          f"max={flat.max().item():.4f}")
    # 每个 env 的时间平均 vx（排除起步瞬态），看分布
    per_env = vx_ss.mean(dim=0)
    print(f"每-env 时间平均 vx: mean={per_env.mean().item():.4f}  min={per_env.min().item():.4f}  "
          f"max={per_env.max().item():.4f}  median={per_env.median().item():.4f}")
    above = {t: (per_env >= t).sum().item() for t in [1.0, 1.2, 1.3, 1.4, 1.5]}
    print(f"达到各速度阈值的 env 数: {above}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
