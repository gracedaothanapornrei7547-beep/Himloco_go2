# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""稳态误差测量：在「平地无障碍 + 固定 1.0 m/s 直行」下，统计速度跟踪的稳态误差。

用法（在 himloco_lab 根目录）：
    python scripts/himloco_rsl_rl/measure_steady_error.py \
        --task Unitree-Go2-Velocity-Play \
        --checkpoint <path/to/model_xxx.pt> \
        --num_steps 2000 \
        --headless

输出：每步瞬时误差 err = ||cmd_xy - vel_xy||（body frame），跳过起步瞬态后的稳态统计。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Measure steady-state velocity-tracking error on flat ground.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of control steps to measure.")
parser.add_argument("--warmup", type=int, default=150, help="Number of initial steps to discard as transient.")
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
    """Measure steady-state tracking error."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ★ 平地无障碍：把程序化地形换成无限平面
    env_cfg.scene.terrain.terrain_type = "plane"
    # 禁用地形难度课程：terrain_levels_vel 课程项会访问 terrain_generator.size，
    # 平地测量不需要地形课程，禁掉它以避免对 generator 的依赖（保留 generator 本身）。
    env_cfg.curriculum.terrain_levels = None
    print("[INFO] Terrain overridden to flat plane (terrain_type='plane', terrain curriculum disabled).")

    # resolve checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
        from isaaclab_tasks.utils import get_checkpoint_path
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length,
    )
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create runner and load checkpoint
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    dt = env.unwrapped.step_dt
    print(f"[INFO] control dt = {dt:.4f} s  (num_steps={args_cli.num_steps}, warmup={args_cli.warmup})")

    # reset environment
    obs = env.get_observations()

    err_steps = []
    vx_steps = []
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            # record current command and measured velocity (body frame, xy plane)
            cmd = env.unwrapped.command_manager.get_command("base_velocity")[:, :2]
            vel = env.unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
            err = torch.norm(cmd - vel, dim=-1)  # (num_envs,)
            err_steps.append(err)
            vx_steps.append(vel[:, 0])

            # step policy + env
            actions = policy(obs)
            obs, *_ = env.step(actions)

    err = torch.stack(err_steps, dim=0)  # (num_steps, num_envs)
    vx = torch.stack(vx_steps, dim=0)

    # split transient vs steady-state
    err_ss = err[args_cli.warmup:]
    vx_ss = vx[args_cli.warmup:]

    def stats(t: torch.Tensor, name: str):
        flat = t.flatten().float()
        print(
            f"[{name}] mean={flat.mean().item():.4f}  std={flat.std().item():.4f}  "
            f"median={flat.median().item():.4f}  p90={torch.quantile(flat, 0.90).item():.4f}  "
            f"max={flat.max().item():.4f}"
        )

    print("\n================ 稳态误差测量结果（平地 · 固定 1.0 m/s 直行）================")
    print(f"num_envs={err.shape[1]}, num_steps={err.shape[0]}, warmup={args_cli.warmup} 步")
    stats(err, "全程瞬时误差 ||cmd-vel||_xy")
    stats(err_ss, "稳态瞬时误差 ||cmd-vel||_xy")
    stats(vx_ss, "稳态前进速度 vx (目标 1.0)")

    # 稳态误差随时间的变化（每 100 步均值），看是否真的收敛到稳态
    print("\n稳态误差均值（每 100 步一个窗口）:")
    for i in range(args_cli.warmup, err.shape[0], 100):
        seg = err[i : i + 100].mean().item()
        print(f"  step {i:5d}~{min(i+100, err.shape[0]):5d}: mean_err = {seg:.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
