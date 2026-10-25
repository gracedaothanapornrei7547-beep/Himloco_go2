# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""诊断速度课程 lin_vel_cmd_levels 为什么不爬升。

在训练配置（RobotEnvCfg + --max_lin_vel_x 1.5）下跑 1 集，手动复算课程内部的
reward_low / reward_high / 阈值，看条件到底卡在哪（守卫 vs 阈值 vs 分组）。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Debug lin_vel_cmd_levels curriculum.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_lin_vel_x", type=float, default=1.5)
cli_args.add_himloco_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import himloco_lab.tasks  # noqa: F401
from himloco_lab.tasks.locomotion import mdp


@hydra_task_config(args_cli.task, "himloco_rsl_rl_cfg")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"
    if args_cli.max_lin_vel_x is not None:
        vmax = args_cli.max_lin_vel_x
        env_cfg.commands.base_velocity.curriculums_limit_ranges = (-vmax, vmax)
        env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (-vmax, vmax)
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    cmd_term = env.command_manager.get_term("base_velocity")
    rw = env.reward_manager
    term_name = "track_lin_vel_xy"
    term_cfg = rw.get_term_cfg(term_name)
    N = env.num_envs

    print("\n=== 静态配置 ===")
    print(f"num_envs={N}  rel_high_vel_envs={cmd_term.cfg.rel_high_vel_envs}")
    print(f"num_envs*rel_high = {N * cmd_term.cfg.rel_high_vel_envs}")
    print(f"ranges.lin_vel_x = {cmd_term.cfg.ranges.lin_vel_x}")
    print(f"curriculums_limit_ranges = {cmd_term.cfg.curriculums_limit_ranges}")
    print(f"low_vel_env_lin_x_ranges = {cmd_term.cfg.low_vel_env_lin_x_ranges}")
    print(f"term {term_name} weight = {term_cfg.weight}  -> threshold = weight*0.6 = {term_cfg.weight * 0.6}")
    print(f"max_episode_length (steps) = {env.max_episode_length}  max_episode_length_s = {env.max_episode_length_s}")
    print(f"step_dt = {env.step_dt}")

    # 跑一整集（零动作，只为看 reward 数值尺度；真实训练有 policy 但阈值关系不变）
    steps = env.max_episode_length
    obs = env.get_observations()
    for i in range(steps):
        act = torch.zeros((N, env.num_actions), device=env.device)
        obs = env.step(act)[0]

    # 此时 common_step_counter == steps（整集跑完），但 _episode_sums 已被 reset 清空。
    # 手动复算课程内部逻辑：用刚结束时（reset 前）本应是整集 sum 的量。
    # 为了拿到 reset 前的 sum，我们改在 reset 前一步（steps-1）复算 —— 见下方第二次循环。
    print(f"\ncommon_step_counter after {steps} steps = {env.common_step_counter}")
    print(f"guard value (counter % max_len) = {env.common_step_counter % env.max_episode_length}")

    # 重新来：跑到 steps-1（reset 前），此时 _episode_sums 仍是整集累计
    env.reset()
    rw = env.reward_manager
    obs = env.get_observations()
    for i in range(steps - 1):
        act = torch.zeros((N, env.num_actions), device=env.device)
        obs = env.step(act)[0]

    counter = env.common_step_counter
    sums = rw._episode_sums[term_name]  # (N,) 每 env 的整集累计（含 weight*dt）
    print(f"\n=== 第 {steps-1} 步（reset 前）===")
    print(f"common_step_counter = {counter}  guard = {counter % env.max_episode_length}")
    print(f"_episode_sums[{term_name}] mean = {sums.mean().item():.4f}  (raw, 含 weight*dt)")

    # 复算课程里的 reward_low / reward_high
    env_ids = torch.arange(N, device=env.device)
    rel = N * cmd_term.cfg.rel_high_vel_envs
    high_mask = env_ids < rel
    low_mask = env_ids >= rel
    high_ids = env_ids[high_mask]
    low_ids = env_ids[low_mask]

    def rw_low_high():
        rl = torch.mean(sums[low_ids]) / env.max_episode_length_s if len(low_ids) > 0 else 0.0
        rh = torch.mean(sums[high_ids]) / env.max_episode_length_s if len(high_ids) > 0 else 0.0
        return rl, rh

    rl, rh = rw_low_high()
    thr = term_cfg.weight * 0.6
    print(f"low group n={len(low_ids)}  high group n={len(high_ids)}")
    print(f"reward_low = {rl.item():.4f}   reward_high = {rh.item():.4f}   threshold = {thr:.4f}")
    print(f"reward_low > thr ? {bool(rl.item() > thr)}   reward_high > thr ? {bool(rh.item() > thr)}")
    print(f"BOTH pass ? {bool(rl.item() > thr and rh.item() > thr)}")

    # 直接调用课程函数（真实入口）看它返回什么、是否改 ranges
    before = cmd_term.cfg.ranges.lin_vel_x
    ret = mdp.lin_vel_cmd_levels(env, env_ids)
    after = cmd_term.cfg.ranges.lin_vel_x
    print(f"\n直接调用 lin_vel_cmd_levels：返回 {ret.item()}  ranges.lin_vel_x {before} -> {after}")

    # 强行把 counter 拨到边界，看守卫是否放行
    env.common_step_counter = env.max_episode_length
    ret2 = mdp.lin_vel_cmd_levels(env, env_ids)
    after2 = cmd_term.cfg.ranges.lin_vel_x
    print(f"counter 拨到 {env.common_step_counter}（guard={env.common_step_counter % env.max_episode_length}）后：返回 {ret2.item()}  ranges {after} -> {after2}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
