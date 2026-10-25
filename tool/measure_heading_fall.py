# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""部署验证：50m 直线行走的「摔倒率 + 航向漂移（横向偏移）」。

在无限平地上命令固定 1.0 m/s 直线（body-frame vx=1.0, vy=0, wz=0，heading_command 关），
关掉 20s time_out 让机器人连续走满 50m，测：

  - 摔倒率：base_contact（身体触地）终止的 env 占比（平地应为 ~0，作 sanity check）
  - 横向偏移：到 10/25/50m 时，横向偏离初始前进方向的距离 |y - y0|（航向漂移的累积效果）
  - 航向漂移：到 10/25/50m 时的 yaw - yaw0（带符号，看是系统性偏左/右还是随机摆动）

用法（在 himloco_lab 根目录）：
    ./run_himloco.sh scripts/himloco_rsl_rl/measure_heading_fall.py \
        --task Unitree-Go2-Velocity-Play \
        --checkpoint <path/to/model_xxx.pt> \
        --num_envs 64 --distance 50 --headless
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Measure fall rate + heading drift over a 50m straight walk.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--env_spacing", type=float, default=5.0, help="Spawn grid spacing (m) to avoid lane collisions.")
parser.add_argument("--distance", type=float, default=50.0, help="Target straight-line distance (m).")
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Pinned forward command (m/s).")
parser.add_argument(
    "--heading_mode",
    action="store_true",
    default=False,
    help="Deployment-faithful mode: use heading command (target pinned to 0 = walk straight in +X), "
    "instead of a raw zero-yaw-rate command. The policy is trained with heading_command=True, so this "
    "is in-distribution; the raw-yaw mode is OOD and produces a spurious left veer.",
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


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rad, about world Z) from quaternions (N,4) in (w,x,y,z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Measure fall rate + heading drift over a straight 50m walk."""
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing

    # ★ 平地无障碍：程序化地形换成无限平面（航向漂移必须在地形无关的平地上测才干净）
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.curriculum.terrain_levels = None
    # 固定直线指令（body-frame vy=0；vx 钉死 cmd_vx）
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    # 关键：同时钉死 80% anchor 组采样范围，否则只有 20% frontier 组被钉在 cmd_vx，
    # 其余 env 仍从默认 (-1,1) 采样 → "13/64 到 50m" 其实是 13 个 frontier env，不是 51 个跑不到。
    env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    if args_cli.heading_mode:
        # 部署模式（策略训练时就是 heading_command=True + rel_heading_envs=1.0）：
        # heading target 钉 0（初始 yaw=0 → 走 +X 直线），yaw 指令由 heading 误差反馈控制器
        # 每步重算（wz = clip(0.5·wrap(0−heading_w), −2, 2)），主动纠正漂移。
        env_cfg.commands.base_velocity.heading_command = True
        env_cfg.commands.base_velocity.rel_heading_envs = 1.0
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # ⚠️★ 必须显式恢复 ang_vel_z 到训练值 (-2,2)：RobotPlayEnvCfg.__post_init__ 把
        #   ranges.ang_vel_z 钉成了 (-0,0)=(0,0)，而 heading 控制器用它作 clamp 范围
        #   （wz = clip(0.5·wrap(0−heading_w), ang_vel_z[0], ang_vel_z[1])）→ 修正量被
        #   clamp 死成 0，航向反馈等于从没生效。这里恢复 (-2,2) 才真正测到航向修正。
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
    else:
        # 裸偏航速率模式：钉死 wz=0。这是 OOD（训练时是 heading 模式），会产生虚假左拐。
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    # 关掉 20s time_out：让机器人连续走满 50m（base_contact 摔倒 + out_of_bounds 保留）
    env_cfg.terminations.time_out = None

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
    max_steps = int(args_cli.distance / args_cli.cmd_vx / dt) + 250  # +5s margin

    print(f"[INFO] control dt={dt:.4f}s  num_envs={N}  distance={args_cli.distance}m  "
          f"cmd_vx={args_cli.cmd_vx}m/s  max_steps={max_steps} (~{max_steps * dt:.1f}s)")

    # --- initial state (before stepping) ---
    obs = env.get_observations()
    pos0 = robot.data.root_pos_w.clone()  # (N,3) world position
    psi0 = yaw_from_quat(robot.data.root_quat_w.clone())  # (N,) initial yaw
    c0, s0 = torch.cos(psi0), torch.sin(psi0)

    # --- per-env tracking ---
    fell = torch.zeros(N, dtype=torch.bool, device=device)  # 摔倒（base_contact）过的 env
    recorded = torch.zeros(N, dtype=torch.bool, device=device)  # 已记录 50m 结果的 env

    targets = [10.0, 25.0, 50.0]
    # 每个目标距离存：signed lateral / |lateral| / heading drift
    lat_signed_at = {t: torch.full((N,), float("nan"), device=device) for t in targets}
    lat_abs_at = {t: torch.full((N,), float("nan"), device=device) for t in targets}
    heading_at = {t: torch.full((N,), float("nan"), device=device) for t in targets}
    reached_at = {t: torch.zeros(N, dtype=torch.bool, device=device) for t in targets}

    for step in range(max_steps):
        # 当前状态（上一拍 env.step 已含 reset）
        pos = robot.data.root_pos_w
        psi = yaw_from_quat(robot.data.root_quat_w)
        dx = pos[:, 0] - pos0[:, 0]
        dy = pos[:, 1] - pos0[:, 1]
        # 在初始前进方向坐标系里的前进/横向位移（psi0≈0 时 forward≈x-x0, lateral≈y-y0）
        forward = dx * c0 + dy * s0
        lateral = -dx * s0 + dy * c0
        hd = wrap_to_pi(psi - psi0)

        # 记录每个目标距离首次越过时的状态（幸存者 + 未记录）
        for t in targets:
            crossed = (forward >= t) & (~fell) & (~reached_at[t])
            lat_signed_at[t][crossed] = lateral[crossed]
            lat_abs_at[t][crossed] = lateral[crossed].abs()
            heading_at[t][crossed] = hd[crossed]
            reached_at[t] |= crossed

        # 步进
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, dones, _, _, _ = env.step(actions)
        fell |= dones.to(torch.bool)

        # 全部幸存者都越过 50m 就提前收工
        if reached_at[targets[-1]].sum().item() >= (~fell).sum().item():
            break

    # --- 汇总 ---
    survivors = ~fell
    n_fell = fell.sum().item()
    n_surv = survivors.sum().item()
    print("\n================ 50m 直线行走验证结果 ================")
    print(f"num_envs={N}  distance={args_cli.distance}m  cmd_vx={args_cli.cmd_vx}m/s  dt={dt:.4f}s")
    print(f"摔倒率（base_contact 终止）: {n_fell}/{N} = {100.0 * n_fell / N:.2f}%")

    def _stats(t: torch.Tensor, name: str, unit: str = ""):
        valid = t[~torch.isnan(t)]
        if valid.numel() == 0:
            print(f"[{name}] no valid samples")
            return
        print(
            f"[{name}] n={valid.numel()}  mean={valid.mean().item():+.4f}{unit}  "
            f"std={valid.std().item():.4f}  median={valid.median().item():+.4f}  "
            f"p90={torch.quantile(valid, 0.90).item():+.4f}  max={valid.abs().max().item():.4f}"
        )

    for t in targets:
        n_reach = reached_at[t].sum().item()
        print(f"\n--- 到 {t:.0f}m（{n_reach}/{n_surv} 幸存者已到达）---")
        _stats(lat_abs_at[t], f"横向偏移 |lateral| @{t:.0f}m", " m")
        _stats(lat_signed_at[t], f"横向偏移 signed @{t:.0f}m", " m")
        _stats(heading_at[t], f"航向漂移 yaw-yaw0 @{t:.0f}m", " rad")

    # 幸存者中没到 50m 的（应该很少，说明没达到 1m/s）
    not_reached = survivors & (~reached_at[targets[-1]])
    if not_reached.any():
        print(f"\n[WARN] {not_reached.sum().item()} 个幸存者未在 {max_steps} 步内到达 {args_cli.distance}m")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
