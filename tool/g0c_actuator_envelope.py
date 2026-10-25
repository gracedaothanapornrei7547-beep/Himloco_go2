# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""G0c：执行路径一致性 —— Go2HV 力矩-转速包络有多约束。

为什么需要这一道：
    G0a 证明"导出没坏"，G0b 证明"obs 拼装与训练一致"。但两者都没碰**执行路径**：
        action --×0.25 + default--> q_des --PD(kp,kd)--> tau --[Go2HV T-N 包络]--> 电机
    三方对这个包络的处理**根本不同**：
      - IsaacLab：`unitree_actuators.py::UnitreeActuator._clip_effort` 实现了它
        （Go2HV: X1=13.5, X2=30 rad/s；Y1=20.2, Y2=23.4 N·m）
      - C++ 部署（`deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`）：**不含**包络，
        只下发 kp/kd/q_des/dq=0/tau=0 —— 包络由真机电机固件执行
      - MuJoCo sim2sim：**必须自己补**，否则 agent.md §8.5 记的"落地瞬间 PD 输出 ~145 N·m
        把脚打进地里翻倒"就会复现（`MUJOCO_LOG.TXT` 里的 NaN 大概率就是它）

本脚本做什么：
    在真环境里逐步算**朴素 PD**（部署侧公式：`kp*(q_des-q) + kd*(0-dq)`），
    与 IsaacLab 实际施加的 `applied_torque`（已过包络）对比，回答：
      (1) 朴素 PD 的力矩有多大？（若远超 Y1=20.2，说明包络是**强约束**）
      (2) 包络多久绑定一次？（绑定比例 = MuJoCo 不加包络会跑偏的时间比例）
      (3) 两者的最大差是多少？

用法：
    ./run_himloco.sh scripts/himloco_rsl_rl/g0c_actuator_envelope.py \
        --task Unitree-Go2-Velocity-Play --checkpoint <run_dir>/model_XXXX.pt \
        --num_envs 8 --num_steps 400
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="G0c: actuator / Go2HV torque-speed envelope check.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="himloco_rsl_rl_cfg", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=400, help="Number of control steps.")
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Pinned forward command (m/s).")
cli_args.add_himloco_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in sys.argv:
    sys.argv.append("--headless")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch
import yaml

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Go2HV T-N 曲线常数（source/himloco_lab/himloco_lab/assets/unitree_actuators.py:113-118）
GO2HV = dict(X1=13.5, X2=30.0, Y1=20.2, Y2=23.4)


def go2hv_clip(tau, dq, x1=GO2HV["X1"], x2=GO2HV["X2"], y1=GO2HV["Y1"], y2=GO2HV["Y2"]):
    """复刻 UnitreeActuator._clip_effort：同向限 Y1、反向限 Y2，|dq|>X1 后线性衰减到 0。"""
    same = (dq * tau) > 0
    max_e = torch.where(same, torch.full_like(tau, y1), torch.full_like(tau, y2))
    k = -max_e / (x2 - x1)
    limit = k * (dq.abs() - x1) + max_e
    limit = limit.clip(min=0.0)
    max_e = torch.where(dq.abs() < x1, max_e, limit)
    return torch.clip(tau, -max_e, max_e)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        from isaaclab_tasks.utils import get_checkpoint_path

        log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    deploy_path = os.path.abspath(os.path.join(os.path.dirname(resume_path), "params", "deploy.yaml"))
    with open(deploy_path) as f:
        deploy = yaml.unsafe_load(f)
    act = deploy["actions"]["JointPositionAction"]
    dev = torch.device(agent_cfg.device)
    scale = torch.tensor(act["scale"], dtype=torch.float, device=dev)
    offset = torch.tensor(act["offset"], dtype=torch.float, device=dev)
    kp = torch.tensor(deploy["stiffness"], dtype=torch.float, device=dev)
    kd = torch.tensor(deploy["damping"], dtype=torch.float, device=dev)
    print(f"[G0c] kp={kp[0].item()} kd={kd[0].item()} scale={scale[0].item()}")
    print(f"[G0c] deploy.yaml 里**没有**力矩包络字段（包络由真机固件执行）→ 本脚本用 Go2HV 常数自算：{GO2HV}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length,
    )
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    # 记录：朴素 PD（未过包络）与 IsaacLab 实际施加力矩
    rec = {"naive": [], "applied": [], "clipped": [], "qdes_gap": []}
    obs = env.get_observations()
    actions = policy(obs)
    for _ in range(args_cli.num_steps):
        with torch.inference_mode():
            obs, *_ = env.step(actions)
            q = robot.data.joint_pos
            dq = robot.data.joint_vel
            q_des = actions * scale + offset            # 部署侧公式（joint_actions.h:43）
            naive = kp * (q_des - q) + kd * (0.0 - dq)  # 部署下发 dq=0 → D 项为 -kd*dq
            applied = robot.data.applied_torque
            rec["naive"].append(naive.detach().clone())
            rec["applied"].append(applied.detach().clone())
            rec["clipped"].append(go2hv_clip(naive, dq).detach().clone())
            rec["qdes_gap"].append((q_des - q).abs().detach().clone())
            actions = policy(obs)

    N = torch.cat(rec["naive"]).abs()
    A = torch.cat(rec["applied"])
    C = torch.cat(rec["clipped"])
    n_clip = int((C - torch.cat(rec["naive"])).abs().gt(1e-6).sum())
    tot = N.numel()

    print("\n" + "=" * 92)
    print(f"[G0c] 朴素 PD vs IsaacLab applied_torque   （{tot} 个 关节×步 样本）")
    print("=" * 92)
    print(f"  朴素 PD |tau| : mean={N.mean():8.3f}  p95={torch.quantile(N.flatten().float(), 0.95):8.3f}  "
          f"p99={torch.quantile(N.flatten().float(), 0.99):8.3f}  max={N.max():8.3f} N·m")
    print(f"  applied |tau| : mean={A.abs().mean():8.3f}  p99={torch.quantile(A.abs().flatten().float(), 0.99):8.3f}  "
          f"max={A.abs().max():8.3f} N·m")
    print(f"  Go2HV 包络把朴素 PD 改写的样本比例 : {n_clip / tot * 100:.2f}%   ({n_clip}/{tot})")
    over_y1 = int((N > GO2HV["Y1"]).sum())
    over_y2 = int((N > GO2HV["Y2"]).sum())
    print(f"  朴素 PD 超过 Y1={GO2HV['Y1']} 的样本 : {over_y1 / tot * 100:.2f}%")
    print(f"  朴素 PD 超过 Y2={GO2HV['Y2']} 的样本 : {over_y2 / tot * 100:.2f}%")
    g = torch.cat(rec["qdes_gap"])
    print(f"  |q_des - q| : mean={g.mean():.4f} p99={torch.quantile(g.flatten().float(), 0.99):.4f} "
          f"max={g.max():.4f} rad")
    d = (torch.cat(rec["clipped"]) - A).abs()
    print(f"\n  自算的 Go2HV 限幅 tau  vs  IsaacLab applied_torque：")
    print(f"      mean={d.mean():.4f}  p99={torch.quantile(d.flatten().float(), 0.99):.4f}  max={d.max():.4f} N·m")
    print("      （残差主要来自 IsaacLab 的 friction=0.01 摩擦项与 armature，部署侧没有这两项）")

    print("\n[G0c] 判读：")
    if over_y2 / tot > 0.01:
        print(f"  ⚠️ 朴素 PD 有 {over_y2 / tot * 100:.2f}% 的样本超过 Y2={GO2HV['Y2']} N·m —— "
              f"**包络是强约束**。MuJoCo sim2sim 若不实现它，这些时刻的力矩会被放大到 "
              f"{N.max():.1f} N·m（实测峰值），足以把脚打进地面 → 就是 §8.5 bug #3。")
    else:
        print(f"  朴素 PD 超过 Y2 的样本仅 {over_y2 / tot * 100:.2f}% —— 包络在这段轨迹上基本不绑定；"
              f"但落地/碰撞等瞬态仍可能触发，MuJoCo 侧仍建议实现。")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
