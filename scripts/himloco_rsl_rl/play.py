# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent trained with HimLoco RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with HimLoco RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--env_cfg_entry_point", type=str, default="play_env_cfg_entry_point", help="Name of the play environment configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--cmd_vx", type=float, default=None, help="Pin forward command (m/s); overrides the play env's default 1.0."
)
parser.add_argument(
    "--heading_mode",
    action="store_true",
    default=False,
    help="Deployment-faithful mode: heading command (target=initial yaw) + ang_vel_z=(-2,2) so the heading "
    "controller actually corrects drift. The play env default is heading_command=False + ang_vel_z=(0,0), "
    "which is OOD (clamps the heading controller to zero) and produces a large left veer.",
)
parser.add_argument(
    "--heading_control_stiffness",
    type=float,
    default=None,
    help="Override the heading P-gain (default 0.5). Match the value the checkpoint was trained with.",
)
parser.add_argument(
    "--step_height",
    type=float,
    default=None,
    help="Override staircase step height (m), fixed for all difficulties. E.g. --step_height 0.10 for 10cm stairs.",
)
parser.add_argument(
    "--terrain_curriculum",
    type=str,
    choices=["keep", "isaaclab", "local"],
    default="keep",
    help="Which terrain_levels_vel to use. 'isaaclab' = IsaacLab 原版（固定 4m 门槛，08-22/08-24 用的）；"
    "'local' = 本地严格门槛版（需整集接近指令速度，均衡 level≈3.4，p5c/p9/p10 用的）。",
)
parser.add_argument(
    "--curriculums_limit",
    type=float,
    default=None,
    help="Override curriculums_limit_ranges to (-V, +V). p9 那一版是 1.0（速度上限锁 ±1.0 m/s）。",
)
parser.add_argument(
    "--zero_cmd",
    action="store_true",
    default=False,
    help="Pin the command to 0 (stand-still check).",
)
parser.add_argument(
    "--stair_only",
    type=str,
    choices=["up", "down", "both"],
    default=None,
    help="If set, focus all terrain columns on stairs only ('up'=pyramid_stairs_inv, 'down'=pyramid_stairs, 'both'=both).",
)
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from himloco_lab.utils import export_himloco_policy_as_jit, export_himloco_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Play with HimLoco RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # 提速可视化：把 play 环境默认的 vx=1.0 钉成 cmd_vx（同时钉死 80% anchor 组的采样范围）
    if args_cli.cmd_vx is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
        env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)

    # 部署口径 heading 指令：play 配置把 heading_command 关掉且 ang_vel_z 钉成 (0,0)，航向反馈被 clamp 死。
    # 恢复 heading_command=True + ang_vel_z=(-2,2) + heading target 钉初始朝向，才与训练口径一致、走直线。
    if args_cli.heading_mode:
        env_cfg.commands.base_velocity.heading_command = True
        env_cfg.commands.base_velocity.rel_heading_envs = 1.0
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
    if args_cli.heading_control_stiffness is not None:
        env_cfg.commands.base_velocity.heading_control_stiffness = args_cli.heading_control_stiffness
    # 地形难度课程版本切换：p11 之前用 IsaacLab 原版（按行走距离升降），p11/p12 用本地均匀重抽版。
    # 两者行为差别很大，可视化时必须和 checkpoint 的训练口径对齐，否则看到的是 OOD 地形分布。
    if args_cli.terrain_curriculum != "keep":
        if args_cli.terrain_curriculum == "isaaclab":
            from isaaclab_tasks.manager_based.locomotion.velocity.mdp import curriculums as _isaac_curr

            env_cfg.curriculum.terrain_levels.func = _isaac_curr.terrain_levels_vel
        else:  # local（本地严格门槛版）
            from himloco_lab.tasks.locomotion.mdp import curriculums as _him_curr

            env_cfg.curriculum.terrain_levels.func = _him_curr.terrain_levels_vel
        print(f"[INFO] terrain_levels_vel -> {env_cfg.curriculum.terrain_levels.func.__module__}")
    # 速度课程上限（p9 那一版是 (-1,1)，即速度上限锁在 ±1.0 m/s）
    if args_cli.curriculums_limit is not None:
        v = args_cli.curriculums_limit
        env_cfg.commands.base_velocity.curriculums_limit_ranges = (-v, v)
        env_cfg.curriculum.lin_vel_cmd_levels = None  # 关掉爬坡项，否则它会按时间把上限推回 (-2,2)
    if args_cli.zero_cmd:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (0.0, 0.0)
    # 覆盖楼梯台阶高度（固定值，不随难度递增）。上/下楼梯都改。
    if args_cli.step_height is not None:
        for key in ("pyramid_stairs", "pyramid_stairs_inv"):
            st = env_cfg.scene.terrain.terrain_generator.sub_terrains.get(key)
            if st is not None:
                st.step_height_range = (args_cli.step_height, args_cli.step_height)

    # 聚焦单向/双向楼梯地形（纯看爬楼梯/下楼梯效果）
    if getattr(args_cli, "stair_only", None) is not None and env_cfg.scene.terrain.terrain_generator is not None:
        gen = env_cfg.scene.terrain.terrain_generator
        for k in list(gen.sub_terrains.keys()):
            gen.sub_terrains[k].proportion = 0.0
        if args_cli.stair_only == "up":
            gen.sub_terrains["pyramid_stairs_inv"].proportion = 1.0
        elif args_cli.stair_only == "down":
            gen.sub_terrains["pyramid_stairs"].proportion = 1.0
        elif args_cli.stair_only == "both":
            gen.sub_terrains["pyramid_stairs_inv"].proportion = 0.5
            gen.sub_terrains["pyramid_stairs"].proportion = 0.5
        print(f"[INFO] 楼梯专属模式已激活: stair_only={args_cli.stair_only}, 所有环境均生成于指定楼梯上")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length
    )

    print(f"[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
    print(f"[INFO] history_length: {env.history_length}")
    print(f"[INFO] num_obs (total): {env.num_obs}")
    if env.num_one_step_privileged_obs is not None:
        print(f"[INFO] num_one_step_privileged_obs: {env.num_one_step_privileged_obs}")
        print(f"[INFO] privileged_history_length: {env.privileged_history_length}")
        print(f"[INFO] num_privileged_obs (total): {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # create runner from HimLoco RSL-RL
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # load the checkpoint
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.device)

    # Export the policy for deployment.  The HIM variant exports a dual
    # (encoder + policy) network; the ordinary-history MLP baseline has no
    # estimator branch, so it is skipped here.
    if hasattr(runner.alg.actor_critic, "estimator"):
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        print(f"[INFO] Exporting HimLoco dual network to: {export_model_dir}")

        # Export as TorchScript JIT (combined encoder + policy)
        export_himloco_policy_as_jit(
            runner.alg.actor_critic,
            path=export_model_dir,
            policy_filename="policy.pt"
        )

        # Export as ONNX (separate encoder and policy)
        export_himloco_policy_as_onnx(
            runner.alg.actor_critic,
            path=export_model_dir,
            encoder_filename="encoder.onnx",
            policy_filename="policy.onnx",
            verbose=False
        )
    else:
        print("[INFO] MLP control policy (no estimator); skipping JIT/ONNX export.")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # simulate environment
    print("[INFO] Starting playback...")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
