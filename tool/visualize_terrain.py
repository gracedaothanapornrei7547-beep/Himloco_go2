# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""多地形可视化：播放 checkpoint，用跟随相机录成 mp4。

HIMLoco 的原始环境 step 返回 6 元组（obs, obs_before_reset, reward, terminated,
timeout, extras），与 gymnasium RecordVideo 的 5 元组约定不兼容（ValueError:
too many values to unpack）。本脚本不依赖 gymnasium RecordVideo，而是在场景里加
一台跟随相机的 CameraCfg，每步从 camera.data.output["rgb"] 取帧，用 imageio 写 mp4。

用法（在 himloco_lab 根目录）：
    ./run_himloco.sh scripts/himloco_rsl_rl/visualize_terrain.py \
        --task Unitree-Go2-Velocity-Play --agent himloco_rsl_rl_cfg \
        --checkpoint <path/to/model_xxx.pt> \
        --cmd_vx 1.5 --num_envs 25 --video_length 800 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize a checkpoint on multi-terrain (record mp4).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=25, help="Number of parallel environments.")
parser.add_argument("--video_length", type=int, default=800, help="Recorded video length (steps, at 50 Hz).")
parser.add_argument("--cmd_vx", type=float, default=1.5, help="Pinned forward command (m/s).")
parser.add_argument("--follow_env", type=int, default=0, help="Which environment the camera follows.")
parser.add_argument("--cam_res", type=int, default=640, help="Camera width (height = width*9/16).")
parser.add_argument(
    "--max_init_terrain_level", type=int, default=None, help="Override play env's max_init_terrain_level."
)
parser.add_argument("--view", action="store_true",
                    help="开 Isaac Sim 图形界面实时看（需 HIMLOCO_KEEP_DISPLAY=1 且不传 --headless）。")
parser.add_argument("--view_follow", action="store_true",
                    help="视口每步跟随机器人。默认**关** —— 视口只在启动时摆一次，之后完全不干预，"
                         "你可以自由拖鼠标/WASD 看任何角度。")
parser.add_argument("--until_closed", action="store_true",
                    help="不设步数上限，一直跑到你关掉 Isaac 窗口为止（需 --view）。"
                         "默认是跑满 --video_length 就退出并关窗。")
parser.add_argument("--keep_cmd", action="store_true",
                    help="不改动命令配置，原样用环境自带的口径（如 p9 的 Play cfg 已钉好 1.0m/s 直线）。")
parser.add_argument("--stair_h", type=float, default=None,
                    help="固定台阶高度（米），如 0.1 = 10cm。设了就把上/下楼梯的 step_height_range "
                         "钉成 (h,h)，并把其余地形的 proportion 全归零（只留楼梯）。只覆盖运行时 cfg。")
parser.add_argument("--stair_dir", type=str, default="up", choices=["up", "down", "both"],
                    help="楼梯方向。★ 反直觉：up → pyramid_stairs_inv（出生在底），down → pyramid_stairs（出生在顶）。")
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras (RTX) so the scene camera can render
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
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
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Play a checkpoint and record an mp4 with a follow camera."""
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 固定直线指令（部署模式：heading_command=True，heading target 钉 0 = 走 +X 直线）
    # --keep_cmd 时整段跳过：p9 的 RobotPlayEnvCfg 自己已经把命令钉成「vx=1.0 直线」，
    # 用环境自带的口径比这里再覆盖一层更忠实（它用 heading_command=False + ang_vel_z=0，
    # 而不是这里的 heading 控制器）。
    if not args_cli.keep_cmd:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
        env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.heading_command = True
        env_cfg.commands.base_velocity.rel_heading_envs = 1.0
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # ⚠️ ang_vel_z 是 heading 控制器输出 clamp 范围，保持训练值 (-2,2)，不要钉 0
    if args_cli.max_init_terrain_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.max_init_terrain_level

    # ---- 固定台阶高度（只覆盖运行时 cfg，不改源码）----
    # ★ 两点必须守住（见 agent.md §10.2）：
    #   1. terrain_generator.curriculum 必须保持 True —— 它控制"按 proportion 切列"。
    #      设 False 会变成子地形随机采样，固定台阶高度就废了。
    #   2. 方向↔地形名反直觉：pyramid_stairs_inv 出生在**底**→上楼梯；pyramid_stairs 出生在**顶**→下楼梯。
    if args_cli.stair_h is not None:
        hh = float(args_cli.stair_h)
        sub = env_cfg.scene.terrain.terrain_generator.sub_terrains
        want = {"up": ("pyramid_stairs_inv",), "down": ("pyramid_stairs",),
                "both": ("pyramid_stairs_inv", "pyramid_stairs")}[args_cli.stair_dir]
        missing = [k for k in want if k not in sub]
        if missing:
            raise SystemExit(f"[ERR] 地形里没有 {missing}；现有：{list(sub)}")
        for v in sub.values():
            v.proportion = 0.0
        for k in want:
            sub[k].proportion = 1.0 / len(want)
            sub[k].step_height_range = (hh, hh)
        assert env_cfg.scene.terrain.terrain_generator.curriculum is True, \
            "terrain_generator.curriculum 必须为 True，否则固定台阶高度失效"
        print(f"[INFO] 固定台阶 {hh*100:.1f}cm  方向={args_cli.stair_dir}  "
              f"保留地形={list(want)}（其余 proportion 归零），step_width={sub[want[0]].step_width}m")

    # 关掉 20s time_out，让机器人连续走完录制长度
    env_cfg.terminations.time_out = None

    # 加跟随相机。注意：必须用 {ENV_REGEX_NS} 按 env 复制 —— 全局单相机（无 regex）在
    # scene.reset(env_ids) 时 sensor_base 会把 env_ids 索引进 (1,) 缓冲，触发 CUDA device-side assert。
    height = int(args_cli.cam_res * 9 / 16)
    env_cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        data_types=["rgb"],
        width=args_cli.cam_res,
        height=height,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
    )

    # resolve checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

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

    robot = env.unwrapped.scene["robot"]
    camera = env.unwrapped.scene["camera"]
    dt = env.unwrapped.step_dt
    fps = int(round(1.0 / dt))

    # 输出 mp4 路径（放在 checkpoint 同目录下）
    out_dir = os.path.join(log_dir, "videos", "play")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"terrain_vx{args_cli.cmd_vx}_env{args_cli.follow_env}.mp4")

    print(f"[INFO] Recording follow-cam video: {out_path}")
    print(f"[INFO] num_envs={env.num_envs}  cmd_vx={args_cli.cmd_vx} m/s  dt={dt:.4f}s  fps={fps}  "
          f"video_length={args_cli.video_length} steps (~{args_cli.video_length * dt:.1f}s)")

    # 跑多少步：默认固定 video_length；--until_closed 则一直跑到你关窗口为止。
    n_steps = args_cli.video_length
    if args_cli.until_closed:
        if not args_cli.view:
            raise SystemExit("[ERR] --until_closed 只在 --view 下有意义（headless 没有窗口可关）")
        n_steps = None
        print("[INFO] --until_closed：不设步数上限，跑到你关掉 Isaac 窗口为止（Ctrl-C 也可以）")

    import imageio.v2 as imageio

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)

    obs = env.get_observations()
    follow = args_cli.follow_env
    cam_eye_off = torch.tensor([-3.5, 0.0, 1.8], device=env.device)
    cam_tgt_off = torch.tensor([1.5, 0.0, 0.1], device=env.device)

    # 图形界面：★ 视口只在**启动时摆一次**，之后绝不再碰 —— 每步都 set_camera_view 会把
    #   用户的鼠标视角在下一帧拽回去，等于锁死视角。要相机一路跟着跑请显式加 --view_follow。
    sim_ctx = env.unwrapped.sim
    view_dt = (1.0 / fps) if args_cli.view else 0.0
    next_tick = time.time() + view_dt
    if args_cli.view:
        root0 = robot.data.root_pos_w[follow]
        sim_ctx.set_camera_view((root0 + cam_eye_off).tolist(), (root0 + cam_tgt_off).tolist())
        print(f"[INFO] 图形界面已开：视口只在启动时对准过 env{follow}，之后不干预 —— "
              f"鼠标/WASD 随便拖；按实时 {fps} fps 节流")
        if args_cli.view_follow:
            print("[INFO] --view_follow 已开：视口会一路跟着跑（这会覆盖你的手动视角）")

    try:
        step = 0
        x_start = robot.data.root_pos_w[follow, 0].item()
        while (n_steps is None and simulation_app.is_running()) or (n_steps is not None and step < n_steps):
            # 跟随相机：每台相机跟随自己的机器人（先摆相机，再步进；渲染发生在 step 内）
            root_pos = robot.data.root_pos_w  # (N,3)
            eyes = root_pos + cam_eye_off     # (N,3)
            tgts = root_pos + cam_tgt_off     # (N,3)
            camera.set_world_poses_from_view(eyes, tgts)
            if args_cli.view and args_cli.view_follow:
                sim_ctx.set_camera_view(eyes[follow].tolist(), tgts[follow].tolist())

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _, _, _, _ = env.step(actions)

            # 取 follow_env 的帧写视频（rgb: (N, H, W, 3) uint8）
            frame = camera.data.output["rgb"][follow].cpu().numpy()
            writer.append_data(frame)

            if view_dt:
                # 不限速界面会一闪而过；按实时节流（跑不动就自然掉帧）
                next_tick += view_dt
                slack = next_tick - time.time()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_tick = time.time()

            step += 1
            if step % 100 == 0:
                # 客观读数，免得只靠肉眼："爬没爬上去"看 z，"走没走"看 x 位移。
                # ★ 地形等级也打出来 —— 否则"看着差"分不清是策略差还是地形太难（p9 训练
                #   均衡在 level≈3.4，而 Play cfg 的 max_init_terrain_level=10 会给它更难的）。
                p = robot.data.root_pos_w[follow]
                try:
                    lvl = env.unwrapped.scene.terrain.terrain_levels[follow].item()
                    lvl_s = f"{lvl:.1f}"
                except Exception:
                    lvl_s = "?"
                print(f"[INFO] step {step}  env{follow}: 高度 z={p[2].item():.3f}m  "
                      f"前进 x={p[0].item() - x_start:.3f}m  地形等级={lvl_s}  "
                      f"{'(无上限，跑到你关窗)' if n_steps is None else f'/{n_steps}'}")
    finally:
        writer.close()
        env.close()

    print(f"[INFO] Video written to: {out_path}")
    print(f"[INFO] done")


if __name__ == "__main__":
    main()
    simulation_app.close()
