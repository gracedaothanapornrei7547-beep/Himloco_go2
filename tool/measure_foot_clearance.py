# Copyright (c) 2022-2025, The Isaac Lab Project Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""部署验证：摆动脚离地净空（foot clearance）+ 机身俯仰 + 机身离地净空。

针对「上楼梯前几阶卡住 → 身体着地」的症状，本探针在默认混合地形（30% 上楼梯 +
30% 下楼梯 + 平地/斜坡/障碍）上命令固定 vx 直线，连续跑固定步数，每步测量：

  - foot_clearance = 脚体中心 Z − base_height_scanner 打到的地形 Z（脚相对脚下局部地形的净空）
    —— 与 foot_min_clearance_reward 里的 foot_height 完全同一口径。只在「摆动相」
       （脚水平速度 > swing_vel_thresh）统计，因为着地脚的 clearance≈脚球半径本就不算数。
       前脚（FL/FR）摆动净空的 min/p5 就是「抬脚够不够高、会不会撞到第一级台阶」的核心指标。
  - base_pitch = 机身绕 Y 轴的俯仰角（nose-down 为负）。前脚撞到台阶 → 机身点头 → 身体着地，
       所以 min pitch（最深点头）是「卡住 → 着地」这一因果链的中间量。
  - base_height = 机身中心 Z − 脚下地形 Z（机身相对脚下地形的净空，同 measure_body_height.py）。
       min 值即「身体着地前」的最低离地净空，是「身体着地」的直接代理量。

对比口径（A/B）：
    基线 p6  model_42988.pt（双边 foot_clearance，std=0.05，无 offset）
    处理 p9  model_45987.pt（单向 foot_min_clearance，std=0.01，target=0.072）
    同一地形、同一 --step_height、同一 --cmd_vx、同一 heading 口径跑两次，
    只看三件事：前脚摆动净空是否抬升、最深点头是否减轻、最低机身净空是否抬升。

用法（在 himloco_lab 根目录）：
    ./run_himloco.sh scripts/himloco_rsl_rl/measure_foot_clearance.py \
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
parser = argparse.ArgumentParser(description="Measure swing-foot clearance + base pitch + body height over stairs.")
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
    "--swing_vel_thresh",
    type=float,
    default=0.3,
    help="Foot horizontal speed (m/s) above which a foot counts as 'swinging'.",
)
parser.add_argument(
    "--stall_vel_thresh",
    type=float,
    default=0.25,
    help="Base forward speed (m/s) below which a commanded-to-move robot counts as 'stalled'.",
)
# ---- 定位实验开关：分离「策略塌了」与「环境 OOD」----
parser.add_argument(
    "--flat_plane",
    action="store_true",
    default=False,
    help="Use an infinite flat plane instead of the generated terrain (removes stairs/slopes/obstacles).",
)
parser.add_argument(
    "--no_perturb",
    action="store_true",
    default=False,
    help="Disable the periodic external force event (±30N every 0.02s) that play keeps enabled.",
)
parser.add_argument(
    "--disable_curriculum",
    action="store_true",
    default=False,
    help="Disable BOTH curriculum terms. Also stops lin_vel_cmd_levels from overwriting ranges.lin_vel_x in play.",
)
parser.add_argument(
    "--pin_command",
    action="store_true",
    default=False,
    help="Freeze command resampling so |cmd| stays exactly cmd_vx for the whole run.",
)
parser.add_argument(
    "--terrain_only",
    type=str,
    default=None,
    help="诊断用：只保留这一种子地形（proportion=1.0，其余全归 0），与 eval_stair_stage.py 同口径。"
         "上楼梯填 pyramid_stairs_inv，下楼梯填 pyramid_stairs。"
         "★ 不隔离的话 Play 地形是 70% 楼梯 + 30% 平地/斜坡，净空统计会被平地样本稀释，"
         "量出来的 p95 会偏低 —— 正是本次要量的那个数，不能混。",
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
from isaaclab.utils.math import euler_xyz_from_quat

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
    """Measure swing-foot clearance + base pitch + body height on stairs."""
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

    # ---- 地形隔离：只留目标楼梯（与 eval_stair_stage.py 同一套做法）----
    # ★ `terrain_generator.curriculum` 必须保持 True —— 它有 curriculum 时按 proportion 切列，
    #   关掉就变成按比例随机抽子地形，proportion 全 0 会直接把地形生成搞坏。
    if args_cli.terrain_only:
        gen = env_cfg.scene.terrain.terrain_generator
        for k in gen.sub_terrains:
            gen.sub_terrains[k].proportion = 0.0
        gen.sub_terrains[args_cli.terrain_only].proportion = 1.0

    # ---- 定位实验开关 ----
    if args_cli.pin_command:
        # 命令不再重采样 → |cmd| 全程 == cmd_vx，排除"命令偶尔掉到 0"的干扰
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    if args_cli.disable_curriculum:
        # 关掉两个课程项：
        #   - terrain_levels_vel 会在每次 reset 把 env 传送随机难度行（位移指标会被污染）
        #   - lin_vel_cmd_levels 会在运行期覆写 ranges.lin_vel_x（冲掉上面钉的 cmd_vx）
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.lin_vel_cmd_levels = None
    if args_cli.no_perturb:
        # play 配置只关了 add_base_mass / randomize_rigid_body_com / push_robot，
        # external_force（±30N 每 0.02s）仍然开着 —— 单独关掉它，隔离"域随机化导致不走"
        env_cfg.events.external_force = None
    if args_cli.flat_plane:
        # 无限平地：彻底去掉台阶/斜坡/障碍，也顺带让 terrain_out_of_bounds 永远不触发
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.max_init_terrain_level = None

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

    # 脚 body 顺序 = 关节树顺序，feet_gait offset [LF, RF, LH, RH] 已确认是 FL/FR/RL/RR。
    foot_ids, foot_names = robot.find_bodies(".*_foot", preserve_order=True)
    n_feet = len(foot_ids)
    print(f"[INFO] control dt={dt:.4f}s  num_envs={N}  num_steps={args_cli.num_steps}  "
          f"cmd_vx={args_cli.cmd_vx}m/s  step_height={args_cli.step_height}m  swing_vel_thresh={args_cli.swing_vel_thresh}m/s")
    print(f"[INFO] foot body order: {foot_names}")

    def _terrain_z():
        ray_z = scanner.data.ray_hits_w[..., 2]  # (N, num_rays)
        valid = ~torch.isnan(ray_z) & ~torch.isinf(ray_z) & (ray_z.abs() < 1e6)
        masked = torch.where(valid, ray_z, torch.full_like(ray_z, float("nan")))
        return torch.nanmean(masked, dim=1)  # (N,)，全无效的 env 为 NaN

    # --- 累积统计 ---
    # foot_clearance（摆动相）per-foot 全样本：clearance 表，foot_clearance_swing[f] 为 list[Tensor]
    foot_clearance_swing = [[] for _ in range(n_feet)]
    # ★ 新增：被摆动门控排除掉的「慢脚」样本单独留存。
    #   脚在世界上几乎不动 = 要么是支撑相（正常），要么是顶在台阶立面上卡住（异常）。
    #   旧版把这些直接丢掉，所以「卡住」在旧指标里完全不可见。
    foot_clearance_slow = [[] for _ in range(n_feet)]
    # ★ 新增：顶在竖直面上的接触计数（水平力 ≫ 竖直力）
    riser_contact_hits = [0] * n_feet
    riser_contact_n = 0
    # ★ 新增：整机卡住步数（有前进命令，但基座前向速度掉到阈值以下）
    stall_steps = 0
    # ★ 新增：基座前向速度全样本（用来判断"卡住"是真卡住，还是量具有问题）
    base_vx_all = []
    # ★ 新增：实际下发的速度指令幅度。探针把 ranges 钉成 (cmd_vx, cmd_vx)，但如果
    #   有课程项（lin_vel_cmd_levels）在运行期改 ranges.lin_vel_x，钉的值会被冲掉 ——
    #   这时 base_vx≈0 就是"命令根本不是 cmd_vx"造成的假象，不是策略卡住。
    cmd_norm_all = []
    # ★ 新增：逐 env 的平均前向速度（看是全体不动，还是只有楼梯列不动）
    base_vx_env_sum = torch.zeros(N, device=device)
    base_vx_env_cnt = torch.zeros(N, device=device)
    # ★ 新增：接触传感器自检 —— 逐脚的「有接触样本占比」与「接触时平均 fz」。
    #   站立时单脚 fz 应 ≈ mg/4 ≈ 20~30N。若这里读到 ~0，说明 c_foot_ids 取错了 body，
    #   下面 riser_contact 的 0% 就是假阴性而不是真结论。
    fz_contact_hits = [0] * n_feet
    fz_sum = [0.0] * n_feet
    # base_pitch 全样本（每步每 env）
    pitch_all = []
    # base_height：min / sum / cnt（同 measure_body_height.py 口径）
    bh_min = torch.full((N,), float("inf"), device=device)
    bh_sum = torch.zeros(N, device=device)
    bh_cnt = torch.zeros(N, device=device)

    # 接触传感器：脚 body 的 id（按名字取，需与 robot 的 foot_ids 顺序一致，否则逐脚统计会串位）
    contact_sensor = env.unwrapped.scene["contact_forces"]
    c_foot_ids, c_foot_names = contact_sensor.find_bodies(".*_foot", preserve_order=True)
    if list(c_foot_names) != list(foot_names):
        raise RuntimeError(f"接触传感器脚顺序不一致: {c_foot_names} vs {foot_names}，逐脚统计不可用")

    obs = env.get_observations()
    # ★ 世界系位移的起点（最鲁棒的"到底动没动"指标，不受 body 系朝向/旋转影响）。
    #   仅在 disable_curriculum 下有意义 —— 否则 terrain_levels_vel 会把 env 传送到别的难度行。
    start_xy = robot.data.root_pos_w[:, :2].clone()
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _, _, _ = env.step(actions)

        # 脚下局部地形高度
        terrain_z = _terrain_z()  # (N,)

        # 脚净空：脚体中心 Z − 脚下地形 Z（与 foot_min_clearance_reward 同口径）
        foot_z = robot.data.body_pos_w[:, foot_ids, 2]  # (N, n_feet)
        foot_clearance = foot_z - terrain_z.unsqueeze(1)  # (N, n_feet)
        # 摆动相门控：脚水平速度 > swing_vel_thresh
        foot_vxy = torch.norm(robot.data.body_lin_vel_w[:, foot_ids, :2], dim=2)  # (N, n_feet)
        swing = foot_vxy > args_cli.swing_vel_thresh
        fin = ~torch.isnan(foot_clearance)

        for f in range(n_feet):
            valid = swing[:, f] & fin[:, f]
            if valid.any():
                foot_clearance_swing[f].append(foot_clearance[:, f][valid].cpu())
            slow = (~swing[:, f]) & fin[:, f]
            if slow.any():
                foot_clearance_slow[f].append(foot_clearance[:, f][slow].cpu())

        # ★ 顶立面检测：接触力水平分量 ≫ 竖直分量（阈值 5 沿用 feet_stumble 原口径），
        #   且要求有真实接触（fz > 1N），否则「脚在空中、力≈0」会被误判为比值无穷大。
        forces = contact_sensor.data.net_forces_w[:, c_foot_ids, :]  # (N, n_feet, 3)
        fz = torch.abs(forces[..., 2])
        fxy = torch.norm(forces[..., :2], dim=-1)
        riser = (fxy > 5.0 * fz) & (fz > 1.0)
        for f in range(n_feet):
            riser_contact_hits[f] += int(riser[:, f].sum())
            fz_contact_hits[f] += int((fz[:, f] > 1.0).sum())
            fz_sum[f] += float(fz[:, f].sum())
        riser_contact_n += N

        # ★ 整机卡住：有前进命令，但基座前向速度低于阈值
        cmd = env.unwrapped.command_manager.get_command("base_velocity")  # (N, 3)
        cmd_norm = torch.norm(cmd[:, :2], dim=1)
        base_vx = robot.data.root_lin_vel_b[:, 0]
        stall_steps += int(((cmd_norm > 0.1) & (base_vx < args_cli.stall_vel_thresh)).sum())
        base_vx_all.append(base_vx.detach().cpu())
        cmd_norm_all.append(cmd_norm.detach().cpu())
        base_vx_env_sum += base_vx.detach()
        base_vx_env_cnt += 1.0

        # 机身俯仰（nose-down 为负）
        _, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)  # each (N,)
        pitch_all.append(pitch.cpu())

        # 机身离地净空
        base_z = robot.data.root_pos_w[:, 2]
        bh = base_z - terrain_z
        valid_bh = ~torch.isnan(bh)
        bh_min = torch.where(valid_bh & (bh < bh_min), bh, bh_min)
        bh_sum = torch.where(valid_bh, bh_sum + bh, bh_sum)
        bh_cnt = bh_cnt + valid_bh.float()

    # --- 汇总 ---
    pitch_all = torch.cat(pitch_all) if pitch_all else torch.zeros(0)
    bh_mean_per_env = bh_sum / bh_cnt.clamp(min=1.0)

    def _stats(t: torch.Tensor, name: str, unit: str = ""):
        valid = t[~torch.isnan(t) & torch.isfinite(t)]
        if valid.numel() == 0:
            print(f"[{name}] no valid samples")
            return
        # ★ p95/p99 是量"摆动脚能抬多高"时真正要看的数：摆动相样本里混着刚离地和
        #   即将触地两端（低），median 会被拉到很低；而单看 max 又容易被一次摔倒中
        #   乱飞的脚污染。上尾分位数才是"峰值抬脚高度"的稳健估计。
        q = lambda p: torch.quantile(valid, p).item()
        print(
            f"[{name}] n={valid.numel()}  mean={valid.mean().item():.4f}{unit}  "
            f"median={valid.median().item():.4f}  p5={q(0.05):.4f}  "
            f"p95={q(0.95):.4f}  p99={q(0.99):.4f}  "
            f"min={valid.min().item():.4f}  max={valid.max().item():.4f}"
        )

    print("\n================ 摆动脚离地净空 / 机身俯仰 / 机身离地 验证结果 ================")
    print(f"num_envs={N}  num_steps={args_cli.num_steps}  cmd_vx={args_cli.cmd_vx}m/s  "
          f"step_height={args_cli.step_height}m  dt={dt:.4f}s  "
          f"swing_thresh={args_cli.swing_vel_thresh}m/s  stall_thresh={args_cli.stall_vel_thresh}m/s")
    print("\n--- foot_clearance = 脚体中心 Z − 脚下地形 Z（摆动相，脚水平速度 > %.1f m/s）---" % args_cli.swing_vel_thresh)
    for f in range(n_feet):
        tag = "前脚" if foot_names[f] in ("FL_foot", "FR_foot") else "后脚"
        samples = torch.cat(foot_clearance_swing[f]) if foot_clearance_swing[f] else torch.zeros(0)
        _stats(samples, f"foot_clearance[{foot_names[f]}]({tag})", " m")
    print("\n--- base_pitch = 机身绕 Y 轴俯仰角（nose-down 为负）---")
    _stats(pitch_all, "base_pitch 全样本", " rad")
    print("  注：min(pitch) 越负 = 越深点头（前脚撞台阶的因果后果）。")
    print("\n--- base_height = 机身中心 Z − 脚下地形 Z（离地净空，标称站高约 0.30m）---")
    _stats(bh_min, "base_height 每 env 最小值（身体着地前最低点）", " m")
    _stats(bh_mean_per_env, "base_height 每 env 均值", " m")

    # ============ ★ 新增：卡住诊断三件套 ============
    # 旧的 foot_clearance 统计带 swing 门控（脚速 > 0.3），「顶在立面上卡住」的样本
    # 恰好全部落在门控外被丢弃 —— 所以旧指标看不见这个失败模式。下面三项专门测它。
    print("\n================ ★ 卡住诊断（新增）================")

    print("\n--- [1] blocked 净空：未过摆动门控的「慢脚」样本（脚在世界系几乎不动）---")
    print("    前脚这一项的 p50 若明显低于摆动相 p50 → 大量慢脚停在低位 = 卡在立面上")
    for f in range(n_feet):
        tag = "前脚" if foot_names[f] in ("FL_foot", "FR_foot") else "后脚"
        samples = torch.cat(foot_clearance_slow[f]) if foot_clearance_slow[f] else torch.zeros(0)
        _stats(samples, f"foot_clearance_slow[{foot_names[f]}]({tag})", " m")

    print("\n--- [2] riser_contact：脚顶在竖直面上的样本占比（水平力 > 5×竖直力 且 fz > 1N）---")
    print("    这是「前腿撞在台阶立面上」的直接信号；越低越好")
    for f in range(n_feet):
        tag = "前脚" if foot_names[f] in ("FL_foot", "FR_foot") else "后脚"
        ratio = riser_contact_hits[f] / max(riser_contact_n, 1)
        print(f"[riser_contact[{foot_names[f]}]({tag})] {ratio * 100:6.2f}%  "
              f"({riser_contact_hits[f]}/{riser_contact_n})")

    print("\n--- [2b] 接触传感器自检：站立时单脚 fz 应 ≈ mg/4 ≈ 20~30N；若读到 ~0 说明取错了 body ---")
    for f in range(n_feet):
        contact_frac = fz_contact_hits[f] / max(riser_contact_n, 1)
        mean_fz = fz_sum[f] / max(riser_contact_n, 1)
        print(f"[contact_sanity[{foot_names[f]}]] 有接触(fz>1N)占比={contact_frac * 100:6.2f}%  "
              f"全样本平均 fz={mean_fz:7.3f}N")

    print(f"\n--- [3] stall：有前进命令但基座前向速度 < {args_cli.stall_vel_thresh} m/s 的步占比 ---")
    print(f"[stall] {stall_steps / max(N * args_cli.num_steps, 1) * 100:6.2f}%  "
          f"({stall_steps}/{N * args_cli.num_steps})")
    print("    这是「狗子卡一下」的直接量化；越低越好")
    print("    配套的基座前向速度（body 系 x）分布 —— 若 p50 远低于 cmd_vx，说明「卡住」是真实存在的，"
          "不是量具问题：")
    if base_vx_all:
        _stats(torch.cat(base_vx_all), "base_vx（body 系前向速度）", " m/s")
    if cmd_norm_all:
        _stats(torch.cat(cmd_norm_all), "实际下发的 |cmd_xy|（期望 = cmd_vx）", " m/s")
        print("    ↑ 若这一项不等于 cmd_vx，说明有课程项在运行期覆写了 ranges.lin_vel_x，"
              "本次的 base_vx 数字不可用于判断策略是否卡住。")
    if base_vx_env_cnt.min() > 0:
        per_env_vx = base_vx_env_sum / base_vx_env_cnt
        _stats(per_env_vx, "逐 env 平均 base_vx", " m/s")
        moving = int((per_env_vx > 0.5 * args_cli.cmd_vx).sum())
        print(f"    ↑ 平均速度 > {0.5 * args_cli.cmd_vx:.2f} m/s 的 env 数：{moving}/{N}"
              f"（若为 0，则不是「某些地形上卡住」，而是全体都不动）")

    # ★ 世界系位移（run 结束位置 − 起点）。cmd_vx=1.0、150 步 ≈ 3s → 走起来应 ≈ 3m 量级。
    #   仅当 --disable_curriculum 时可信（否则课程会传送 env）。
    disp = torch.norm(robot.data.root_pos_w[:, :2] - start_xy, dim=1)
    _stats(disp, "世界系位移（起点→终点）", " m")
    expected = args_cli.cmd_vx * args_cli.num_steps * dt
    print(f"    ↑ 若全程跟上命令应为 ≈ {expected:.2f} m。远低于此 = 没在走。"
          f"（需配合 --disable_curriculum 才可信）")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
