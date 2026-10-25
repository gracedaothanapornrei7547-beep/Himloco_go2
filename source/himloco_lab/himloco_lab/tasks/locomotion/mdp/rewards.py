from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from himloco_lab.envs import HimlocoManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def roll_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """专门惩罚横滚角（Roll）偏离，通过惩罚投影重力在 Y 轴的分量 gy² 实现；不惩罚俯仰角（gx），保留楼梯倾斜裕度。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def pitch_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """专门惩罚机头俯仰角速度（Pitch Rate），抑制撞击台阶瞬间的低头前栽动量。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 1])


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.linalg.norm(diff, dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def hip_deviation_adaptive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_hip_joint"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize hip abduction angle deviation from nominal, gated by turning and lateral commands.
    Enforces straight sagittal alignment during straight trot while permitting necessary abduction during turning.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cost = torch.sum(torch.square(diff), dim=1)
    cmd = env.command_manager.get_command(command_name)
    gate = torch.clamp(1.0 - torch.abs(cmd[:, 1]) - 0.5 * torch.abs(cmd[:, 2]), min=0.0, max=1.0)
    return cost * gate


def smoothness(env: HimlocoManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action*2 + env.pre_pre_action), dim=1)

"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 5 * forces_z, dim=1).float()
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.2
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air upon touchdown.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum(torch.clamp(last_air_time - threshold, min=0.0) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
) -> torch.Tensor:
    """Penalize the swinging feet for not clearing a specified height off the ground.

    ★ 已对齐官方 ``_reward_foot_clearance``（legged_robot.py:1146-1156）：逐行等价
      （body 系脚位/脚速、``square(z - target) * lateral_vel``、按脚求和）。
      此处原先多一道 ``command_name`` 门控（``reward *= ‖cmd‖ > 0.1``），官方没有——
      那会让零指令的 env（``rel_high_vel_envs`` 占 20%）整项恒为 0，属偏离，已删除。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(env.num_envs, -1)
    reward = torch.sum(foot_z_target_error * foot_leteral_vel, dim=1)
    return reward


def feet_min_height_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = -0.305,
    margin: float = 0.035,
) -> torch.Tensor:
    """Penalize swinging feet for not clearing a minimum safety height in the body frame.

    One-sided linear barrier variant of feet_height_body:
    - Only penalizes feet swinging BELOW target_height (nominal ground is ~ -0.34m in body frame,
      target_height = -0.305m corresponds to 3.5cm safe ground clearance).
    - Lifting higher than target_height incurs ZERO penalty, allowing high stair step traversal.
    - Uses normalized linear hinge error (target - z) / margin to ensure healthy non-vanishing gradient
      without the quadratic compression that drowns small clearances.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    # Normalized linear hinge error: ranges [0, 1] when dragging on ground
    foot_z_below = torch.clamp(target_height - footpos_in_body_frame[:, :, 2], min=0.0) / margin
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(env.num_envs, -1)
    reward = torch.sum(foot_z_below * foot_leteral_vel, dim=1)
    return reward


def feet_min_height_body_pitch_adaptive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ground_ref: float = -0.340,
    min_margin: float = 0.035,
    max_margin: float = 0.100,
    pitch_threshold_deg: float = 4.0,
    pitch_ramp_deg: float = 6.0,
) -> torch.Tensor:
    """机身爬楼姿态自适应安全净空门限（Pitch-Adaptive One-Sided Clearance）。

    - 平时平地（Pitch > -4°）：margin 锁定在 3.5cm，保持 0.70cm 极致无震；
    - 一旦入阶爬坡（Pitch < -10°）：margin 自动平滑放大至 10.0cm，自发转入高提膝大步幅，
      彻底消除第 2 阶及后续台阶的撞立面卡死与二次抬腿。
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 1. 提取机身俯仰角 Pitch（单位：度）
    _, pitch_rad, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch_deg = torch.rad2deg(pitch_rad)

    # 2. 计算动态自适应安全门限 current_margin（后仰爬坡时平滑激活）
    # 机头仰起爬坡时 pitch < 0（-pitch_deg > 0）
    pitch_tilt = torch.clamp((-pitch_deg - pitch_threshold_deg) / pitch_ramp_deg, min=0.0, max=1.0)
    current_margin = min_margin + (max_margin - min_margin) * pitch_tilt.unsqueeze(1)  # (num_envs, 1)

    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )

    # 3. 归一化单侧线性 Hinge 损失
    target_height = ground_ref + current_margin  # 平地 -0.305m，楼梯 -0.240m
    foot_z_below = torch.clamp(target_height - footpos_in_body_frame[:, :, 2], min=0.0) / current_margin
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(env.num_envs, -1)
    reward = torch.sum(foot_z_below * foot_lateral_vel, dim=1)
    return reward


def _update_stumble_reflex_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_ratio: float = 3.0,
    min_force_xy: float = 12.0,
    max_force_z: float = 25.0,
    damping_duration_s: float = 0.10,
    cooldown_duration_s: float = 0.50,
    timeout_duration_s: float = 0.60,
):
    """Central state tracker for tactile reflex: one-shot 100ms compliant pause and milestone tracking."""
    step_count = getattr(env, "common_step_counter", 0)
    if getattr(env, "_last_stumble_update_step", -1) == step_count:
        return

    env._last_stumble_update_step = step_count
    step_dt = getattr(env, "step_dt", 0.02)
    num_feet = len(sensor_cfg.body_ids)

    # Initialize buffers if not present
    if not hasattr(env, "stumble_damping_timer") or env.stumble_damping_timer is None:
        env.stumble_damping_timer = torch.zeros(env.num_envs, device=env.device)
        env.stumble_cooldown_timer = torch.zeros(env.num_envs, device=env.device)
        env.stumbling_foot_active = torch.zeros((env.num_envs, num_feet), dtype=torch.bool, device=env.device)
        env.stumbling_foot_initial_z = torch.zeros((env.num_envs, num_feet), device=env.device)
        env.stumbling_foot_timeout = torch.zeros((env.num_envs, num_feet), device=env.device)

    # Reset buffers for newly reset environments
    if hasattr(env, "reset_buf") and env.reset_buf is not None and env.reset_buf.any():
        env.stumble_damping_timer[env.reset_buf] = 0.0
        env.stumble_cooldown_timer[env.reset_buf] = 0.0
        env.stumbling_foot_active[env.reset_buf] = False
        env.stumbling_foot_initial_z[env.reset_buf] = 0.0
        env.stumbling_foot_timeout[env.reset_buf] = 0.0

    # Decrement timers
    env.stumble_damping_timer = torch.clamp(env.stumble_damping_timer - step_dt, min=0.0)
    env.stumble_cooldown_timer = torch.clamp(env.stumble_cooldown_timer - step_dt, min=0.0)
    env.stumbling_foot_timeout = torch.clamp(env.stumbling_foot_timeout - step_dt, min=0.0)

    # Expire feet that timed out without stepping up
    timed_out_feet = env.stumbling_foot_timeout <= 0.0
    env.stumbling_foot_active[timed_out_feet] = False

    # Detect current collision with vertical riser
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    is_riser = (forces_xy > min_force_xy) & (forces_xy > force_ratio * forces_z) & (forces_z < max_force_z)

    # Check if a new collision can trigger the one-shot damping window
    has_collision = torch.any(is_riser, dim=1)
    can_trigger = has_collision & (env.stumble_cooldown_timer <= 0.0)

    env.stumble_damping_timer[can_trigger] = damping_duration_s
    env.stumble_cooldown_timer[can_trigger] = cooldown_duration_s

    # Track newly stumbling feet
    robot: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_tactile_foot_body_ids") or env._tactile_foot_body_ids is None:
        env._tactile_foot_body_ids, _ = robot.find_bodies(".*_foot", preserve_order=True)

    new_stumbling = is_riser & can_trigger.unsqueeze(1)
    if new_stumbling.any():
        env.stumbling_foot_active[new_stumbling] = True
        feet_pos_z = robot.data.body_pos_w[:, env._tactile_foot_body_ids, 2]
        env.stumbling_foot_initial_z[new_stumbling] = feet_pos_z[new_stumbling]
        env.stumbling_foot_timeout[new_stumbling] = timeout_duration_s


def feet_stumble_upward_escape_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_ratio: float = 3.0,
    min_force_xy: float = 12.0,
    max_force_z: float = 25.0,
    max_vz: float = 2.0,
) -> torch.Tensor:
    """Reward upward foot velocity when colliding with a vertical obstacle surface (riser).

    Foot-Trapping Reflex (Stumbling Corrective Reaction):
    Active when a swinging foot hits a vertical surface (riser) or is actively in its
    reflex lifting phase before clearing the step tread.
    Provides positive reward for vertical lift velocity (v_z > 0), inducing immediate leg flexion
    and knee pulling to clear the step.
    """
    _update_stumble_reflex_state(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        force_ratio=force_ratio,
        min_force_xy=min_force_xy,
        max_force_z=max_force_z,
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    is_riser_collision = (forces_xy > min_force_xy) & (forces_xy > force_ratio * forces_z) & (forces_z < max_force_z)

    robot: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_tactile_foot_body_ids") or env._tactile_foot_body_ids is None:
        env._tactile_foot_body_ids, _ = robot.find_bodies(".*_foot", preserve_order=True)

    foot_z = robot.data.body_pos_w[:, env._tactile_foot_body_ids, 2]
    foot_vz = robot.data.body_lin_vel_w[:, env._tactile_foot_body_ids, 2]
    upward_vz = torch.clamp(foot_vz, min=0.0, max=max_vz)

    # Foot is lifting if either actively colliding OR registered as stumbling and still under +0.10m
    is_lifting_reflex = is_riser_collision | (
        env.stumbling_foot_active & (foot_z < env.stumbling_foot_initial_z + 0.10)
    )

    reward = torch.sum(upward_vz * is_lifting_reflex.float(), dim=1)
    return reward


def feet_swing_height_ceiling(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ground_ref: float = -0.340,
    max_clearance: float = 0.038,
    pitch_flat_deg: float = 4.0,
    sensor_cfg: SceneEntityCfg | None = None,
    force_ratio: float = 3.0,
    min_force_xy: float = 12.0,
    max_force_z: float = 25.0,
) -> torch.Tensor:
    """Penalize swinging feet for lifting excessively high on flat terrain.

    Targets the 'pre-lifting cheat' (e.g. permanently lifting front legs to 8.5~9cm on flat ground):
    - Active only on flat ground where |pitch| < pitch_flat_deg (smoothly ramps down to 0 on stairs/slopes).
    - If a foot is in a stumble/riser collision (encountering an obstacle), it is exempted from this ceiling penalty.
    - Imposes a quadratic barrier whenever a swinging foot clears more than max_clearance (3.8cm):
        penalty = ((z_foot - (ground_ref + max_clearance)) / 0.035)^2 * v_lateral
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 1. Flat ground gating based on body pitch & roll (only flat ground enforces ceiling)
    roll_rad, pitch_rad, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch_deg = torch.rad2deg(pitch_rad)
    roll_deg = torch.rad2deg(roll_rad)
    flat_gate = torch.clamp(1.0 - torch.abs(pitch_deg) / pitch_flat_deg, min=0.0, max=1.0) * \
                torch.clamp(1.0 - torch.abs(roll_deg) / pitch_flat_deg, min=0.0, max=1.0)
    flat_gate = flat_gate.unsqueeze(1)  # (num_envs, 1)

    # 2. Foot positions and velocities in body frame
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )

    # 3. Compute excess height above ceiling (e.g. > 3.8cm clearance)
    ceiling_z = ground_ref + max_clearance
    excess_height = torch.clamp(footpos_in_body_frame[:, :, 2] - ceiling_z, min=0.0)

    # 4. Exemption mask: if foot is actively colliding or in stumble reflex, exempt it
    is_colliding = torch.zeros(env.num_envs, len(asset_cfg.body_ids), dtype=torch.bool, device=env.device)
    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        is_riser = (forces_xy > min_force_xy) & (forces_xy > force_ratio * forces_z) & (forces_z < max_force_z)
        is_colliding = is_riser
        if hasattr(env, "stumbling_foot_active") and env.stumbling_foot_active is not None:
            is_colliding = is_colliding | env.stumbling_foot_active

    # 5. Quadratic penalty on swinging feet
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2))
    penalty = torch.square(excess_height / 0.035) * foot_lateral_vel
    penalty = penalty * (~is_colliding).float() * flat_gate
    return torch.sum(penalty, dim=1)


def track_lin_vel_xy_one_shot_compliant(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_ratio: float = 3.0,
    min_force_xy: float = 12.0,
    max_force_z: float = 25.0,
    damping_duration_s: float = 0.10,
    cooldown_duration_s: float = 0.50,
    compliant_cmd_vx: float = 0.20,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands with one-shot 100ms compliance on obstacle impact.

    Standard velocity tracking (vx_cmd, vy_cmd) using exponential kernel.
    When any swinging foot impacts a vertical stair riser (F_xy > min_force_xy, F_xy > force_ratio * F_z, F_z < max_force_z):
    - Triggers a STRICT 100ms (5 steps) one-shot compliance window.
    - During the 100ms window, the forward tracking target is compliant (compliant_cmd_vx, default 0.2 m/s):
      allows body to buffer / brake ('机身停一下') so the leg can flex and clear without body overturn.
    - Once the 100ms window expires, target forward velocity IMMEDIATELY snaps back to full speed (0.8 m/s).
    - If the robot stays stopped in front of the stair, it receives severe velocity tracking penalties!
    - Cooldown (0.50s) prevents re-triggering while standing against the wall.
    """
    _update_stumble_reflex_state(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        force_ratio=force_ratio,
        min_force_xy=min_force_xy,
        max_force_z=max_force_z,
        damping_duration_s=damping_duration_s,
        cooldown_duration_s=cooldown_duration_s,
    )

    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_vel = env.command_manager.get_command(command_name)[:, :2].clone()

    in_damping_window = env.stumble_damping_timer > 0.0
    effective_vx = torch.where(
        in_damping_window,
        torch.full_like(cmd_vel[:, 0], compliant_cmd_vx),
        cmd_vel[:, 0],
    )
    cmd_vel[:, 0] = effective_vx

    lin_vel_error = torch.sum(
        torch.square(cmd_vel - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / (std**2))


# Backward compatibility alias
track_lin_vel_xy_collision_compliant = track_lin_vel_xy_one_shot_compliant


def feet_stumble_step_up_milestone(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_step_height: float = 0.065,
    min_touchdown_force_z: float = 25.0,
    touchdown_force_ratio: float = 1.5,
) -> torch.Tensor:
    """Milestone reward for successfully clearing a stair riser and landing on the tread.

    Triggered when a foot that collided with a vertical obstacle:
    1. Lifts up by at least min_step_height (e.g. 6.5cm) above the collision height.
    2. Establishes solid vertical contact on the tread (F_z > min_touchdown_force_z and F_z > 1.5 * F_xy).

    Delivers a one-shot bonus (+4.0) when the foot successfully mounts the tread,
    strongly reinforcing the complete reflex chain: [touch -> brake & lift -> plant on tread].
    """
    if not hasattr(env, "stumbling_foot_active") or env.stumbling_foot_active is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)

    robot: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_tactile_foot_body_ids") or env._tactile_foot_body_ids is None:
        env._tactile_foot_body_ids, _ = robot.find_bodies(".*_foot", preserve_order=True)

    current_foot_z = robot.data.body_pos_w[:, env._tactile_foot_body_ids, 2]
    delta_z = current_foot_z - env.stumbling_foot_initial_z

    # Conditions for successful touchdown on new tread
    height_cleared = delta_z >= min_step_height
    solid_touchdown = (forces_z > min_touchdown_force_z) & (forces_z > touchdown_force_ratio * forces_xy)
    success = env.stumbling_foot_active & height_cleared & solid_touchdown

    reward = torch.sum(success.float(), dim=1)

    # Deactivate successfully completed feet so milestone is awarded exactly once per step-up
    if success.any():
        env.stumbling_foot_active[success] = False

    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.

    When ``sensor_cfg`` points at a downward ray-caster over the base, clearance is
    measured relative to the local terrain height under the base instead of the world
    z=0 plane. This is required for stairs: otherwise a foot planted on a raised step
    reads as "lifted too high" and the reward actively fights the swing.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        hits_z = sensor.data.ray_hits_w[..., 2]  # (num_envs, num_rays)
        # ignore misses (NaN/inf/large sentinel) when averaging the local ground height
        hits_z = torch.where(torch.isfinite(hits_z) & (torch.abs(hits_z) < 1e6), hits_z, torch.nan)
        terrain_z = torch.nanmean(hits_z, dim=1)  # (num_envs,) local ground under base
        terrain_z = torch.nan_to_num(terrain_z, nan=0.0)
        foot_z = foot_z - terrain_z.unsqueeze(1)

    foot_z_target_error = torch.square(foot_z - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_min_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward the swinging feet for clearing at least ``target_height`` off the ground.

    One-sided variant of :func:`foot_clearance_reward`: it only penalizes a foot that
    swings *below* ``target_height`` (insufficient clearance) and is flat above it, so
    lifting higher than the target is free rather than penalized. This is the right shape
    for obstacle/stair traversal, where the foot should clear a minimum height rather than
    hover at a fixed one. When ``sensor_cfg`` is provided, clearance is measured relative
    to the local terrain height under the base (same as :func:`foot_clearance_reward`).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        hits_z = sensor.data.ray_hits_w[..., 2]
        hits_z = torch.where(torch.isfinite(hits_z) & (torch.abs(hits_z) < 1e6), hits_z, torch.nan)
        terrain_z = torch.nanmean(hits_z, dim=1)
        terrain_z = torch.nan_to_num(terrain_z, nan=0.0)
        foot_height = foot_height - terrain_z.unsqueeze(1)

    # one-sided error: positive only where the foot is below target_height
    foot_z_below = torch.square(torch.clamp(target_height - foot_height, min=0.0))
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_below * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_forward_reach(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """奖励摆动足在机身前向（Body x 轴）的舒展探步速度，引导四足在空中主动向前探步翻越障碍，而非滑稽高抬腿。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 摆动相：垂直接触力 < 1.0 N
    is_swing = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] < 1.0  # (num_envs, num_feet)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w.unsqueeze(1)
    foot_vx_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
        foot_vx_b[:, i] = footvel_in_body_frame[:, 0]

    # 仅在摆动相奖励向前探出的正向速度（截断至 1.5 m/s，防乱踢）
    swing_reach = torch.clamp(foot_vx_b, min=0.0, max=1.5) * is_swing.float()
    cmd_x = env.command_manager.get_command("base_velocity")[:, 0]
    return torch.sum(swing_reach, dim=1) * (cmd_x > 0.1).float()


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    
def base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        
        # Replace invalid values (NaN, Inf, too large) with NaN for masked mean
        valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6)
        # 对齐 Project-Robo legged_robot.py:1063 的底盘采样点均值 (torch.mean)：
        ray_hits_masked = torch.where(valid_mask, ray_hits, torch.nan)
        adjusted_heights = torch.nanmean(ray_hits_masked, dim=1)
        
        all_invalid_mask = torch.isnan(adjusted_heights)
        if all_invalid_mask.any():
            invalid_env_count = all_invalid_mask.sum().item()
            total_invalid_points = (~valid_mask).sum().item()
            print(f"[WARNING] base_height - {invalid_env_count} envs with all invalid points, total invalid: {total_invalid_points}/{ray_hits.numel()} ({total_invalid_points/ray_hits.numel():.2%})")
            adjusted_heights[all_invalid_mask] = asset.data.root_link_pos_w[all_invalid_mask, 2] - target_height
        
        adjusted_target_height = target_height + adjusted_heights
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    # ★ 已对齐官方 _reward_base_height（legged_robot.py:1141-1144）：官方就是
    #   square(base_height - base_height_target)，没有任何 gravity 门控
    #   （projected_gravity 在官方全文件只用于 _reward_orientation:1131）。
    #   此处原先多乘一道 clamp(-g_b_z, 0, 0.7)/0.7，会让机身倾斜时高度惩罚被削弱，属偏离，已删。
    #   代数上与官方等价：官方 square(mean_i(root_z - terrain_z_i) - 0.30)，
    #   本函数 square(root_z - (0.30 + mean(terrain_z)))，root_z 对各采样点相同，两式恒等。
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return reward


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward
    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def trot_contact_consistency(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    sync_weight: float = 1.0,
    bound_penalty_weight: float = 1.5,
    pace_penalty_weight: float = 0.5,
) -> torch.Tensor:
    """Reward diagonal trot contact consistency and heavily penalize bounding / pacing.

    Assumes feet order in sensor_cfg is [FL, FR, RL, RR].
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if len(body_ids) != 4:
        raise ValueError(f"trot_contact_consistency expects exactly 4 feet, got {len(body_ids)}")

    # Contact indicators (1 for stance, 0 for air)
    c = (contact_sensor.data.current_contact_time[:, body_ids] > 0).float()
    c_fl = c[:, 0]
    c_fr = c[:, 1]
    c_rl = c[:, 2]
    c_rr = c[:, 3]

    # 1) Diagonal synchrony (FL with RR, FR with RL)
    diag_1 = 1.0 - torch.abs(c_fl - c_rr)
    diag_2 = 1.0 - torch.abs(c_fr - c_rl)
    s_diag = 0.5 * (diag_1 + diag_2)

    # 2) Anti-Bound penalty (front pair FL+FR or rear pair RL+RR touching simultaneously)
    p_bound_front = c_fl * c_fr
    p_bound_rear = c_rl * c_rr
    p_bound = p_bound_front + p_bound_rear

    # 3) Anti-Pace penalty (left pair FL+RL or right pair FR+RR touching simultaneously)
    p_pace = (c_fl * c_rl) + (c_fr * c_rr)

    reward = sync_weight * s_diag - bound_penalty_weight * p_bound - pace_penalty_weight * p_pace

    # Apply forward command mask (only active when commanded to move)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward = torch.where(cmd_norm > command_threshold, reward, torch.zeros_like(reward))

    # Mask out when robot is tipped over
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 1.0)
    return reward


def _bernstein_torch(n: int, k: int, t: torch.Tensor) -> torch.Tensor:
    """Bernstein basis B_k^n(t) for tensor t in [0, 1]."""
    coeff = float(math.comb(n, k))
    return coeff * (1.0 - t) ** (n - k) * t**k


def _bezier_curve_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier curve points for batched parameter t."""
    n = control_points.shape[0] - 1
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n + 1):
        out = out + _bernstein_torch(n, k, t).unsqueeze(-1) * control_points[k]
    return out


def _bezier_curve_derivative_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate d/dt of Bezier curve points for batched parameter t."""
    n = control_points.shape[0] - 1
    delta_ctrl = n * (control_points[1:] - control_points[:-1])
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n):
        out = out + _bernstein_torch(n - 1, k, t).unsqueeze(-1) * delta_ctrl[k]
    return out


def phase_foot_trajectory_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.12,
    command_threshold: float = 0.1,
    cycle_time: float = 0.425,
    phase_offsets: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0),
    gait_span: float = 0.0,
    gait_psi: float = 0.05,
    gait_delta: float = 0.02,
    x_offset: float = 0.0,
    stance_span: float = 0.0,
    stand_ref_z_offset: float = 0.0,
    velocity_weight: float = 0.0,
) -> torch.Tensor:
    """Track phase foot trajectory in body frame with exponential kernel (Bezier Trot gait engine)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_feet = len(body_ids)

    if num_feet == 0:
        return torch.zeros(env.num_envs, device=env.device)
    if len(phase_offsets) != num_feet:
        raise ValueError(f"phase_offsets length ({len(phase_offsets)}) must match tracked feet ({num_feet}).")

    # Build and cache base-fixed stand references from the first call.
    if (not hasattr(env, "phase_foot_ref_body")) or (env.phase_foot_ref_body.shape[1] != num_feet):
        rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
        foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
        for i in range(num_feet):
            foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        ref = foot_pos_b[0].detach().clone()
        ref[:, 2] += stand_ref_z_offset
        env.phase_foot_ref_body = ref.unsqueeze(0)

    stand_ref_body = env.phase_foot_ref_body.to(env.device).expand(env.num_envs, -1, -1)

    # Build phase S in [0, 2).
    phase_time = env.episode_length_buf.float() * env.step_dt
    phase_offsets_t = torch.tensor(phase_offsets, device=env.device, dtype=phase_time.dtype).unsqueeze(0)
    S = torch.remainder((2.0 * phase_time / max(cycle_time, 1e-6)).unsqueeze(1) + phase_offsets_t, 2.0)

    # Piecewise trajectory in local (q, z).
    tau = float(gait_span)
    psi = float(gait_psi)
    delta = float(gait_delta)
    stance_span = float(stance_span)
    stance_span = min(max(stance_span, 0.0), 2.0 - 1e-6)

    q = torch.zeros_like(S)
    z = torch.zeros_like(S)
    dq_dS = torch.zeros_like(S)
    dz_dS = torch.zeros_like(S)

    if stance_span > 1e-6:
        stance_mask = S < stance_span
        if stance_mask.any():
            s_stance = S / stance_span
            q_stance = tau * (1.0 - 2.0 * s_stance)
            z_stance = torch.full_like(S, delta)
            dq_dS_stance = torch.full_like(S, -2.0 * tau / stance_span)
            dz_dS_stance = torch.zeros_like(S)

            q = torch.where(stance_mask, q_stance, q)
            z = torch.where(stance_mask, z_stance, z)
            dq_dS = torch.where(stance_mask, dq_dS_stance, dq_dS)
            dz_dS = torch.where(stance_mask, dz_dS_stance, dz_dS)

        swing_mask = ~stance_mask
    else:
        swing_mask = torch.ones_like(S, dtype=torch.bool)

    if swing_mask.any():
        t_bezier = torch.clamp((S - stance_span) / (2.0 - stance_span), 0.0, 1.0)
        ctrl = torch.tensor(
            [
                [-tau, 0.0],
                [-0.95 * tau, 0.80 * psi],
                [-0.55 * tau, 1.00 * psi],
                [0.55 * tau, 1.00 * psi],
                [0.95 * tau, 0.80 * psi],
                [tau, 0.0],
            ],
            device=env.device,
            dtype=S.dtype,
        )
        qz_swing = _bezier_curve_torch(ctrl, t_bezier)
        dqz_dt = _bezier_curve_derivative_torch(ctrl, t_bezier)
        dt_dS = 1.0 / (2.0 - stance_span)

        q = torch.where(swing_mask, qz_swing[..., 0], q)
        z = torch.where(swing_mask, qz_swing[..., 1] + delta, z)
        dq_dS = torch.where(swing_mask, dqz_dt[..., 0] * dt_dS, dq_dS)
        dz_dS = torch.where(swing_mask, dqz_dt[..., 1] * dt_dS, dz_dS)

    dS_dt = 2.0 / max(cycle_time, 1e-6)
    dq_dt = dq_dS * dS_dt
    dz_dt = dz_dS * dS_dt

    ref_pos_b = stand_ref_body + torch.stack(
        [q + float(x_offset), torch.zeros_like(q), z],
        dim=-1,
    )
    ref_vel_b = torch.stack(
        [dq_dt, torch.zeros_like(dq_dt), dz_dt],
        dim=-1,
    )

    # Actual foot states in body frame.
    rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    rel_foot_vel_w = asset.data.body_lin_vel_w[:, body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    foot_vel_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    for i in range(num_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_vel_w[:, i, :])

    pos_offset = foot_pos_b - ref_pos_b
    vel_offset = foot_vel_b - ref_vel_b

    pos_err = torch.sum(torch.square(pos_offset), dim=1)
    vel_err = torch.sum(torch.square(vel_offset), dim=1)

    total_err = torch.sum(pos_err, dim=1) + float(velocity_weight) * torch.sum(vel_err, dim=1)
    reward = torch.exp(-total_err / max(std, 1e-6) ** 2)

    command = env.command_manager.get_command(command_name)
    gate = torch.linalg.norm(command[:, :3], dim=1) > command_threshold

    res = reward * gate.float()
    return res

