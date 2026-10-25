import math
import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from himloco_lab.assets.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion import mdp
import himloco_lab.terrains as him_terrains

# =====================================================================
# Go2 速度跟踪任务的环境配置（RobotEnvCfg）。
# 由 gym 注册名 "Unitree-Go2-Velocity" -> velocity_env_cfg:RobotEnvCfg 解析。
# 定义：机器人 / 地形 / 观测 / 动作 / 奖励 / 终止 / 课程。
# ★ 要调 reward 权重 -> 改下面 RewardsCfg 里各 RewTerm 的 weight=。
# =====================================================================

# 地形生成器：程序化生成「鹅卵石路」风格多难度地形。
# num_rows=10 难度等级 × num_cols=20 列；难度随 curriculum.terrain_levels 逐步解锁更难地形。
# 下面 sub_terrains 的 proportion 是各类型地形的面积占比（合计 ≈ 1.0）。
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 每块地形 8×8 m
    border_width=25.0,  # 四周缓冲带宽度（防机器人跑出地形边界）
    num_rows=10,  # number of difficulty levels
    num_cols=20,  # number of terrain types
    horizontal_scale=0.1,  # [m] grid resolution
    vertical_scale=0.005,  # [m] height scale
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        # 上坡金字塔（斜面上坡），占 5%
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,               # 总占比（各类型合计 ≈ 1.0）
            slope_range=(0.0, 0.4),        # 坡度范围（弧度），随 difficulty 0→0.4 rad 递增
            platform_width=3.0,            # 中央平台宽度（米），机器人出生区
            border_width=0.0,              # 边界缓冲宽度（米），地形间过渡
        ),
        # 下坡金字塔（斜面下坡），占 5%
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,               # 总占比
            slope_range=(0.0, 0.4),        # 坡度范围（弧度），随 difficulty 递增
            platform_width=3.0,            # 中央平台宽度（米）
            border_width=0.0,              # 边界缓冲宽度（米）
        ),
        # 带随机噪声的斜坡，占 20%（最接近真实鹅卵石路，噪声幅度 0.01~0.08m）
        "hf_slope_with_noise": him_terrains.HfPyramidSlopeWithNoiseCfg(
            proportion=0.2,                # 总占比
            slope_range=(0.0, 0.4),        # 坡度范围（弧度）
            platform_width=3.0,            # 中央平台宽度（米）
            border_width=0.0,              # 边界缓冲宽度（米）
            noise_amplitude_range=(0.01, 0.08),  # 噪声幅度范围（米），随 difficulty 递增
            noise_step=0.005,              # 噪声高度采样步长（米），越小噪声越细腻
            downsampled_scale=0.2,         # 噪声降采样尺度（米）：先低分辨率生成再样条插值放大
        ),
        # 上楼梯，占 30%（台阶高随难度 5→20cm 递增，保证 15cm 在第 6 行即覆盖，高行充裕攻坚）
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,                 # 总占比
            step_height_range=(0.05, 0.20),
            step_width=0.30,                # 台阶宽度（对齐官方 step_width=0.30）
            platform_width=3.0,             # 楼梯 顶部 的平台长度（对齐官方 platform_size=3.）
            border_width=0.0,               # ★ 回 IsaacLab 默认：官方 terrain_utils 无 border 概念
        ),
        # 下楼梯，占 30%（台阶高随难度 5→20cm 递增，保证 15cm 在第 6 行即覆盖，高行充裕攻坚）
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,                 # 总占比
            step_height_range=(0.05, 0.20),
            step_width=0.30,                # 台阶宽度（对齐官方 step_width=0.30）
            platform_width=3.0,             # 楼梯 底部 的平台长度（对齐官方 platform_size=3.）
            border_width=0.0,               # ★ 回 IsaacLab 默认（官方无 border 概念）
        ),                                                                    
        # 离散障碍块，占 10%（1~2m 方块 × 20 个）
        "discrete_obstacles": him_terrains.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,                 # 总占比
            max_height_range=(0.05, 0.15),  # 障碍最大高度范围（米），0.05 + 0.1 * difficulty
            obstacle_size_range=(1.0, 2.0), # 障碍尺寸范围（米），min=1m, max=2m
            num_obstacles=20,               # 障碍块数量
            platform_width=3.0,             # 中央平台宽度（米），机器人出生区
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # A local preview material keeps headless training independent from Nucleus.
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.35, 0.35),
            roughness=0.8,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # 启动时随机化全身摩擦/恢复系数（域随机化：模拟不同地面材质，sim2real 关键）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 启动时给 base 随机加 -1~+2 kg 质量（增强对负载变化的鲁棒性）
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    
    # 启动时随机化 base 质心位置（±5cm）
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset

    # 每轮 reset：随机初始位置（±0.5m），yaw 固定 0，速度归零
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # 每轮 reset：关节角度固定默认站立值（对齐官方，彻底杜绝开局趴窝），速度归零
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0, 0),
        },
    )

    # interval
    # 周期性外力：给 base 施加 ±30N 随机力（抗扰训练，模拟被推/被撞）
    external_force = EventTerm(
        func=mdp.apply_periodic_external_force_torque,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "period_step": 8,
            "force_range": (-30.0, 30.0),
            "torque_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    # 每 16s 随机推一把 base（±1 m/s），测抗推扰恢复能力
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={
            "velocity_range": {"x": (-1, 1), "y": (-1, 1)},
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 速度指令：对齐 Project-Robo 楼梯攻坚冲量规范（前向 0.8~1.4 m/s 饱满推力）
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        heading_command=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 1.4), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi)
        ),
        heading_control_stiffness=0.5,
        curriculums_limit_ranges=(-1.5, 1.5),  
        low_vel_env_lin_x_ranges=(0.6, 1.0),
        rel_high_vel_envs=0.2,
        min_command_norm=0.2,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 动作 = 12 个关节的目标位置偏移；对齐官方基准：全 12 关节统一 scale=0.25，clip=(-100.0, 100.0)
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )

# observation compute step in lab: noise clip scale
# observation compute step in gym: clip scale noise
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测（policy 实际看到的东西 = 盲走的本体感觉，无任何地形/高度信息）。

        组成：速度指令 + 本体角速度 + 重力投影 + 关节位置/速度 + 上一帧动作。
        注意这里没有 height_scan —— 那是 CriticCfg 里的特权观测，policy 看不到。"""

        # observation terms (order preserved)
        #   顺序说明：IsaacLab 是 clip→scale，此处 clip=±100 在 scale 前生效；|cmd|≤2 → 乘完 ≤2.0，
        #   clip 完全惰性，故与官方「先乘后裁」等价。
        #   部署端无需改 C++：export_deploy_cfg.py 会把该 scale 写进 deploy.yaml，
        #   observation_manager.h:147/64-66 读 yaml 后逐元素相乘。
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            scale=(2.0, 2.0, 0.25),
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True  # 允许给观测加噪（观测域随机化，sim2real 鲁棒性）

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(PolicyCfg):
        """Critic 观测（特权观测，训练期 critic 额外看到的，policy 永远拿不到）。

        在 policy 观测基础上追加：base_lin_vel(真实线速度) + 外力 + height_scanner(高度扫描)。
        这是 teacher-student 里 teacher 的「上帝视角」，让 critic 更容易拟合 value，sim2real 部署时不需要。"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100))
        base_external_force = ObsTerm(
            func=mdp.base_external_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            clip=(-100, 100),
        )
        height_scanner = ObsTerm(func=mdp.height_scan_clip,
            scale=5.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        def __post_init__(self):
            self.enable_corruption = True  # 特权观测同样加噪，防止 critic 过拟合

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """奖励项配置（MDP Reward Specification）。

    【核心机制与架构关系】：
    1. 标量奖励计算：每个控制周期（dt=0.02s），环境计算各项奖励值，单步总奖励 R_t = Σ (weight_i × term_func_i)。
    2. 算法衔接机制：总奖励 R_t 直接喂给底层策略优化器（HIMPPO），通过广义优势估计（GAE-λ）计算优势值 A_t，
       并利用 PPO 的裁剪替代目标损失（Clipped Surrogate Loss）更新 Actor 与 Critic 网络。
    3. 理论框架：以 HIMLoco（ICLR 2024）特权隐式估计为基石，吸收 robot_lab 与 unitree_rl_lab 开源量产调参精髓。
    """

    # ==============================================================================
    # 第一层：核心任务正向牵引（Task Tracking Rewards —— 3:1 绝对主导优势）
    # 物理意义：赋予机器人强烈的奔跑意图，在数学上压倒一切“原地踏步”或“消极站桩”的保守局部最优解。
    # ==============================================================================

    # 前向与侧向线速度跟踪奖励：强力拉高至 4.0 形成绝对主导驱动力，彻底打碎原地站桩
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    # 偏航角速度跟踪奖励：1.5
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    # ==============================================================================
    # 第二层：机身平稳性与动态解耦阻尼（Body Stability & Damping）
    # ==============================================================================

    # 机身垂向速度阻尼：lin_vel_z = -1.0（放宽垂向速度阻尼，允许机身在台阶上爬升）
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    # 核心机身高度锚：对齐 Project-Robo 标称 -1.0（目标高度 0.30m），允许爬坡时自然仰头起伏
    base_height = RewTerm(
        func=mdp.base_height,
        weight=-1.0,
        params={
            "target_height": 0.30,
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )

    # 机身角速度阻尼：ang_vel_xy = -0.05
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 水平姿态回正：orientation = -0.2
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.20)

    # 动作变化率一阶阻尼：action_rate = -0.01
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 动作二阶平滑惩罚：smoothness = -0.01
    smoothness = RewTerm(func=mdp.smoothness, weight=-0.01)

    # ==============================================================================
    # 第三层：关节能耗与动力学约束（Joint Dynamics & Energy）
    # ==============================================================================

    # 关节角加速度正则：dof_acc = -2.5e-7
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # 机械功率能耗正则：joint_power = -2e-5
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # ★ 髋关节标称构型约束（根治平地内八与楼梯外八）：
    # 针对 4 个髋关节（.*_hip_joint），双向二次方惩罚偏离标称角度（FL/RL=+0.10, FR/RR=-0.10 rad）
    # 直线行走时全权重施加，转向时自动门控放宽，迫使四足在矢状面内前向抬腿
    hip_deviation = RewTerm(
        func=mdp.hip_deviation_adaptive,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "command_name": "base_velocity",
        },
    )

    # ==============================================================================
    # 第四层：足端动力学与台阶自适应抬腿（彻底解除低空下压枷锁）
    # ==============================================================================

    # 解除硬性贝塞尔引导项（平地大步幅已定型，楼梯需要触觉反射自由抬腿）
    phase_foot_trajectory = None

    # 激活单侧防擦地安全抬腿（仅在脚离地低于 3.5cm 拖步时微惩罚，爬楼梯高抬腿零惩罚）
    foot_clearance = RewTerm(
        func=mdp.feet_min_height_body,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.305,
            "margin": 0.035,
        },
    )

    # 彻底清空毒药项：切断下蹲压制与早触地负分惩罚牢笼
    feet_air_time = None



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 正常跑满 20s 结束（Episode_Termination/time_out 曲线 = 存活率）
    base_contact = DoneTerm(  # 身体触地→终止（摔倒）。Episode_Termination/base_contact 曲线 = 摔倒率
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # 越界结束：跑出地形边界 + 3m 缓冲（按 time_out 处理，不算惩罚性终止）
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)  # 地形难度课程：跟踪得好就升难度(0~5)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)  # 速度指令幅度课程：跟踪得好就提高指令速度


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings（控制频率与单集长度）
        self.decimation = 4  # 策略控制频率 = 1/(sim.dt × 4) = 50Hz（每 4 个物理步走一步策略）
        self.episode_length_s = 20.0  # 单集 20s = 20/(0.005×4) = 1000 步
        
        # simulation settings（物理步长 5ms = 200Hz 仿真）
        self.sim.dt = 0.005  # 物理步长 5ms = 200Hz 仿真
        self.sim.render_interval = self.decimation  # 渲染帧间隔 = decimation（与策略步同步，省渲染开销）
        self.sim.physics_material = self.scene.terrain.physics_material  # 物理材质跟地形一致（摩擦系数）
        
        # PhysX settings（TGS 求解器 + 较少位置迭代，速度/精度折中）
        self.sim.physx.solver_type = 1  # TGS solver
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.5
        
        self.sim.physx.gpu_max_rigid_patch_count = 2**23   # GPU 刚体 patch 上限（2^23≈840 万，防大 env 数量 OOM）
        self.sim.physx.gpu_max_rigid_contact_count = 2**23   # GPU 接触对上限
        # self.sim.physx.gpu_found_lost_pairs_capacity = 2**23

        # 传感器更新周期 = 物理步长 × decimation（即每 4 步 / 50Hz 更新一次，与策略控制同步）
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.base_height_scanner.update_period = self.sim.dt * self.decimation

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # （配置了 terrain_levels 课程 → 打开地形生成器的 curriculum，随训练逐步变难）
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play/部署模式环境：固定 1.0 m/s 前进，关闭大部分域随机化，用于播放 policy 验证。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64  # 部署时用 64 个并行环境（可视化播放）
        self.scene.terrain.terrain_generator.num_cols = 10  # 只保留 10 列地形类型（减少地形生成开销）
        self.scene.terrain.max_init_terrain_level = 0  # 初始从 0 级（基础台阶）开始回放
        self.scene.terrain.terrain_generator.curriculum = True  # 保持课程开（play 时也按难度排地形）
        self.commands.base_velocity.heading_command = False  # 关闭 heading 指令（只走直线前进）
        self.commands.base_velocity.debug_vis = False  # 关掉速度指令箭头可视化（避免 play 时去云端拉 arrow_x.usd 卡住启动）
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1, 1), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0, 0),   # 固定 vx=1.0 m/s，vy/wz 归零
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges=(1, 1)  # 低速环境也固定 1.0 m/s
        
        # 关闭域随机化事件（play 只需确定性验证，不需要抗扰训练）
        self.events.add_base_mass = None              # 关：base 随机加质量
        self.events.randomize_rigid_body_com = None   # 关：质心随机偏移
        self.events.push_robot = None                 # 关：随机推扰


@configclass
class RobotFlatEnvCfg(RobotEnvCfg):
    """阶段一：纯平地环境配置。用于大步幅高速直行（1.5~1.8 m/s Trot）特训。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # 强化大步幅高速直行：将速度目标拉升至 1.5 ~ 1.8 m/s
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.8)
        self.commands.base_velocity.low_vel_env_lin_x_ranges = (1.2, 1.5)
        self.commands.base_velocity.curriculums_limit_ranges = (-1.8, 1.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class RobotFlatPlayEnvCfg(RobotFlatEnvCfg):
    """阶段一：纯平地可视化回放环境。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1.6, 1.6), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges = (1.6, 1.6)
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_robot = None


# =====================================================================
# 楼梯分档环境（固定台阶高度 + 外部评测升级）
# ---------------------------------------------------------------------
# 各档位的台阶高度（米）。5 → 7.5 → 10cm。
STAIR_STAGE_HEIGHTS = (0.05, 0.075, 0.10)


def stair_height_from_env() -> float:
    """当前档位的台阶高度（米）。由环境变量 `HIMLOCO_STAIR_H` 控制，默认 0.05。"""
    return float(os.environ.get("HIMLOCO_STAIR_H", str(STAIR_STAGE_HEIGHTS[0])))


# 台阶占 70%，其余地形保留做多样性（同时也是 G4 保持性的观察窗口）。
# 20 列 × 这些比例 = [1, 1, 2, 7, 7, 2] = 20 列，恰好整数。
_STAIR_PROPORTIONS = {
    "hf_pyramid_slope": 0.05,
    "hf_pyramid_slope_inv": 0.05,
    "hf_slope_with_noise": 0.10,
    "pyramid_stairs": 0.35,
    "pyramid_stairs_inv": 0.35,
    "discrete_obstacles": 0.10,
}


@configclass
class StairStageEnvCfg(RobotEnvCfg):
    """单档固定台阶高度的楼梯训练环境（训练用；评测用 StairStagePlayEnvCfg）。"""

    def __post_init__(self):
        super().__post_init__()
        h = stair_height_from_env()
        gen = self.scene.terrain.terrain_generator
        if gen is None:
            raise RuntimeError("楼梯档必须有 terrain_generator")

        # ---- 1) 台阶钉死成单一高度 ----
        for k in ("pyramid_stairs", "pyramid_stairs_inv"):
            gen.sub_terrains[k].step_height_range = (h, h)

        # ---- 2) 占比：台阶 70% ----
        if set(gen.sub_terrains) != set(_STAIR_PROPORTIONS):
            raise RuntimeError(
                f"地形集合与预期不符：{sorted(gen.sub_terrains)} != {sorted(_STAIR_PROPORTIONS)}"
            )
        for k, v in _STAIR_PROPORTIONS.items():
            gen.sub_terrains[k].proportion = v
        n_cols = gen.num_cols
        if any(abs(n_cols * v - round(n_cols * v)) > 1e-9 for v in _STAIR_PROPORTIONS.values()):
            raise RuntimeError(f"占比必须能整除 {n_cols} 列")

        # ---- 3) 难度课程交给外部调度器 ----
        gen.curriculum = True                       # 保留"行"结构
        self.scene.terrain.max_init_terrain_level = None   # None → 全行均匀散布
        self.curriculum.terrain_levels = None       # 关掉位移驱动的难度漂移
        # 速度课程同样交给外部（本档只练"能不能爬上去"，不练提速）
        self.curriculum.lin_vel_cmd_levels = None

        # ---- 4) 命令范围：0 ~ 1.1 m/s 前进 ----
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.1), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges = (0.0, 1.1)
        self.commands.base_velocity.curriculums_limit_ranges = (0.0, 1.1)

        print(
            f"[StairStage] 台阶高度 = {h * 100:.1f} cm（HIMLOCO_STAIR_H={os.environ.get('HIMLOCO_STAIR_H', '<默认>')}）\n"
            f"[StairStage] 地形占比 = {_STAIR_PROPORTIONS}（台阶合计 0.70）\n"
            f"[StairStage] terrain_levels 课程 = 关（难度不漂移）；命令范围 = 0~1.1 m/s"
        )


@configclass
class StairStagePlayEnvCfg(StairStageEnvCfg):
    """评测用：固定直线前进 + 可选的域随机化保留，便于逐类判定通过/失败。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.commands.base_velocity.debug_vis = False  # 避免 play 时去云端拉 arrow_x.usd 卡住启动
        self.commands.base_velocity.heading_command = False
        # 评测口径：固定 vx，vy/wz 归零（逐类判定"这一档能不能直线走完"）
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges = (1.0, 1.0)

        keep_dr = os.environ.get("HIMLOCO_EVAL_KEEP_DR", "0") == "1"
        if not keep_dr:
            self.events.add_base_mass = None
            self.events.randomize_rigid_body_com = None
            self.events.push_robot = None
        print(f"[StairStagePlay] 域随机化 = {'开（保留）' if keep_dr else '关'}")
