import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
        "ab_mlp_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.ab_rsl_rl_cfg:MLPControlPPORunnerCfg",
        "ab_him_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.ab_rsl_rl_cfg:HIMABPPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"himloco_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Play",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
        "ab_mlp_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.ab_rsl_rl_cfg:MLPControlPPORunnerCfg",
        "ab_him_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.ab_rsl_rl_cfg:HIMABPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Flat",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotFlatEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"himloco_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Flat-Play",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotFlatPlayEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)

# ---- 楼梯分档（见 velocity_env_cfg.StairStageEnvCfg：固定台阶高度 + 外部评测升级）----
# 台阶高度由环境变量 HIMLOCO_STAIR_H 传入（默认 0.05）；评测侧另有 HIMLOCO_EVAL_KEEP_DR。
gym.register(
    id="Unitree-Go2-Stair-Stage",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:StairStageEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Stair-Stage-Play",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:StairStagePlayEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)
