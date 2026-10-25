# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controlled ordinary-history MLP versus HIM configuration pair.

A/B 对照实验的 Agent 配置：两个变体共享同一套超参，唯一区别是 policy 网络结构，
用来隔离「HIM 速度估计器」相对「普通历史拼接 MLP」到底有没有额外收益。
"""

from isaaclab.utils import configclass

from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg, HIMPPPOAlgorithmCfg, HIMPPOActorCriticCfg


# A/B 协议标识 id，会写进 ab_manifest.json，用来把两个变体的日志/checkpoint 配对归档。
AB_PROTOCOL_ID = "go2_him_vs_history_mlp_v1"


@configclass
class ABCommonPPORunnerCfg(HIMOnPolicyRunnerCfg):
    """Everything that must remain identical between the two variants.

    两个变体「必须完全一致」的公共部分：种子、每环境步数、总迭代数、history 长度、
    policy/critic 网络结构、PPO 超参。子类只改 policy_class_name / experiment_name / ab_variant。
    """

    seed = 42                       # 随机种子（决定 env 初始状态 / 命令采样 / 噪声）
    num_steps_per_env = 100         # 每个 env 采 100 步再做一次 PPO 更新；batch = num_envs × 100
    max_iterations = 15000          # 总迭代轮数（正式 run 默认值；CLI --max_iterations 可覆盖）
    save_interval = 100             # 每 100 轮存一次 checkpoint
    history_length = 5              # current frame + five past frames = 6 x 45 = 270（当前帧 + 过去 5 帧）
    privileged_history_length = 0   # one 238-dimensional critic frame（critic 用 1 帧特权观测，不做堆叠）
    ab_protocol_id = AB_PROTOCOL_ID

    policy = HIMPPOActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],  # actor（策略）三层 MLP 隐层维度
        critic_hidden_dims=[512, 256, 128], # critic（值函数）三层 MLP，与 actor 同构
        init_noise_std=1.0,                 # 初始动作噪声标准差（探索强度，随训练衰减）
        activation="elu",                   # 激活函数
        normalize_obs=False,                # 是否归一化观测（关闭 = 靠网络自己适应尺度）
        initialization_seed=42,             # 网络权重初始化种子（与 env seed 解耦）
    )
    algorithm = HIMPPPOAlgorithmCfg(
        value_loss_coef=1.0,            # critic 值函数损失权重
        use_clipped_value_loss=True,    # 用 value clip（限制 value 更新幅度，防过冲）
        clip_param=0.2,                 # PPO 裁剪阈值：新旧策略概率比超出 [0.8,1.2] 就截断
        entropy_coef=0.01,              # 熵正则系数：鼓励探索，随训练熵会下降
        num_learning_epochs=5,          # 每个 rollout 重复更新 5 轮
        num_mini_batches=4,             # 分 4 个 minibatch，batch = num_envs*nsteps/4
        learning_rate=1.0e-3,           # 初始学习率（adaptive schedule 会动态调）
        schedule="adaptive",            # 按 KL 自适应调 lr，desired_kl 超了就降
        gamma=0.99,                     # 折扣因子
        lam=0.95,                       # GAE λ：控制优势估计的 bias-variance 权衡
        desired_kl=0.01,                # 目标 KL：adaptive 调 lr 的锚点
        max_grad_norm=1.0,              # 梯度裁剪上限
    )


@configclass
class MLPControlPPORunnerCfg(ABCommonPPORunnerCfg):
    """对照组 A：普通「历史拼接 MLP」策略，没有 HIM 速度估计器。"""

    experiment_name = "go2_ab_history_mlp"
    policy_class_name = "HistoryMLPActorCritic"     # 直接把 270 维历史观测喂给 MLP，无估计器分支
    algorithm_class_name = "HIMPPO"                 # 算法仍用 HIMPPO，只换 policy 结构，保证公平对照
    ab_variant = "history_mlp_270"


@configclass
class HIMABPPORunnerCfg(ABCommonPPORunnerCfg):
    """实验组 B：HIM 策略（HIMActorCritic），带速度估计器 + latent 原型对比损失。"""

    experiment_name = "go2_ab_him"
    policy_class_name = "HIMActorCritic"  # 估计器 270→128→64→19（速度 3 + latent 16）
    algorithm_class_name = "HIMPPO"
    ab_variant = "him_release_encoder_270_to_19"
