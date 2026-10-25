# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg, HIMPPPOAlgorithmCfg, HIMPPOActorCriticCfg

@configclass
class PPORunnerCfg(HIMOnPolicyRunnerCfg):
    num_steps_per_env = 100
    # 官方 legged_robot_config.py:267 max_iterations = 200000。此处 15000 是**预算**取舍，
    # 非语义偏离：本机是 RTX 4060 Laptop / 8GB，只能用 1024 envs（cfg 默认 4096 放不下），
    # 实测 5.66 s/迭代 → 200000 迭代要 13.1 天，而样本量仍只有官方的 25%
    # （官方 200000×100×4096 ≈ 820 亿，此处 15000×100×1024 = 15 亿）。
    # 15 亿样本跑通链路即可，之后可用 --resume 续训。
    max_iterations = 15000
    # ★ 已对齐官方 legged_robot_config.py:270 save_interval = 20（原 100）。
    #   15000 迭代下 = 750 个 checkpoint × 6.3MB ≈ 4.7GB，磁盘余量 53GB 可承受；
    #   换来更细的续训粒度。
    save_interval = 20
    experiment_name = "go2_rough"
    history_length = 5
    policy = HIMPPOActorCriticCfg(
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        init_noise_std = 1.0,
        activation = 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        normalize_obs = False,  # Enable observation normalization for numerical stability
    )
    algorithm = HIMPPPOAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.01,  # 100% 对齐 Project-Robo/HIMLOCO_GO2 与官方 HIMLoco 冷启动探索标准
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.0,
    )

    