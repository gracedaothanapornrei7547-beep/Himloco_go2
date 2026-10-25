# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
HIMLOCO 自定义地形生成函数（配合 him_terrains_cfg.py 里的配置类一起用）。
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import him_terrains_cfg


@height_field_to_mesh
def hf_pyramid_slope_with_noise_terrain(
    difficulty: float,
    cfg: him_terrains_cfg.HfPyramidSlopeWithNoiseCfg
) -> np.ndarray:
    """带随机噪声的斜坡：官方金字塔斜坡 + 随机均匀噪声，最接近真实鹅卵石路。"""

    # 1. 复用官方金字塔斜坡函数，拿到基础高度场（int16 离散单位，未转 mesh）
    from isaaclab.terrains.height_field import hf_terrains

    pyramid_func = hf_terrains.pyramid_sloped_terrain.__wrapped__
    height_field = pyramid_func(difficulty, cfg)  # int16 数组，离散单位

    # 2. 按难度插值出当前噪声幅度：amplitude = min + difficulty*(max - min)
    amplitude = cfg.noise_amplitude_range[0] + difficulty * (
        cfg.noise_amplitude_range[1] - cfg.noise_amplitude_range[0]
    )

    # 3. 先降采样生成低分辨率噪声（省计算，且噪声更平滑）
    downsampled_scale = cfg.downsampled_scale
    noise_step = cfg.noise_step

    width = height_field.shape[0]
    length = height_field.shape[1]
    horizontal_scale = cfg.horizontal_scale
    vertical_scale = cfg.vertical_scale

    width_downsampled = int(width * horizontal_scale / downsampled_scale)
    length_downsampled = int(length * horizontal_scale / downsampled_scale)

    # 4. 噪声高度也转成离散单位（除以 vertical_scale），对齐 isaacgym terrain_utils 做法
    height_min = int(-amplitude / vertical_scale)
    height_max = int(amplitude / vertical_scale)
    height_step = int(noise_step / vertical_scale)

    heights_range = np.arange(height_min, height_max + height_step, height_step)
    height_field_downsampled = np.random.choice(heights_range, (width_downsampled, length_downsampled))

    # 5. 用样条插值把低分辨率噪声放大到全分辨率
    x = np.linspace(0, width * horizontal_scale, width_downsampled)
    y = np.linspace(0, length * horizontal_scale, length_downsampled)
    f = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    x_upsampled = np.linspace(0, width * horizontal_scale, width)
    y_upsampled = np.linspace(0, length * horizontal_scale, length)
    noise_field = f(x_upsampled, y_upsampled)

    # 6. 四舍五入回 int16，叠加到基础高度场上
    height_field = height_field + np.rint(noise_field).astype(np.int16)

    return height_field


@height_field_to_mesh
def hf_discrete_obstacles_terrain(
    difficulty: float,
    cfg: him_terrains_cfg.HfDiscreteObstaclesTerrainCfg
) -> np.ndarray:
    """离散障碍块地形：随机散布不同高度的矩形障碍（对齐 HIMLOCO_GO2 实现）。"""

    # 1. 按难度插值出障碍最大高度：max_height = min + difficulty*(max - min)
    max_height = cfg.max_height_range[0] + difficulty * (
        cfg.max_height_range[1] - cfg.max_height_range[0]
    )

    # 2. 把所有物理量从「米」转成离散像素单位（除以对应 scale）
    max_height_discrete = int(max_height / cfg.vertical_scale)
    min_size = int(cfg.obstacle_size_range[0] / cfg.horizontal_scale)
    max_size = int(cfg.obstacle_size_range[1] / cfg.horizontal_scale)
    platform_size = int(cfg.platform_width / cfg.horizontal_scale)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # 3. 初始化全 0 高度场
    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    # 4. 障碍高度取值集合（对齐 HIMLOCO）：[-max, -max/2, +max/2, +max]
    height_range = [
        -max_height_discrete,
        -max_height_discrete // 2,
        max_height_discrete // 2,
        max_height_discrete
    ]

    # 5. 障碍尺寸按步长 4 离散化（对齐 HIMLOCO）
    width_range = list(range(min_size, max_size, 4))
    length_range = list(range(min_size, max_size, 4))

    # 6. 随机生成 num_obstacles 个矩形障碍块
    for _ in range(cfg.num_obstacles):
        if len(width_range) == 0 or len(length_range) == 0:
            break

        width = np.random.choice(width_range)
        length = np.random.choice(length_range)

        if width_pixels - width <= 0 or length_pixels - length <= 0:
            continue

        start_i = np.random.choice(range(0, width_pixels - width, 4))
        start_j = np.random.choice(range(0, length_pixels - length, 4))

        height_field[start_i:start_i + width, start_j:start_j + length] = np.random.choice(height_range)

    # 7. 中央削平一块方形平台（机器人出生区），高度置 0
    x1 = (width_pixels - platform_size) // 2
    x2 = (width_pixels + platform_size) // 2
    y1 = (length_pixels - platform_size) // 2
    y2 = (length_pixels + platform_size) // 2
    height_field[x1:x2, y1:y2] = 0

    return height_field
