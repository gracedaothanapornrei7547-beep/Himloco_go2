# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
自定义 HIMLOCO 地形的配置类（配合 him_terrains.py 里的生成函数一起用）。
"""

from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass
from . import him_terrains


@configclass
class HfPyramidSlopeWithNoiseCfg(HfTerrainBaseCfg):
    """带随机噪声的斜坡地形：官方金字塔斜坡 + 随机噪声叠加，最接近真实鹅卵石路。"""

    function = him_terrains.hf_pyramid_slope_with_noise_terrain  # 生成该地形的函数（在 him_terrains.py 里）

    slope_range: tuple[float, float] = MISSING          # 斜坡坡度范围（弧度），(min, max)，随 difficulty 线性插值
    platform_width: float = 1.0                         # 中央方形平台宽度（米），机器人出生区
    inverted: bool = False                              # 是否倒置：True=平台在底、坡朝上；默认 False=平台在顶、坡朝下
    noise_amplitude_range: tuple[float, float] = MISSING  # 噪声幅度范围（米），(min, max)，实际 = min + difficulty*(max-min)
    noise_step: float = MISSING                         # 噪声高度采样步长（米），越小噪声越细腻
    downsampled_scale: float = MISSING                  # 噪声降采样尺度（米）：先低分辨率生成噪声，再样条插值放大


@configclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """离散障碍块地形：随机散布不同高度的矩形障碍（对齐 HIMLOCO_GO2 实现）。"""

    function = him_terrains.hf_discrete_obstacles_terrain  # 生成该地形的函数（在 him_terrains.py 里）

    max_height_range: tuple[float, float] = MISSING       # 障碍最大高度范围（米），(min, max)，实际 = min + difficulty*(max-min)
    obstacle_size_range: tuple[float, float] = MISSING    # 障碍块尺寸范围（米），(min, max)，每块随机取宽/长
    num_obstacles: int = MISSING                          # 随机障碍块数量
    platform_width: float = 1.0                           # 中央方形平台宽度（米），机器人出生区（最后会被削平为 0）
