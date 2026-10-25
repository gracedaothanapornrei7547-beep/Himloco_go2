from __future__ import annotations

import math

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.curriculums_limit_ranges
    
    # Split environments into low-vel and high-vel groups
    low_vel_env_ids = (env_ids >= (env.num_envs * command_term.cfg.rel_high_vel_envs))
    high_vel_env_ids = (env_ids < (env.num_envs * command_term.cfg.rel_high_vel_envs))
    low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
    high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    
    if env.common_step_counter % env.max_episode_length == 0:
        # Calculate rewards separately for low and high vel groups
        reward_low = torch.mean(env.reward_manager._episode_sums[reward_term_name][low_vel_env_ids]) / env.max_episode_length if len(low_vel_env_ids) > 0 else 0.0
        reward_high = torch.mean(env.reward_manager._episode_sums[reward_term_name][high_vel_env_ids]) / env.max_episode_length if len(high_vel_env_ids) > 0 else 0.0

        # Only update curriculum if both groups are performing well
        # ★ 对齐官方 update_command_curriculum（legged_robot.py:590）：门槛 = weight * 0.8，
        #   且 episode_sums 除以 max_episode_length（步数，1000），不是 max_episode_length_s（秒，20）。
        #   原实现两处都偏：阈值 0.6、除数用秒 —— 除数差 dt=0.02 倍即把判据放大 50 倍，几乎恒真。
        #   当时靠 curriculums_limit_ranges=(-1,1) 把速度锁死在 ±1.0 兜底，现已一并回官方 (-2,2)。
        if reward_low > reward_term.weight * 0.8 and reward_high > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """地形难度课程（判据已对齐官方 HIMLoco）。

    升难度 = 本集实际位移 > 半块地形边长（``terrain_length/2``，默认 8/2 = 4m），
    降难度 = 本集实际位移 < ‖cmd‖ × 单集时长 × 0.5；两者互斥（降难度会乘 ``~move_up``）。
    这与官方 ``legged_robot.py:567-571`` 的 ``_update_terrain_curriculum`` 逐行等价。

    与官方的第二处对齐是**首次 reset 跳过**：官方用 ``if not self.init_done: return``
    （``legged_robot.py:564-566``）挡住初始化那一次，本文件用 ``common_step_counter == 0``
    表达同一语义。少了它，第一次 ``env.reset()`` 会拿机器人尚未摆放的原始位姿当位移，
    把全体 env 白升一级（详见函数内注释）。

    历史说明：本文件曾把升难度改成「0.8 × ‖cmd‖ × 单集时长」（1m/s 下 16m，比官方严 4 倍），
    意图是让策略在难地形多练几集。实测该门槛使课程几乎不推进（p19 有 97.4% 的 env 停在
    level 0），故按「全面对齐官方」回退到官方判据。

    Returns:
        分项状态 dict（CurriculumManager 会把每个 key 记成单独 scalar，见
        isaaclab/managers/curriculum_manager.py:108-113）：
          - ``mean``           : 全体 env 的平均难度行
          - ``frac_10cm``      : 落在「≥10cm 台阶」难度行的 env 占比（由台阶公式反解行号，
                                 当前地形配置下 = level ≥ 3）
          - ``move_up_frac``   : 本批 reset 中「升难度」的比例
          - ``move_down_frac`` : 本批 reset 中「降难度」的比例
          - ``lvl<i>``         : 每一行难度上的 env 占比，i ∈ [0, num_rows)
        之所以要直方图而不是均值：均值 3.4 既可能是「全体挤在 3-4 行」，也可能是
        「一半环境在 0-1、另一半在 6-7」，两者对训练的含义完全不同。move_up/down 占比
        则直接暴露门槛是否饱和（长期 100%/0% 说明门槛已失效）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # 本集走过的位移（自上次换难度后；env_origins 在换难度时会重置）
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # ★ 判据对齐官方 _update_terrain_curriculum（legged_robot.py:567-571）：
    #   官方升难度 = 走够半块地形（固定 terrain.env_length/2 = 8/2 = 4m），与指令速度无关；
    #   官方降难度 = 实际位移 < ‖cmd‖ * max_episode_length_s * 0.5。
    #   原实现把升难度改成 0.8 * ‖cmd‖ * max_episode_length_s（1m/s 下 16m，严 4 倍），
    #   课程因此几乎不推进 —— p19 实测 97.4% 的 env 卡在 level 0，就是它的直接后果。
    terrain_length = env.scene.terrain.cfg.terrain_generator.size[0]
    move_up = distance > terrain_length / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # ★ init 守卫，等价官方 `if not self.init_done: return`（legged_robot.py:564-566）。
    #   `_reset_idx` 里 curriculum.compute() 排在 scene.reset() 之前（manager_based_rl_env.py:356 vs 358），
    #   这是有意设计——要在重置前量「本集走了多远」；但「把机器人摆到 env_origins」这件事只在
    #   scene.reset() 里发生。于是第一次 env.reset() 量到的是机器人尚未被摆放的原始位姿：
    #   实测 root_pos_w 与 env_origins 相差均值 50.2m（max 108m，网格出生布局 vs 地形原点），
    #   且此刻指令全为 0 → move_down 阈值恒为 0；结果 move_up 对 99.6% 的 env 为真，
    #   全体白升一级（terrain_levels 2.5137 → 3.5098），level 0 直接空掉。
    #   TerrainImporter 没有 reset()，update_env_origins 是 terrain_levels 的唯一改写者，
    #   这一级升上去就再也回不来，故必须挡在门外。
    #   common_step_counter 在 step() 内、_reset_idx 之前自增（manager_based_rl_env.py:202），
    #   所以 `== 0` 恰好只命中「训练脚本第一次 env.reset()」这一次。
    if env.common_step_counter > 0:
        terrain.update_env_origins(env_ids, move_up, move_down)
    else:
        # 首次 reset 不产生任何难度变动；统计口径也归 0，避免那份脏 distance 的判据污染曲线
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    # ---- 分项统计（全体 env，不只是本批 reset 的）----
    num_rows = int(terrain.max_terrain_level)
    levels = terrain.terrain_levels  # (num_envs,) int64，全体
    counts = torch.bincount(levels, minlength=num_rows).float() / levels.numel()

    # 「≥10cm 台阶」从第几行开始？台阶高公式已对齐官方 0.05 + 0.18*d（d = row/num_rows），
    # 反解得 0.05 + 0.18*(row/num_rows) >= 0.10 → row >= 2.78 → 第 3 行起。
    # 之前地形上限只有 5cm，这个键恒为 0 只能当护栏；现在它是真指标了。
    first_10cm_row = num_rows - 2  # 兜底：取不到地形配置时退回原来的「最上两行」
    try:
        stairs_cfg = env.scene.terrain.cfg.terrain_generator.sub_terrains["pyramid_stairs"]
        lo, hi = stairs_cfg.step_height_range
        if hi > lo:
            first_10cm_row = int(math.ceil((0.10 - lo) / (hi - lo) * num_rows))
        first_10cm_row = max(0, min(first_10cm_row, num_rows - 1))
    except Exception:
        pass

    state: dict[str, torch.Tensor] = {
        "mean": levels.float().mean(),
        "frac_10cm": (levels >= first_10cm_row).float().mean(),
        "move_up_frac": move_up.float().mean(),
        "move_down_frac": move_down.float().mean(),
    }
    for i in range(num_rows):
        state[f"lvl{i}"] = counts[i]
    return state
