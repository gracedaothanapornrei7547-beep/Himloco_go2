# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""G0b：部署链路 obs 拼装一致性检查（不是导出保真度 —— 那由 compare_pt_onnx.py 负责）。

为什么需要这一道：
    `compare_pt_onnx.py` 里仿真永远由 PT 动作驱动（`env.step(pt_actions)`），ONNX 只是
    拿同一个 obs 算一遍比动作 —— 它证明"导出没坏"，但**证明不了部署链路的 obs 拼装是对的**。
    agent.md §8.5 记的三个真实 bug（last_action 未乘 action_scale、关节顺序 gather/scatter、
    Go2HV 力矩限幅）全都发生在拼装环节，恰好是那道脚本够不到的地方。

本脚本做什么：
    在**同一个 IsaacLab 环境**里，用 `params/deploy.yaml` 的规则（= C++ 运行时读的那份规格）
    从**原始机器人状态**重新拼一遍 45 维 obs，然后与 IsaacLab 自己喂给策略的 obs 逐项对拍。
    逐项拆开报告（而不是比整个向量），这样一旦不一致，能直接指出是**哪一项**错的。

    关节顺序用**枚举候选排列**的方式判定，而不是靠读 `joint_ids_map` 猜方向 ——
    哪个排列能对上就是哪个。

用法：
    ./run_himloco.sh scripts/himloco_rsl_rl/g0b_deploy_obs_parity.py \
        --task Unitree-Go2-Velocity-Play \
        --checkpoint <run_dir>/model_XXXX.pt \
        --num_envs 4 --num_steps 200
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="G0b: deploy-chain observation assembly parity check.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="himloco_rsl_rl_cfg", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of control steps to compare.")
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Pin forward command (m/s) for the whole run.")
cli_args.add_himloco_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in sys.argv:
    sys.argv.append("--headless")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch
import yaml

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _perm_from_joint_ids_map(m, n=12):
    """把 deploy.yaml 的 joint_ids_map 解释成一个排列。

    只做**两种**自然解释，不猜第三种：
      - "deploy_from_train": deploy_q[i] = train_q[m[i]]
      - "train_from_deploy": train_q[m[i]] = deploy_q[i]
    两者互为逆排列。
    """
    a = torch.tensor(m, dtype=torch.long)
    b = torch.empty_like(a)
    b[a] = torch.arange(n, dtype=torch.long)
    return a, b


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # 钉死指令，让对拍只反映装配差异而不反映指令采样
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (args_cli.cmd_vx, args_cli.cmd_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # ★ 关掉观测噪声（策略观测组的域随机化）。开着的话，env 喂给策略的 obs 是
    #   "真值 + U(-b,b) 噪声"，而我们重拼的是真值 —— 两者本来就不该相等。
    #   base_ang_vel/gravity/joint_pos/joint_vel 四项都带 Unoise，只有 velocity_commands
    #   和 last_action 不带。所以对拍必须关噪声，否则测的是噪声界不是装配误差。
    env_cfg.observations.policy.enable_corruption = False

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    deploy_path = os.path.abspath(os.path.join(os.path.dirname(resume_path), "params", "deploy.yaml"))
    print(f"[G0b] checkpoint : {resume_path}")
    print(f"[G0b] deploy.yaml: {deploy_path}")
    with open(deploy_path) as f:
        deploy = yaml.unsafe_load(f)

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length,
    )
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)
    N = env.num_envs
    device = env.device

    # ---- 关节顺序：IsaacLab 侧的实际顺序 ----
    sim_joint_names = list(robot.joint_names)
    print(f"\n[G0b] IsaacLab joint order ({len(sim_joint_names)}): {sim_joint_names}")
    print(f"[G0b] deploy.yaml default_joint_pos: {deploy['default_joint_pos']}")
    print(f"[G0b] deploy.yaml joint_ids_map    : {deploy['joint_ids_map']}")

    tr_default = robot.data.default_joint_pos[0].cpu()
    print(f"[G0b] sim default_joint_pos         : {[round(float(x), 4) for x in tr_default]}")

    # 用"默认位姿"作为关节身份指纹来判定排列：deploy.yaml 的 default_joint_pos 与
    # sim 的 default_joint_pos 是同一组物理值，只是顺序可能不同。
    dp = torch.tensor(deploy["default_joint_pos"], dtype=torch.float)
    cand = {}
    cand["identity(不换序)"] = torch.arange(12)
    a, b = _perm_from_joint_ids_map(deploy["joint_ids_map"])
    cand["deploy_from_train(joint_ids_map)"] = a
    cand["train_from_deploy(joint_ids_map)"] = b
    # 再从默认位姿本身反推一个（仅当两串值互为重排时才有唯一解）
    try:
        inv = torch.empty(12, dtype=torch.long)
        used = set()
        for i in range(12):
            hits = [j for j in range(12) if j not in used and abs(float(dp[i] - tr_default[j])) < 1e-6]
            if len(hits) != 1:
                raise ValueError
            inv[i] = hits[0]
            used.add(hits[0])
        cand["从 default_joint_pos 反推"] = inv
    except ValueError:
        print("[G0b] （default_joint_pos 两串值不构成唯一重排，无法据此反推排列）")

    # ---- 拼装规则（全部来自 deploy.yaml / observations.h）----
    obs_cfg = deploy["observations"]
    sc_ang = torch.tensor(obs_cfg["base_ang_vel"]["scale"], dtype=torch.float, device=device)
    sc_grav = torch.tensor(obs_cfg["projected_gravity"]["scale"], dtype=torch.float, device=device)
    sc_jpos = torch.tensor(obs_cfg["joint_pos_rel"]["scale"], dtype=torch.float, device=device)
    sc_jvel = torch.tensor(obs_cfg["joint_vel_rel"]["scale"], dtype=torch.float, device=device)
    sc_lact = torch.tensor(obs_cfg["last_action"]["scale"], dtype=torch.float, device=device)
    sc_cmd = torch.tensor(obs_cfg["velocity_commands"]["scale"], dtype=torch.float, device=device)
    action_scale = float(deploy["actions"]["JointPositionAction"]["scale"][0])

    # ---- 对拍累加器 ----
    n_hist = agent_cfg.history_length
    one_step = 45
    # 键 = "当前帧位置 | 关节排列候选"。
    # 当前帧在历史里的位置（头部还是尾部）**不能靠猜** —— 两种都算，分开报，
    # 哪一组全部对上就说明历史缓冲是那个方向。
    terms = ["velocity_commands", "base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel_rel", "last_action"]
    TERM_DIM = [3, 3, 3, 12, 12, 12]  # 与 terms 一一对应；合计 45
    layouts = {0: "当前帧@头(idx0)", n_hist: f"当前帧@尾(idx{n_hist})"}
    stats = {}
    for li in layouts:
        for name in cand:
            stats[(li, name)] = {t: 0.0 for t in terms}

    obs = env.get_observations()

    # 诊断：逐步留存两边的 45 维原始值（只留 identity 排列），事后定位 max 出现在哪一步
    diag_env, diag_mine = [], []
    _off = {}
    _o = 0
    for _t, _d in zip(terms, TERM_DIM):
        _off[_t] = _o
        _o += _d

    DUMP_AT = 50  # 在第 50 步（机器人已在运动）转储一次原始值
    # ★ 时序：必须"先 step 一步、再读状态、再与 obs 对拍"，否则会拿零初始化的
    #   history 缓冲去比（get_observations() 返回缓冲，而缓冲只在 step() 里更新）。
    actions = policy(obs)
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            obs, _, _, dones, *_ = env.step(actions)
            # ★ 有 env 在本步终止/reset 时，`robot.data` 已是 reset 后的新状态，而 obs
            #   里那一步仍是旧状态 —— 这一拍的对拍无效，跳过。否则会看到"单位向量差 3.8"
            #   这种物理上不可能的数字。
            if bool(dones.any()):
                actions = policy(obs)
                continue

            # ---- 训练侧：env 喂给策略的 obs ----
            # ★ 关键：270 维是**按项拼接**（term-major），不是按帧拼接。依据是 C++ 侧
            #   `deploy/include/isaaclab/manager/manager_term_cfg.h`：每个 obs 项各自维护
            #   一个 history 缓冲，最后 `obs.insert(obs.end(), obs_i.begin(), obs_i.end())`
            #   逐项串接。所以布局是 [cmd×6 | ang_vel×6 | grav×6 | jpos×6 | jvel×6 | act×6]。
            #   验证方式就是本脚本：若分块对，逐项差会掉到浮点噪声量级。
            # 帧优先布局（已在 himloco_vec_env_wrapper.py:280-286 确认）：
            #   obs = [帧t(45), 帧t-1(45), ..., 帧t-5(45)]，最新在头。
            # 所以先 reshape 成 (N, 6, 45)，再在每帧内部按项切。
            frames = obs.reshape(N, n_hist + 1, one_step)
            seg = {}
            _o2 = 0
            for _t, _d in zip(terms, TERM_DIM):
                seg[_t] = frames[:, :, _o2:_o2 + _d]
                _o2 += _d
            assert _o2 == one_step, f"分块不齐: {_o2} != {one_step}"
            # ---- 部署侧：从原始状态按 C++ 规则重拼 ----
            d = robot.data
            cmd = env.unwrapped.command_manager.get_command("base_velocity")
            raw_action = env.unwrapped.action_manager.action  # C++: action_manager->action()，raw
            ang = d.root_ang_vel_b
            grav = d.projected_gravity_b
            jp = d.joint_pos - d.default_joint_pos
            jv = d.joint_vel

            deploy_terms = {}
            deploy_terms["velocity_commands"] = cmd * sc_cmd
            deploy_terms["base_ang_vel"] = ang * sc_ang
            deploy_terms["projected_gravity"] = grav * sc_grav
            deploy_terms["last_action"] = raw_action * sc_lact

            if step == DUMP_AT:
                print("\n" + "=" * 100)
                print("[G0b][DUMP] 第 0 步原始值对照（env 第 0 个环境）")
                print("=" * 100)
                print(f"env obs 总维度 = {obs.shape[1]}  (num_one_step_obs=45, history+1={n_hist + 1})")
                print(f"  obs[0][ 0: 3] (帧0 cmd ？)   = {obs[0, 0:3].tolist()}")
                print(f"  obs[0][ 3: 6] (帧0 ang ？)   = {obs[0, 3:6].tolist()}")
                print(f"  obs[0][ 6: 9] (帧0 grav ？)  = {obs[0, 6:9].tolist()}")
                print(f"  obs[0][ 9:21] (帧0 jpos ？)  = {[round(x, 4) for x in obs[0, 9:21].tolist()]}")
                print(f"  obs[0][21:33] (帧0 jvel ？)  = {[round(x, 4) for x in obs[0, 21:33].tolist()]}")
                print(f"  obs[0][33:45] (帧0 act ？)   = {[round(x, 4) for x in obs[0, 33:45].tolist()]}")
                print(f"  obs[0][45:48] (帧1 cmd ？)   = {obs[0, 45:48].tolist()}")
                print("\n部署侧重拼：")
                print(f"  velocity_commands = {deploy_terms['velocity_commands'][0].tolist()}"
                      f"   (cmd from command_manager = {cmd[0].tolist()})")
                print(f"  base_ang_vel      = {deploy_terms['base_ang_vel'][0].tolist()}"
                      f"   (root_ang_vel_b = {ang[0].tolist()})")
                print(f"  projected_gravity = {deploy_terms['projected_gravity'][0].tolist()}"
                      f"   (projected_gravity_b = {grav[0].tolist()})")
                print(f"  joint_pos_rel     = {[round(x, 4) for x in (jp[0] * sc_jpos).tolist()]}")
                print(f"  joint_vel_rel     = {[round(x, 4) for x in (jv[0] * sc_jvel).tolist()]}")
                print(f"  last_action       = {[round(x, 4) for x in (raw_action[0] * sc_lact).tolist()]}")
                print(f"  sim default_joint_pos = {[round(float(x), 4) for x in tr_default]}")
                print("=" * 100 + "\n")

            # 诊断留存：env 侧帧 0（帧优先布局下就是前 45 维）+ 我方 identity 排列的各项
            _ident = torch.arange(12, device=device)
            diag_env.append(obs[:, 0:45].detach().clone())
            diag_mine.append({
                "velocity_commands": (cmd * sc_cmd).detach().clone(),
                "base_ang_vel": (ang * sc_ang).detach().clone(),
                "projected_gravity": (grav * sc_grav).detach().clone(),
                "joint_pos_rel": (jp[:, _ident] * sc_jpos).detach().clone(),
                "joint_vel_rel": (jv[:, _ident] * sc_jvel).detach().clone(),
                "last_action": (raw_action * sc_lact).detach().clone(),
            })

            for cur_idx in layouts:
                for name, perm in cand.items():
                    p = perm.to(device)
                    deploy_terms["joint_pos_rel"] = jp[:, p] * sc_jpos
                    deploy_terms["joint_vel_rel"] = jv[:, p] * sc_jvel
                    for t in terms:
                        dd = (seg[t][:, cur_idx, :] - deploy_terms[t]).abs().max().item()
                        stats[(cur_idx, name)][t] = max(stats[(cur_idx, name)][t], dd)
            actions = policy(obs)

    # ---- 报告 ----
    print("\n" + "=" * 100)
    print("[G0b] 部署侧重拼 obs  vs  IsaacLab 喂给策略的 obs —— 每项最大绝对差")
    print(f"      样本：{N} envs x {args_cli.num_steps} steps；action_scale={action_scale}")
    print("=" * 100)
    hdr = f"{'当前帧位置 | 关节排列候选':44s}" + "".join(f"{t[:12]:>14s}" for t in terms)
    print(hdr)
    print("-" * len(hdr))
    for li, lname in layouts.items():
        for name in cand:
            row = f"{lname + ' | ' + name:44s}"
            for t in terms:
                row += f"{stats[(li, name)][t]:14.3e}"
            print(row)
        print("-" * len(hdr))

    # ---- 诊断：max 出现在哪一步，那一步两边各是什么 ----
    print("\n" + "=" * 100)
    print("[G0b][DIAG] max 出现在哪一步（identity 排列 / 帧优先·当前帧在头）")
    print("=" * 100)
    if diag_env:
        E = torch.stack(diag_env).cpu()               # (S, N, 45) env 侧真值
        for t in terms:
            d_ = TERM_DIM[terms.index(t)]
            M = torch.stack([m[t] for m in diag_mine]).cpu()  # (S, N, d_)
            Ev = E[:, :, _off[t]:_off[t] + d_]
            diff = (Ev - M).abs()
            mx = diff.max()
            if mx <= 1e-6:
                print(f"  {t:18s} max={mx:.2e}  （完全一致）")
                continue
            s, n, k = [int(x) for x in torch.nonzero(diff == mx)[0]]
            print(f"  {t:18s} max={mx:.4f} @ step={s} env={n} dim={k}   "
                  f"env={Ev[s, n, k]:+.6f}  mine={M[s, n, k]:+.6f}")
        # 把偏离最大的那一步整帧打出来
        worst_t = max(terms, key=lambda t: (E[:, :, _off[t]:_off[t] + TERM_DIM[terms.index(t)]]
                                            - torch.stack([m[t] for m in diag_mine]).cpu()).abs().max())
        sw, nw, kw = [int(x) for x in torch.nonzero(
            (E[:, :, _off[worst_t]:_off[worst_t] + TERM_DIM[terms.index(worst_t)]]
             - torch.stack([m[worst_t] for m in diag_mine]).cpu()).abs()
            == (E[:, :, _off[worst_t]:_off[worst_t] + TERM_DIM[terms.index(worst_t)]]
               - torch.stack([m[worst_t] for m in diag_mine]).cpu()).abs().max())[0]]
        print(f"\n  偏离最大的一步：step={sw} env={nw}（项 {worst_t}）—— 该步整帧 45 维对照：")
        print(f"    env  : {[round(float(x), 4) for x in E[sw, nw].tolist()]}")
        mine45 = torch.cat([diag_mine[sw][t][nw] for t in terms]).cpu()
        print(f"    mine : {[round(float(x), 4) for x in mine45.tolist()]}")
        # 前后各一步，看是不是时间错位
        for ds in (-1, +1):
            i2 = sw + ds
            if 0 <= i2 < E.shape[0]:
                m2 = torch.cat([diag_mine[i2][t][nw] for t in terms]).cpu()
                dd = (E[sw, nw] - m2).abs().max().item()
                print(f"    （若把 mine 换成 step={i2} 的值，与环境该步的 max 差 = {dd:.6f}）")

    print("\n判定提示：")
    print("  - 某一行的 6 项**全部** < 1e-4  → 该关节排列正确，且拼装与训练一致")
    print("  - joint_pos_rel/joint_vel_rel 大而其余项小 → **关节顺序**问题（§8.5 bug #2）")
    print("  - last_action 恰好差 ~0.25x 量级        → **last_action 乘了 action_scale**（bug #1）")
    print("  - 全项都大                              → 装配规则整体不对，先查 deploy.yaml 是否对应当前 checkpoint")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
