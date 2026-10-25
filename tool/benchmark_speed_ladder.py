import os
import sys
import argparse
import torch

parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import himloco_lab.tasks
from himloco_lab.rsl_rl.wrappers import HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.runners import HIMOnPolicyRunner
from himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg import PPORunnerCfg

ckpt = "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-10-14_20-06-04_p71_himloco_flat_fast_stride_1500/model_2200.pt"

spec = gym.spec("Unitree-Go2-Flat-Play")
import importlib
mod_name, cls_name = spec.kwargs["env_cfg_entry_point"].rsplit(":", 1)
mod = importlib.import_module(mod_name)
cfg_cls = getattr(mod, cls_name)
cfg = cfg_cls()
cfg.scene.num_envs = 16

env = gym.make("Unitree-Go2-Flat-Play", cfg=cfg)
agent_cfg = PPORunnerCfg()
env = HimlocoVecEnvWrapper(env, history_length=agent_cfg.history_length, privileged_history_length=agent_cfg.privileged_history_length)

runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(ckpt)
policy = runner.get_inference_policy(device="cuda:0")
robot = env.unwrapped.scene["robot"]

test_cmds = [1.0, 1.5, 1.8, 2.0, 2.2]
results = {}

for cmd_val in test_cmds:
    env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:, 0] = cmd_val
    env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:, 1] = 0.0
    env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:, 2] = 0.0
    
    obs = env.get_observations()
    vx_history = []
    
    # Run 100 control steps (2.0s), steady state in last 60 steps
    for step in range(100):
        with torch.inference_mode():
            act = policy(obs)
            obs, _, _, _, _, _, _ = env.step(act)
        vx = robot.data.root_lin_vel_b[:, 0].clone()
        if step >= 40:
            vx_history.append(vx)
            
    vx_all = torch.stack(vx_history).cpu().numpy()
    mean_vx = float(vx_all.mean())
    max_vx = float(vx_all.max())
    p90_vx = float(torch.quantile(torch.tensor(vx_all), 0.90).item())
    height = float(robot.data.root_pos_w[:, 2].mean().item())
    
    results[cmd_val] = {
        "mean_vx": mean_vx,
        "max_vx": max_vx,
        "p90_vx": p90_vx,
        "height": height,
        "ratio": mean_vx / cmd_val
    }
    print(f"[CMD {cmd_val:.1f} m/s] Achieved Mean={mean_vx:.3f} m/s, Max={max_vx:.3f} m/s, P90={p90_vx:.3f} m/s, Height={height:.3f} m, Ratio={mean_vx/cmd_val*100:.1f}%")

print("\n=== FINAL BENCHMARK SUMMARY FOR MODEL_2200 ===")
for cmd_val, res in results.items():
    print(f"Target: {cmd_val:.1f} m/s -> Mean: {res['mean_vx']:.3f} m/s | Max: {res['max_vx']:.3f} m/s | P90: {res['p90_vx']:.3f} m/s | Ratio: {res['ratio']*100:.1f}%")

simulation_app.close()

