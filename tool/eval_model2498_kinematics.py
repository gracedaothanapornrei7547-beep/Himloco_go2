import os
import sys
import torch
from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import himloco_lab.tasks
from himloco_lab.rsl_rl.wrappers import HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.runners import HIMOnPolicyRunner
from himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg import PPORunnerCfg

ckpt = "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-10-14_16-33-22_p70_himloco_stage2_stairs_1500/model_2498.pt"
print("Loading checkpoint:", ckpt)

agent_cfg = PPORunnerCfg()
spec = gym.spec("Unitree-Go2-Velocity-Play")
import importlib
mod_name, cls_name = spec.kwargs["env_cfg_entry_point"].rsplit(":", 1)
mod = importlib.import_module(mod_name)
cfg_cls = getattr(mod, cls_name)
cfg = cfg_cls()
cfg.scene.num_envs = 4

env = gym.make("Unitree-Go2-Velocity-Play", cfg=cfg)
env = HimlocoVecEnvWrapper(env, history_length=agent_cfg.history_length, privileged_history_length=agent_cfg.privileged_history_length)

runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(ckpt)
policy = runner.get_inference_policy(device="cuda:0")
robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

obs = env.get_observations()
cmd = env.unwrapped.command_manager.get_command("base_velocity")

print(f"=== EVALUATION OF MODEL_2498 on Velocity-Play ===")
print(f"Command target: vx={cmd[0, 0].item():.2f}, vy={cmd[0, 1].item():.2f}, wz={cmd[0, 2].item():.2f}")

vx_list, vz_list, height_list = [], [], []

for step in range(100):
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, dones, _, _, _ = env.step(act)
    lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    air_time = contact_sensor.data.last_air_time[0].cpu().numpy()
    in_contact = (contact_sensor.data.net_forces_w[0, :, 2] > 1.0).cpu().numpy()
    
    vx_list.append(lin_vel[0])
    vz_list.append(lin_vel[2])
    height_list.append(root_pos[2])
    if step % 20 == 0 or step == 99:
        print(f"Step {step:2d}: vx={lin_vel[0]:.3f} m/s, root_pos={root_pos.round(3)}, act_norm={torch.norm(act[0]).item():.2f}")
        print(f"         air_time={air_time[:4].round(2)}, contact={in_contact[:4]}")

print("\n--- Summary ---")
print(f"Mean vx: {sum(vx_list)/len(vx_list):.3f} m/s (Target: {cmd[0, 0].item():.2f} m/s)")
print(f"Mean root_z: {sum(height_list)/len(height_list):.3f} m")
print(f"Net displacement X: {root_pos[0]:.3f} m, Y: {root_pos[1]:.3f} m, Z: {root_pos[2]:.3f} m")
simulation_app.close()
