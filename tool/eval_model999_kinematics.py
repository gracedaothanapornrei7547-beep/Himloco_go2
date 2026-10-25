import os
import sys
import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import himloco_lab.tasks
from himloco_lab.rsl_rl.wrappers import HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.runners import HIMOnPolicyRunner

ckpt = "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-10-14_13-52-18_p69_himloco_flat_surgical_1000/model_999.pt"
print("Loading checkpoint:", ckpt)

from himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg import PPORunnerCfg
agent_cfg = PPORunnerCfg()
spec = gym.spec("Unitree-Go2-Flat-Play")
import importlib
mod_name, cls_name = spec.kwargs["env_cfg_entry_point"].rsplit(":", 1)
mod = importlib.import_module(mod_name)
cfg_cls = getattr(mod, cls_name)
cfg = cfg_cls()
cfg.scene.num_envs = 4

env = gym.make("Unitree-Go2-Flat-Play", cfg=cfg)
env = HimlocoVecEnvWrapper(env, history_length=agent_cfg.history_length, privileged_history_length=agent_cfg.privileged_history_length)

runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(ckpt)
policy = runner.get_inference_policy(device="cuda:0")
robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

obs = env.get_observations()
cmd = env.unwrapped.command_manager.get_command("base_velocity")

print(f"=== EVALUATION OF MODEL_999 (Surgical Flat 1000 iters) ===")
print(f"Command target: vx={cmd[0, 0].item():.2f}, vy={cmd[0, 1].item():.2f}, wz={cmd[0, 2].item():.2f}")

vx_list = []
vz_list = []
height_list = []

for step in range(100):
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, _, _, _, _ = env.step(act)
    lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    air_time = contact_sensor.data.last_air_time[0].cpu().numpy()
    in_contact = (contact_sensor.data.net_forces_w[0, :, 2] > 1.0).cpu().numpy()
    
    vx_list.append(lin_vel[0])
    vz_list.append(lin_vel[2])
    height_list.append(root_pos[2])
    if step % 20 == 0 or step == 99:
        print(f"Step {step:2d}: vx={lin_vel[0]:.3f} m/s, root_z={root_pos[2]:.3f} m, act_norm={torch.norm(act[0]).item():.2f}")
        print(f"         air_time={air_time[:4].round(2)}, contact={in_contact[:4]}")
        print(f"         thighs={joint_pos[[1,4,7,10]].round(2)}, calves={joint_pos[[2,5,8,11]].round(2)}")

print("\n--- 100-Step Summary ---")
print(f"Mean vx: {sum(vx_list)/len(vx_list):.3f} m/s (Target was {cmd[0, 0].item():.2f} m/s)")
print(f"Mean root_z: {sum(height_list)/len(height_list):.3f} m (Nominal is 0.33m)")
print(f"Mean |vz|: {sum(abs(v) for v in vz_list)/len(vz_list):.3f} m/s")
simulation_app.close()
