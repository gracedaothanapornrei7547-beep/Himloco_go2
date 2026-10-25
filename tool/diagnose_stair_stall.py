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

os.environ["HIMLOCO_STAIR_H"] = "0.10"
os.environ["HIMLOCO_EVAL_KEEP_DR"] = "0"

spec = gym.spec("Unitree-Go2-Stair-Stage-Play")
import importlib
mod_name, cls_name = spec.kwargs["env_cfg_entry_point"].rsplit(":", 1)
mod = importlib.import_module(mod_name)
cfg_cls = getattr(mod, cls_name)
cfg = cfg_cls()
cfg.scene.num_envs = 1

# Make sure it faces the stairs
# pyramid_stairs_inv is up stairs
cfg.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
cfg.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 1.0
cfg.commands.base_velocity.heading_command = True
cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

env = gym.make("Unitree-Go2-Stair-Stage-Play", cfg=cfg)
agent_cfg = PPORunnerCfg()
env = HimlocoVecEnvWrapper(env, history_length=agent_cfg.history_length, privileged_history_length=agent_cfg.privileged_history_length)

ckpt = "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-10-14_16-33-22_p70_himloco_stage2_stairs_1500/model_2498.pt"
runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(ckpt)
policy = runner.get_inference_policy(device="cuda:0")

robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

obs = env.get_observations()
print("Starting 120-step diagnosis on 10cm stairs...")
for step in range(120):
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, dones, _, _, _ = env.step(act)
    
    pos = robot.data.root_pos_w[0].cpu().numpy()
    vel = robot.data.root_lin_vel_b[0].cpu().numpy()
    rpy = robot.data.root_euler_w[0].cpu().numpy()
    # Feet positions relative to world
    feet_pos = robot.data.body_pos_w[0, [4, 8, 14, 18]].cpu().numpy() # FL, FR, RL, RR
    forces = contact_sensor.data.net_forces_w[0, :, 2].cpu().numpy()
    
    if step % 10 == 0 or vel[0] < 0.1:
        print(f"Step {step:3d}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f} | vx={vel[0]:.2f}, vz={vel[2]:.2f} | pitch={rpy[1]:.2f} rad | feet_z={feet_pos[:, 2].round(2)}")
        if vel[0] < 0.1 and step > 30:
            print(f"   --> STALL DETECTED at step {step}: act={act[0].cpu().numpy().round(2)}")

print("Diagnosis finished.")
simulation_app.close()

