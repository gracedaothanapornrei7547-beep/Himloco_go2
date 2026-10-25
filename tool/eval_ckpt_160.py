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

latest_run = sorted([d for d in os.listdir("logs/himloco_rsl_rl/go2_rough") if "fixed" in d])[-1]
ckpt = os.path.join("logs/himloco_rsl_rl/go2_rough", latest_run, "model_160.pt")
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

obs = env.get_observations()
cmd = env.unwrapped.command_manager.get_command("base_velocity")

print(f"--- EVAL CHECKPOINT 160 (Target Command: {cmd[0].cpu().numpy().round(2)}) ---")
for step in range(50):
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, _, _, _, _ = env.step(act)
    if step % 5 == 0 or step == 49:
        lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        joint_pos = robot.data.joint_pos[0].cpu().numpy()
        print(f"Step {step:02d}: root_z={root_pos[2]:.3f}, vx={lin_vel[0]:.3f}, vy={lin_vel[1]:.3f}, act_norm={torch.norm(act[0]).item():.2f}")
        print(f"         calves={joint_pos[8:12].round(2)}")

simulation_app.close()
os._exit(0)

