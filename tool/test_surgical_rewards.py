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
from himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg import PPORunnerCfg

agent_cfg = PPORunnerCfg()
spec = gym.spec("Unitree-Go2-Velocity")
import importlib
mod_name, cls_name = spec.kwargs["env_cfg_entry_point"].rsplit(":", 1)
mod = importlib.import_module(mod_name)
cfg_cls = getattr(mod, cls_name)
cfg = cfg_cls()
cfg.scene.num_envs = 4

env = gym.make("Unitree-Go2-Velocity", cfg=cfg)

obs, _ = env.reset()
actions = torch.zeros(4, 12, device=env.unwrapped.device)
step_ret = env.step(actions)
rew = step_ret[1]
print("\n>>> SUCCESS! Single-step reward tensor for Velocity env:", rew)
print("Active reward terms:", env.unwrapped.reward_manager._term_names)
simulation_app.close()
