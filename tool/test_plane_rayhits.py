import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app = AppLauncher(args_cli)

import gymnasium as gym
import himloco_lab.tasks

spec = gym.spec("Unitree-Go2-Flat-Play")
import importlib
mod_name, cls_name = spec.kwargs["env_cfg_entry_point"].rsplit(":", 1)
cfg = getattr(importlib.import_module(mod_name), cls_name)()
cfg.scene.num_envs = 2

env = gym.make("Unitree-Go2-Flat-Play", cfg=cfg)
env.reset()
sensor = env.unwrapped.scene.sensors["base_height_scanner"]
hits = sensor.data.ray_hits_w
print("Ray hits shape:", hits.shape)
print("Ray hits values (first 5 points):", hits[0, :5].cpu().numpy())
print("Is NaN:", torch.isnan(hits).any().item())
print("Is Inf:", torch.isinf(hits).any().item())

robot = env.unwrapped.scene["robot"]
print("root_link_pos_w:", robot.data.root_link_pos_w.cpu().numpy())
print("root_pos_w:", robot.data.root_pos_w.cpu().numpy())

# test mdp.base_height
from himloco_lab.tasks.locomotion import mdp
from isaaclab.managers import SceneEntityCfg
bh = mdp.base_height(env.unwrapped, target_height=0.33, sensor_cfg=SceneEntityCfg("base_height_scanner"))
print("mdp.base_height output:", bh.cpu().numpy())

app.app.close()

