#!/usr/bin/env python3
import os
import sys
import copy
import importlib.util
import torch
import torch.nn as nn

# Load him_actor_critic.py directly by file path
spec = importlib.util.spec_from_file_location(
    "him_actor_critic",
    "/home/rum/himloco_lab/source/himloco_lab/himloco_lab/rsl_rl/modules/him_actor_critic.py"
)
hac_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hac_mod)
HIMActorCritic = hac_mod.HIMActorCritic

class _HimlocoEncoderOnnxExporter(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = copy.deepcopy(actor_critic.estimator.encoder)

    def forward(self, obs_history):
        parts = self.encoder(obs_history)
        vel = parts[:, :3]
        latent = parts[:, 3:]
        latent = torch.nn.functional.normalize(latent, dim=1, p=2)
        return torch.cat([vel, latent], dim=-1)

class _HimlcoPolicyOnnxExporter(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)

    def forward(self, obs):
        return self.actor(obs)

def main():
    run_dir = "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-09-23_22-20-12_p37_straight_leg_anti_circumduction"
    ckpt_path = os.path.join(run_dir, "model_14051.pt")
    export_dir = os.path.join(run_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    net = HIMActorCritic(num_actor_obs=270, num_critic_obs=238, num_one_step_obs=45, num_actions=12)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # 1. Export encoder.onnx
    enc_exporter = _HimlocoEncoderOnnxExporter(net)
    dummy_hist = torch.zeros(1, 270, dtype=torch.float32)
    enc_path = os.path.join(export_dir, "encoder.onnx")
    torch.onnx.export(
        enc_exporter,
        dummy_hist,
        enc_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["obs_history"],
        output_names=["estimator_output"],
        dynamic_axes={"obs_history": {0: "batch_size"}, "estimator_output": {0: "batch_size"}},
    )
    print(f"[Export] Saved encoder.onnx -> {enc_path}")

    # 2. Export policy.onnx
    pol_exporter = _HimlcoPolicyOnnxExporter(net)
    dummy_obs = torch.zeros(1, 64, dtype=torch.float32)
    pol_path = os.path.join(export_dir, "policy.onnx")
    torch.onnx.export(
        pol_exporter,
        dummy_obs,
        pol_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
    )
    print(f"[Export] Saved policy.onnx -> {pol_path}")

if __name__ == "__main__":
    main()

