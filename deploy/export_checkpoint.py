#!/usr/bin/env python3
"""
Lightweight Checkpoint Exporter for HimLoco.
Exports HimLoco PyTorch checkpoints to TorchScript JIT (policy.pt) and ONNX (encoder.onnx, policy.onnx)
without requiring Isaac Sim / Omniverse runtime.
"""

import argparse
import importlib.util
import os
import sys
import types
import numpy as np
import torch
import onnxruntime as ort

def load_himloco_modules(base_dir):
    """Dynamically import HIMActorCritic and export utilities without loading IsaacLab pxr."""
    p1 = types.ModuleType("himloco_lab")
    p2 = types.ModuleType("himloco_lab.rsl_rl")
    p3 = types.ModuleType("himloco_lab.rsl_rl.modules")
    sys.modules["himloco_lab"] = p1
    sys.modules["himloco_lab.rsl_rl"] = p2
    sys.modules["himloco_lab.rsl_rl.modules"] = p3

    spec_est = importlib.util.spec_from_file_location(
        "himloco_lab.rsl_rl.modules.him_estimator",
        os.path.join(base_dir, "rsl_rl/modules/him_estimator.py")
    )
    est_mod = importlib.util.module_from_spec(spec_est)
    sys.modules["himloco_lab.rsl_rl.modules.him_estimator"] = est_mod
    spec_est.loader.exec_module(est_mod)

    spec_ac = importlib.util.spec_from_file_location(
        "himloco_lab.rsl_rl.modules.him_actor_critic",
        os.path.join(base_dir, "rsl_rl/modules/him_actor_critic.py")
    )
    ac_mod = importlib.util.module_from_spec(spec_ac)
    sys.modules["himloco_lab.rsl_rl.modules.him_actor_critic"] = ac_mod
    spec_ac.loader.exec_module(ac_mod)

    spec_exp = importlib.util.spec_from_file_location(
        "export_policy",
        os.path.join(base_dir, "utils/export_policy.py")
    )
    exp_mod = importlib.util.module_from_spec(spec_exp)
    spec_exp.loader.exec_module(exp_mod)

    return ac_mod.HIMActorCritic, exp_mod

def export_checkpoint(checkpoint_path, output_dir=None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if output_dir is None:
        output_dir = os.path.join(run_dir, "exported")
    os.makedirs(output_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    base_dir = os.path.join(repo_root, "source/himloco_lab/himloco_lab")

    HIMActorCritic, exp_mod = load_himloco_modules(base_dir)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    num_actor_obs = state_dict["estimator.encoder.0.weight"].shape[1]
    num_actions = state_dict["actor.6.weight"].shape[0]
    num_critic_obs = state_dict["critic.0.weight"].shape[1]
    # Infer num_one_step_obs dynamically: actor input is (num_one_step_obs + 3 [vel] + 16 [latent])
    num_one_step_obs = state_dict["actor.0.weight"].shape[1] - 19
    print(f"[INFO] Inferred observation topology: num_one_step_obs={num_one_step_obs}, num_actor_obs={num_actor_obs}")

    actor_critic = HIMActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_one_step_obs=num_one_step_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()

    print(f"[INFO] Exporting TorchScript JIT and ONNX to: {output_dir}")
    exp_mod.export_himloco_policy_as_jit(actor_critic, output_dir, "policy.pt")
    exp_mod.export_himloco_policy_as_onnx(actor_critic, output_dir, "encoder.onnx", "policy.onnx")

    # Verify ONNX
    enc_path = os.path.join(output_dir, "encoder.onnx")
    pol_path = os.path.join(output_dir, "policy.onnx")

    sess_opts = ort.SessionOptions()
    enc_sess = ort.InferenceSession(enc_path, sess_opts, providers=["CPUExecutionProvider"])
    pol_sess = ort.InferenceSession(pol_path, sess_opts, providers=["CPUExecutionProvider"])

    dummy_hist = np.zeros((1, num_actor_obs), dtype=np.float32)
    enc_out = enc_sess.run(None, {"obs_history": dummy_hist})[0]

    dummy_curr = np.zeros((1, num_one_step_obs), dtype=np.float32)
    pol_in = np.concatenate([dummy_curr, enc_out], axis=-1)
    act_out = pol_sess.run(None, {"obs": pol_in})[0]

    print(f"[SUCCESS] Export completed and verified!")
    print(f"          Encoder output shape: {enc_out.shape}")
    print(f"          Policy action shape:  {act_out.shape}")
    print(f"          Files written: {os.listdir(output_dir)}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Export HimLoco checkpoint to JIT and ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to checkpoint_dir/exported)")
    args = parser.parse_args()

    export_checkpoint(args.checkpoint, args.output_dir)

if __name__ == "__main__":
    main()

