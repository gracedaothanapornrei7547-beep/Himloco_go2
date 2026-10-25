#!/usr/bin/env python3
"""Test Project-Robo/HIMLOCO_GO2's pre_train/go2/policy.pt in MuJoCo Sim2Sim."""

import os
import sys
import numpy as np
import torch
import mujoco

sys.path.insert(0, "/home/rum/himloco_lab/deploy/sim2sim")
from sim2sim_mujoco import Go2MuJoCoSim2Sim, quat_apply_inverse, quat_to_euler_rpy

class ProjectRoboSim2Sim(Go2MuJoCoSim2Sim):
    def __init__(self, scene_path, policy_pt_path, kp=20.0, kd=0.5, torque_limit=23.7):
        # Initialize base MuJoCo model
        super().__init__(scene_path, "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/exported_best_13600", kp=kp, kd=kd, torque_limit=torque_limit)
        
        # Load Project-Robo's TorchScript policy
        self.device = torch.device("cpu")
        self.policy = torch.jit.load(policy_pt_path, map_location=self.device)
        self.policy.eval()
        print(f"[ProjectRobo] Successfully loaded TorchScript policy from {policy_pt_path}")

    def run_policy(self, obs_history_270):
        # Convert to tensor and run
        obs_tensor = torch.from_numpy(obs_history_270).float().unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)
        return action_tensor.squeeze(0).numpy()

def main():
    scene_path = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2/scene_stairs_10cm.xml"
    policy_path = "/home/rum/reference_repos/HIMLOCO_GO2/legged_gym/deploy/pre_train/go2/policy.pt"
    
    sim = ProjectRoboSim2Sim(scene_path, policy_path, kp=20.0, kd=0.5)
    
    # Run simulation with 0.8 m/s command
    vx_cmd = 0.8
    duration = 10.0
    dt_policy = 0.02
    substeps = 10
    total_steps = int(duration / dt_policy)
    
    # Observation buffer
    obs_history = np.zeros(270, dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)
    
    print("\nStarting Project-Robo simulation on 10cm stairs...")
    for step in range(total_steps):
        # 1. Get state
        quat = sim.data.qpos[3:7]  # [w, x, y, z]
        base_pos = sim.data.qpos[:3]
        base_lin_vel_w = sim.data.qvel[:3]
        base_ang_vel_w = sim.data.qvel[3:6]
        
        base_lin_vel_b = quat_apply_inverse(quat, base_lin_vel_w)
        base_ang_vel_b = quat_apply_inverse(quat, base_ang_vel_w)
        proj_gravity = quat_apply_inverse(quat, np.array([0, 0, -1.0]))
        
        # Joint positions & velocities (Isaac ordering)
        qpos_mujoco = sim.data.qpos[7:19]
        qvel_mujoco = sim.data.qvel[6:18]
        qpos_isaac = qpos_mujoco[sim.mujoco_to_isaac]
        qvel_isaac = qvel_mujoco[sim.mujoco_to_isaac]
        
        dof_pos_rel = qpos_isaac - sim.default_joint_pos_isaac
        dof_vel_rel = qvel_isaac
        
        # 2. Build 45-dim obs
        obs_cmd = np.array([vx_cmd * 2.0, 0.0, 0.0], dtype=np.float32)
        obs_ang_vel = base_ang_vel_b * 0.25
        obs_gravity = proj_gravity
        obs_dof_pos = dof_pos_rel * 1.0
        obs_dof_vel = dof_vel_rel * 0.05
        obs_last_act = last_action
        
        obs_1step = np.concatenate([
            obs_cmd, obs_ang_vel, obs_gravity, obs_dof_pos, obs_dof_vel, obs_last_act
        ]).astype(np.float32)
        
        # Roll FIFO
        obs_history = np.concatenate([obs_1step, obs_history[:-45]])
        
        # 3. Policy step
        action_isaac = sim.run_policy(obs_history)
        last_action = action_isaac.copy()
        
        # 4. Convert action to target position & PD in MuJoCo
        action_mujoco = action_isaac[sim.isaac_to_mujoco]
        target_pos = sim.default_joint_pos_mujoco + 0.25 * action_mujoco
        
        for _ in range(substeps):
            cur_pos = sim.data.qpos[7:19]
            cur_vel = sim.data.qvel[6:18]
            tau = sim.kp * (target_pos - cur_pos) - sim.kd * cur_vel
            tau = np.clip(tau, -23.7, 23.7)
            sim.data.ctrl[:] = tau
            mujoco.mj_step(sim.model, sim.data)
            
        rpy = quat_to_euler_rpy(quat)
        if step % 25 == 0:
            print(f"[{step*dt_policy:5.2f}s] Pos: x={base_pos[0]:5.2f}, y={base_pos[1]:5.2f}, z={base_pos[2]:5.2f}m | Vel: vx={base_lin_vel_b[0]:5.2f}m/s | Pitch: {rpy[1]:+5.1f}°")

if __name__ == "__main__":
    main()

