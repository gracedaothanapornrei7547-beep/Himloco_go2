#!/usr/bin/env python3
"""
Extract joint angles and foot telemetry from MuJoCo on flat ground and stairs.
Quantifies hip joint adduction/abduction ("内八" vs "外八").
"""

import os
import sys
import numpy as np

# Add himloco_lab deploy path
sys.path.append("/home/rum/himloco_lab/deploy/sim2sim")
from sim2sim_mujoco import Go2MuJoCoSim2Sim, quat_to_euler_rpy

import glob

ASSETS_DIR = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"

def get_latest_model_dir():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return sys.argv[1]
    dirs = sorted(glob.glob("/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/*p73*"))
    if dirs:
        return dirs[-1]
    return "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-10-14_21-43-56_p72_himloco_stage2_stairs_assault_3000"

MODEL_DIR = get_latest_model_dir()
print(f"[Model Under Test]: {MODEL_DIR}")

def analyze_scenario(scene_name, vx=0.8, duration=4.0):
    scene_path = os.path.join(ASSETS_DIR, f"{scene_name}.xml")
    env = Go2MuJoCoSim2Sim(scene_path=scene_path, model_dir=MODEL_DIR)
    cmd = np.array([vx, 0.0, 0.0], dtype=np.float32)
    num_steps = int(duration / env.policy_dt)

    # Track joints: IsaacLab order is [FL_hip, FR_hip, RL_hip, RR_hip, ...]
    hip_angles = [] # (N, 4) [FL, FR, RL, RR]
    foot_pos_body = [] # (N, 4, 3) foot positions in body frame
    thigh_angles = [] # (N, 4)
    calf_angles = [] # (N, 4)

    # Let robot settle
    for step_idx in range(num_steps):
        info = env.step(cmd)
        q_mj = env.data.qpos[7:19]
        q_isaac = q_mj[env.mujoco_to_isaac]
        
        # [FL_hip, FR_hip, RL_hip, RR_hip]
        hip_angles.append(q_isaac[0:4].copy())
        thigh_angles.append(q_isaac[4:8].copy())
        calf_angles.append(q_isaac[8:12].copy())

    hip_angles = np.array(hip_angles)
    thigh_angles = np.array(thigh_angles)
    calf_angles = np.array(calf_angles)

    print(f"\n=======================================================")
    print(f"  SCENARIO: {scene_name} (vx={vx} m/s, duration={duration}s)")
    print(f"=======================================================")
    print(f"Nominal Hip Angles: FL=+0.10, FR=-0.10, RL=+0.10, RR=-0.10 rad")
    print(f"Hip Angles (Mean ± Std rad):")
    print(f"  FL_hip: {hip_angles[:, 0].mean():+.3f} ± {hip_angles[:, 0].std():.3f} rad  (deg: {np.rad2deg(hip_angles[:, 0].mean()):+.1f}°)")
    print(f"  FR_hip: {hip_angles[:, 1].mean():+.3f} ± {hip_angles[:, 1].std():.3f} rad  (deg: {np.rad2deg(hip_angles[:, 1].mean()):+.1f}°)")
    print(f"  RL_hip: {hip_angles[:, 2].mean():+.3f} ± {hip_angles[:, 2].std():.3f} rad  (deg: {np.rad2deg(hip_angles[:, 2].mean()):+.1f}°)")
    print(f"  RR_hip: {hip_angles[:, 3].mean():+.3f} ± {hip_angles[:, 3].std():.3f} rad  (deg: {np.rad2deg(hip_angles[:, 3].mean()):+.1f}°)")
    
    # Calculate Stance Width (FL to FR, RL to RR)
    # Hip deviation from nominal:
    dev_fl = hip_angles[:, 0] - 0.10
    dev_fr = hip_angles[:, 1] - (-0.10)
    dev_rl = hip_angles[:, 2] - 0.10
    dev_rr = hip_angles[:, 3] - (-0.10)
    
    # Splay analysis:
    # Front legs: FL - FR. Nominal = 0.10 - (-0.10) = 0.20 rad.
    # If FL - FR < 0.20 -> inward (内八); if FL - FR > 0.20 -> outward (外八)
    front_spread = hip_angles[:, 0] - hip_angles[:, 1]
    rear_spread = hip_angles[:, 2] - hip_angles[:, 3]
    print(f"\nSpread Angle (Left - Right, Nominal = +0.200 rad / +11.5°):")
    print(f"  Front Spread: {front_spread.mean():+.3f} rad ({np.rad2deg(front_spread.mean()):+.1f}°) [Min: {front_spread.min():+.3f}, Max: {front_spread.max():+.3f}]")
    print(f"  Rear Spread:  {rear_spread.mean():+.3f} rad ({np.rad2deg(rear_spread.mean()):+.1f}°) [Min: {rear_spread.min():+.3f}, Max: {rear_spread.max():+.3f}]")
    
    if front_spread.mean() < 0.12:
        print(f"  --> [DIAGNOSIS] Severe FRONT 内八 (pigeon-toed adduction): spread collapsed to {np.rad2deg(front_spread.mean()):.1f}° (nominal 11.5°)")
    elif front_spread.mean() > 0.35:
        print(f"  --> [DIAGNOSIS] Severe FRONT 外八 (duck-footed abduction): spread splayed to {np.rad2deg(front_spread.mean()):.1f}° (nominal 11.5°)")
    else:
        print(f"  --> Front spread is near nominal.")

    if rear_spread.mean() < 0.12:
        print(f"  --> [DIAGNOSIS] Severe REAR 内八 (pigeon-toed adduction): spread collapsed to {np.rad2deg(rear_spread.mean()):.1f}° (nominal 11.5°)")
    elif rear_spread.mean() > 0.35:
        print(f"  --> [DIAGNOSIS] Severe REAR 外八 (duck-footed abduction): spread splayed to {np.rad2deg(rear_spread.mean()):.1f}° (nominal 11.5°)")
    else:
        print(f"  --> Rear spread is near nominal.")

if __name__ == "__main__":
    print("Extracting MuJoCo telemetry...")
    analyze_scenario("scene", vx=1.2, duration=3.0)
    analyze_scenario("scene_stairs_5steps_15cm", vx=0.8, duration=5.0)

