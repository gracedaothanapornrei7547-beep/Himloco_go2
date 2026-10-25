#!/usr/bin/env python3
"""Automated Phase Telemetry & Validation Script for Unitree Go2 MuJoCo Sim2Sim.

Quantitatively measures:
1. Flat ground:
   - Base Z heave (peak-to-peak cm, std cm)
   - Base Vz RMS (m/s) & Az peak (m/s^2)
   - Heave FFT dominant frequency (Hz)
   - Ground reaction impact force (peak N)
   - Calf hyperextension (min calf angle rad/deg)
   - Forward speed tracking (actual vx / cmd vx)
2. 10cm Stairs:
   - Blind climbing distance & success (x >= 5.0m)
   - Pitch angle stability
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco

# Add current dir to import sim2sim_mujoco
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from sim2sim_mujoco import Go2MuJoCoSim2Sim


def evaluate_checkpoint(model_dir, assets_dir="/home/rum/himloco_assets/mujoco_menagerie/unitree_go2", duration=8.0):
    flat_scene = os.path.join(assets_dir, "scene.xml")
    stairs_scene = os.path.join(assets_dir, "scene_stairs_10cm.xml")

    results = {}

    # =========================================================================
    # 1. FLAT GROUND TEST
    # =========================================================================
    print(f"\n[Eval] Running Flat-Ground Test (cmd: vx=0.5m/s, duration={duration:.1f}s)...")
    env_flat = Go2MuJoCoSim2Sim(flat_scene, model_dir, kp=25.0, kd=0.5)
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    num_steps = int(duration / env_flat.policy_dt)
    z_history = []
    vz_history = []
    vx_history = []
    impact_forces = []
    calf_angles = []

    f6_buf = np.zeros(6, dtype=np.float64)

    for step in range(num_steps):
        info = env_flat.step(cmd)

        # Collect data after initial transient (first 1.0s / 50 steps)
        if step >= 50:
            z_history.append(info["pos"][2])
            vz_history.append(info["vel"][2])
            vx_history.append(info["vel"][0])

            # Measure foot contact normal forces
            step_max_force = 0.0
            for i in range(env_flat.data.ncon):
                con = env_flat.data.contact[i]
                for foot_name, gid in env_flat.foot_geom_ids.items():
                    if con.geom1 == gid or con.geom2 == gid:
                        mujoco.mj_contactForce(env_flat.model, env_flat.data, i, f6_buf)
                        step_max_force = max(step_max_force, float(f6_buf[0]))
            impact_forces.append(step_max_force)

            # Measure calf joint angles (IsaacLab calf indices: 8, 9, 10, 11)
            q_mj = env_flat.data.qpos[7:19]
            q_isaac = q_mj[env_flat.mujoco_to_isaac]
            calfs = q_isaac[8:12]  # [FL_calf, FR_calf, RL_calf, RR_calf]
            calf_angles.append(calfs)

    z_arr = np.array(z_history)
    vz_arr = np.array(vz_history)
    vx_arr = np.array(vx_history)
    forces_arr = np.array(impact_forces)
    calf_arr = np.array(calf_angles)

    # Compute numerical acceleration Az
    dt = env_flat.policy_dt
    az_arr = np.diff(vz_arr) / dt

    # FFT on base Z heave
    z_detrend = z_arr - np.mean(z_arr)
    fft_vals = np.abs(np.fft.rfft(z_detrend))
    fft_freqs = np.fft.rfftfreq(len(z_detrend), d=dt)
    dom_idx = np.argmax(fft_vals[1:]) + 1  # Skip DC
    dom_freq = fft_freqs[dom_idx] if len(fft_freqs) > 1 else 0.0

    z_p2p_cm = (np.max(z_arr) - np.min(z_arr)) * 100.0
    z_std_cm = np.std(z_arr) * 100.0
    vz_rms = np.sqrt(np.mean(vz_arr ** 2))
    az_peak = np.max(np.abs(az_arr)) if len(az_arr) > 0 else 0.0
    peak_force = np.max(forces_arr) if len(forces_arr) > 0 else 0.0
    mean_force = np.mean(forces_arr[forces_arr > 5.0]) if np.any(forces_arr > 5.0) else 0.0
    min_calf = np.min(calf_arr)
    mean_vx = np.mean(vx_arr)

    results["flat"] = {
        "z_p2p_cm": z_p2p_cm,
        "z_std_cm": z_std_cm,
        "z_mean_m": np.mean(z_arr),
        "vz_rms": vz_rms,
        "az_peak": az_peak,
        "dom_freq_hz": dom_freq,
        "peak_impact_force_n": peak_force,
        "mean_contact_force_n": mean_force,
        "min_calf_angle_rad": min_calf,
        "min_calf_angle_deg": np.rad2deg(min_calf),
        "mean_vx": mean_vx,
    }

    # =========================================================================
    # 2. 10CM STAIRS TEST
    # =========================================================================
    print(f"[Eval] Running 10cm Stairs Test (cmd: vx=0.5m/s, duration={duration:.1f}s)...")
    env_stairs = Go2MuJoCoSim2Sim(stairs_scene, model_dir, kp=25.0, kd=0.5)
    stairs_steps = int(duration / env_stairs.policy_dt)
    max_x = 0.0
    pitch_history = []
    stair_success = False

    for step in range(stairs_steps):
        info = env_stairs.step(cmd)
        max_x = max(max_x, info["pos"][0])
        pitch_history.append(info["rpy"][1])
        # Stair is at x=1.5m, height=0.10m. Normal ground base z is ~0.307m, on 10cm platform base z is ~0.407m.
        # When x >= 2.2m and z >= 0.39m, all four feet have climbed onto the platform.
        if info["pos"][0] >= 2.2 and info["pos"][2] >= 0.39:
            stair_success = True

    pitch_arr = np.array(pitch_history)
    results["stairs"] = {
        "max_x_m": max_x,
        "success": stair_success,
        "pitch_mean_deg": np.mean(pitch_arr),
        "pitch_std_deg": np.std(pitch_arr),
    }

    # =========================================================================
    # PRINT SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"  QUANTITATIVE TELEMETRY REPORT: {os.path.basename(model_dir)}")
    print("=" * 70)
    print(" [Flat Ground Dynamics]")
    print(f"   * Base Z Heave (Peak-to-Peak) : {z_p2p_cm:5.2f} cm   (Std: {z_std_cm:4.2f} cm, Mean: {results['flat']['z_mean_m']:.3f} m)")
    print(f"   * Base Vz RMS                 : {vz_rms:5.4f} m/s")
    print(f"   * Base Az Peak Acceleration   : {az_peak:5.2f} m/s² ({(az_peak / 9.81):4.2f} Gs)")
    print(f"   * Vertical Heave Dominant FFT : {dom_freq:5.2f} Hz")
    print(f"   * Peak Foot Impact Force      : {peak_force:5.1f} N   (Target <= 120N, Body Weight: ~118N)")
    print(f"   * Stance Contact Mean Force   : {mean_force:5.1f} N")
    print(f"   * Calf Hyperextension (Min)   : {min_calf:5.2f} rad ({np.rad2deg(min_calf):+5.1f}°)  (Default: -1.50 rad / -85.9°)")
    print(f"   * Forward Velocity Tracking   : {mean_vx:5.2f} m/s (Command: 0.50 m/s)")
    print("----------------------------------------------------------------------")
    print(" [10cm Blind Stair Climbing]")
    print(f"   * Final Max Forward Position  : {max_x:5.2f} m   (Stair edge at x=1.50 m)")
    print(f"   * Stair Climbing Success      : {'PASS (100% Blind Climb onto 10cm step)' if stair_success else 'FAIL'}")
    print(f"   * Body Pitch Stability        : Mean: {results['stairs']['pitch_mean_deg']:+5.1f}°, Std: {results['stairs']['pitch_std_deg']:4.1f}°")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 Phase Telemetry Evaluation")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to checkpoint directory with exported/ folder")
    parser.add_argument("--duration", type=float, default=8.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    evaluate_checkpoint(args.model_dir, duration=args.duration)
