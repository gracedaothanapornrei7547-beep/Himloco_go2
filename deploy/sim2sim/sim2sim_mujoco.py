#!/usr/bin/env python3
"""
Sim2Sim Deployment Pipeline for Unitree Go2: IsaacLab to MuJoCo (Python Native)

Features:
- Dual ONNX model execution (encoder.onnx + policy.onnx)
- Exact 45-dim observation pipeline with 6-step history FIFO (270 dims)
- Strict joint topology mapping between IsaacLab and MuJoCo Menagerie
- Low-level PD torque control with motor limits (Kp=25.0, Kd=0.5, clip=±23.7 Nm)
- Supports flat ground and 10cm/20cm stairs obstacle scenes
- Interactive passive viewer with real-time HUD telemetry
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

# Default paths
DEFAULT_MODEL_DIR = "/home/rum/himloco_lab/logs/himloco_rsl_rl/go2_rough/exported_best_13600"
DEFAULT_ASSETS_DIR = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2"


def quat_apply_inverse(quat, vec):
    """Rotate a 3D vector by the conjugate (inverse) of a quaternion [w, x, y, z].
    
    Equivalent to IsaacLab's math_utils.quat_apply_inverse.
    """
    w = quat[0]
    xyz = quat[1:]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


def quat_to_euler_rpy(quat):
    """Convert quaternion [w, x, y, z] to Roll-Pitch-Yaw (in degrees)."""
    w, x, y, z = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


class Go2MuJoCoSim2Sim:
    """Manages the full Sim2Sim inference and simulation loop for Go2 in MuJoCo."""

    def __init__(self, scene_path, model_dir, kp=25.0, kd=0.5, torque_limit=23.7):
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

        # 1. Load MuJoCo model
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo scene XML not found: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Zero out Menagerie's passive joint damping to align perfectly with IsaacLab PhysX
        self.model.dof_damping[6:] = 0.0

        # 2. Joint ordering & bidirectional mappings
        # IsaacLab order: [hip(4), thigh(4), calf(4)]
        self.isaac_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        self.default_joint_pos_isaac = np.array([
            0.1, -0.1, 0.1, -0.1,
            0.8, 0.8, 1.0, 1.0,
            -1.5, -1.5, -1.5, -1.5
        ], dtype=np.float32)

        # MuJoCo joints (1 to 12)
        self.mujoco_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(1, 13)
        ]

        # Build permutations
        self.mujoco_to_isaac = np.array([self.mujoco_joint_names.index(name) for name in self.isaac_joint_names])
        self.isaac_to_mujoco = np.array([self.isaac_joint_names.index(name) for name in self.mujoco_joint_names])
        self.default_joint_pos_mujoco = self.default_joint_pos_isaac[self.isaac_to_mujoco]

        # 3. Observation scales & limits
        self.obs_scales = {
            "cmd": np.array([2.0, 2.0, 0.25], dtype=np.float32),
            "ang_vel": 0.25,
            "gravity": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "last_action": 1.0,
        }

        # Dynamic action scale and observation terms from params/deploy.yaml
        deploy_yaml = os.path.join(model_dir, "params", "deploy.yaml")
        self.use_gait_phase = False
        self.gait_period = 0.40
        self.global_phase = 0.0

        if os.path.exists(deploy_yaml):
            try:
                import yaml
                with open(deploy_yaml, "r") as f:
                    dcfg = yaml.load(f, Loader=getattr(yaml, "Loader", yaml.SafeLoader))
                scale_val = dcfg["actions"]["JointPositionAction"]["scale"]
                if isinstance(scale_val, list):
                    self.action_scale = np.array(scale_val, dtype=np.float32)
                else:
                    self.action_scale = np.array([float(scale_val)] * 12, dtype=np.float32)
                print(f"[INFO] Loaded dynamic action scale from deploy.yaml: {self.action_scale}")

                obs_cfg = dcfg.get("observations", {})
                if "gait_phase" in obs_cfg:
                    self.use_gait_phase = True
                    self.gait_period = float(obs_cfg["gait_phase"].get("params", {}).get("period", 0.40))
                    print(f"[INFO] Using explicit gait_phase clock from deploy.yaml: period={self.gait_period:.2f}s (freq={1.0/self.gait_period:.2f}Hz)")
            except Exception as e:
                print(f"[WARNING] Could not parse deploy.yaml ({e}), defaulting action scale to 0.25")
                self.action_scale = np.array([0.25] * 12, dtype=np.float32)
        else:
            self.action_scale = np.array([0.25] * 12, dtype=np.float32)

        # 4. History buffer (6 steps * num_obs)
        self.num_obs = 47 if self.use_gait_phase else 45
        self.history_len = 6
        self.history_buffer = np.zeros(self.num_obs * self.history_len, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_action_raw = np.zeros(12, dtype=np.float32)
        self.current_target_pos_mujoco = self.default_joint_pos_mujoco.copy()

        # 5. Load ONNX models
        if os.path.exists(os.path.join(model_dir, "encoder.onnx")) and os.path.exists(os.path.join(model_dir, "policy.onnx")):
            enc_path = os.path.join(model_dir, "encoder.onnx")
            pol_path = os.path.join(model_dir, "policy.onnx")
        else:
            enc_path = os.path.join(model_dir, "exported", "encoder.onnx")
            pol_path = os.path.join(model_dir, "exported", "policy.onnx")
        if not os.path.exists(enc_path) or not os.path.exists(pol_path):
            raise FileNotFoundError(f"ONNX models missing in: {model_dir} or {model_dir}/exported/")
        
        # CPU provider with 2 threads is plenty for real-time (sub-millisecond) inference
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.enc_sess = ort.InferenceSession(enc_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.pol_sess = ort.InferenceSession(pol_path, sess_options=so, providers=["CPUExecutionProvider"])

        # Foot geom IDs for contact detection
        self.foot_geom_ids = {}
        for foot_name in ["FL", "FR", "RL", "RR"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if gid != -1:
                self.foot_geom_ids[foot_name] = gid

        # Timing
        self.sim_dt = self.model.opt.timestep  # 0.002s (500Hz)
        self.policy_dt = 0.02                  # 0.020s (50Hz)
        self.decimation = int(np.round(self.policy_dt / self.sim_dt))

        self.reset()

    def reset(self, init_pos=(0.0, 0.0, 0.35)):
        """Reset robot pose and history buffers."""
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = init_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity orientation
        self.data.qpos[7:19] = self.default_joint_pos_mujoco
        self.data.qvel[:] = 0.0

        self.history_buffer.fill(0.0)
        self.last_action.fill(0.0)
        self.last_action_raw = np.zeros(12, dtype=np.float32)
        self.current_target_pos_mujoco = self.default_joint_pos_mujoco.copy()
        self.global_phase = 0.0

        # Settle robot gently onto the terrain
        for _ in range(150):
            q_curr = self.data.qpos[7:19]
            dq_curr = self.data.qvel[6:18]
            tau = self.kp * (self.default_joint_pos_mujoco - q_curr) - self.kd * dq_curr
            self.data.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.model, self.data)

    def get_observation(self, cmd):
        """Construct the 45-dim (or 47-dim with gait_phase) current observation exactly matching IsaacLab."""
        q_mj = self.data.qpos[7:19]
        dq_mj = self.data.qvel[6:18]
        q_isaac = q_mj[self.mujoco_to_isaac]
        dq_isaac = dq_mj[self.mujoco_to_isaac]

        quat = self.data.qpos[3:7]  # [w, x, y, z]
        ang_vel_w = self.data.qvel[3:6]
        ang_vel_b = quat_apply_inverse(quat, ang_vel_w)
        proj_gravity_b = quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        # Concatenate terms
        obs_terms = [
            cmd * self.obs_scales["cmd"],
            ang_vel_b * self.obs_scales["ang_vel"],
            proj_gravity_b * self.obs_scales["gravity"],
            (q_isaac - self.default_joint_pos_isaac) * self.obs_scales["joint_pos"],
            dq_isaac * self.obs_scales["joint_vel"],
            self.last_action * self.obs_scales["last_action"],
        ]
        if self.use_gait_phase:
            phase_rad = self.global_phase * 2.0 * np.pi
            obs_terms.append(np.array([np.sin(phase_rad), np.cos(phase_rad)], dtype=np.float32))
        current_obs = np.concatenate(obs_terms).astype(np.float32)
        return np.clip(current_obs, -100.0, 100.0)

    def step(self, cmd):
        """Execute one policy step (50Hz) and corresponding physics substeps (500Hz)."""
        current_obs = self.get_observation(cmd)
        if self.use_gait_phase:
            self.global_phase = (self.global_phase + self.policy_dt / self.gait_period) % 1.0

        # Update history buffer (newest at front)
        self.history_buffer[self.num_obs:] = self.history_buffer[:-self.num_obs]
        self.history_buffer[:self.num_obs] = current_obs

        # Dual network forward pass
        latent = self.enc_sess.run(None, {"obs_history": self.history_buffer.reshape(1, -1)})[0].flatten()
        policy_in = np.concatenate([current_obs, latent]).reshape(1, -1).astype(np.float32)
        action = self.pol_sess.run(None, {"obs": policy_in})[0].flatten()
        action = np.clip(action, -2.5, 2.5)
        self.last_action = action.copy()

        # Action to joint targets (IsaacLab convention: target = nominal + action_scale * action)
        target_pos_isaac = self.default_joint_pos_isaac + self.action_scale * action
        target_pos_mujoco = target_pos_isaac[self.isaac_to_mujoco]

        # Substep physics (direct zero-order hold exactly matching IsaacLab physics)
        peak_torque = 0.0
        for _ in range(self.decimation):
            q_curr = self.data.qpos[7:19]
            dq_curr = self.data.qvel[6:18]
            tau = self.kp * (target_pos_mujoco - q_curr) - self.kd * dq_curr
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
            peak_torque = max(peak_torque, float(np.max(np.abs(tau))))
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        # Telemetry
        contacts = self.get_foot_contacts()
        return {
            "pos": self.data.qpos[0:3].copy(),
            "vel": self.data.qvel[0:3].copy(),
            "rpy": quat_to_euler_rpy(self.data.qpos[3:7]),
            "contacts": contacts,
            "peak_torque": peak_torque,
        }

    def get_foot_contacts(self):
        """Check contact state for each foot geom."""
        contacts = {name: False for name in self.foot_geom_ids}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for foot_name, gid in self.foot_geom_ids.items():
                if con.geom1 == gid or con.geom2 == gid:
                    contacts[foot_name] = True
        return contacts


def run_sim2sim(args):
    """Main Sim2Sim execution function."""
    scene_file = "scene.xml" if args.scene == "flat" else "scene_stairs_10cm.xml"
    if args.scene in ["stairs", "stairs_5steps"]:
        scene_file = "scene_stairs_5steps.xml"
    elif args.scene == "stairs_10cm":
        scene_file = "scene_stairs_10cm.xml"
    elif args.scene == "stairs_5steps_5cm":
        scene_file = "scene_stairs_5steps_5cm.xml"
    elif args.scene == "stairs_5steps_15cm":
        scene_file = "scene_stairs_5steps_15cm.xml"
    else:
        scene_file = "scene.xml"
    scene_path = os.path.join(args.assets_dir, scene_file)

    print("===================================================================")
    print("  Unitree Go2 MuJoCo Sim2Sim Deployment Runner")
    print(f"  - Checkpoint Model: {args.model_dir}")
    print(f"  - Scene: {args.scene} ({scene_path})")
    print(f"  - Velocity Command: vx={args.vx:.2f} m/s, vy={args.vy:.2f} m/s, wz={args.wz:.2f} rad/s")
    print(f"  - Duration: {args.duration:.1f} s (Decimation: {args.decimation})")
    print(f"  - Viewer Mode: {'Headless' if args.headless else 'Interactive Passive Viewer'}")
    print("===================================================================\n")

    env = Go2MuJoCoSim2Sim(
        scene_path=scene_path,
        model_dir=args.model_dir,
        kp=args.kp,
        kd=args.kd,
    )

    cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    target_yaw_rad = 0.0
    num_steps = int(args.duration / env.policy_dt) if args.duration > 0 else int(1e9)

    if args.record_video:
        import imageio
        out_dir = os.path.dirname(os.path.abspath(args.record_video))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = args.cam_distance
        camera.elevation = args.cam_elevation
        camera.azimuth = args.cam_azimuth
        camera.lookat[:] = [0.8, 0.0, 0.35]

        writer = imageio.get_writer(args.record_video, fps=args.fps, codec="libx264", quality=8)
        print(f"[Recorder] Recording video to {args.record_video} ({args.duration}s at {args.fps}fps, az={args.cam_azimuth}, el={args.cam_elevation})...")

        for step_idx in range(num_steps):
            if args.heading_mode:
                current_rpy = quat_to_euler_rpy(env.data.qpos[3:7])
                yaw_rad = np.deg2rad(current_rpy[2])
                yaw_err = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi
                cmd[2] = np.clip(args.heading_kp * yaw_err, -1.0, 1.0)

            info = env.step(cmd)
            lookat_z = info["pos"][2] + 0.15 if "stair" in args.scene else 0.38
            cam_target = np.array([info["pos"][0] + 0.4, info["pos"][1], lookat_z])
            camera.lookat[:] = 0.90 * camera.lookat + 0.10 * cam_target

            # Auto-reset in stairs scene after climbing platform or after 30m on flat ground
            if "stair" in args.scene and info["pos"][0] > 6.0:
                print("\n[Auto-Reset] Successfully climbed stair platform! Respawning at start position...\n")
                env.reset()
                camera.lookat[:] = [0.8, 0.0, 0.35]
            elif args.scene == "flat" and info["pos"][0] > 30.0:
                print("\n[Auto-Reset] Ran 30m on flat ground! Respawning at start position...\n")
                env.reset()
                camera.lookat[:] = [0.8, 0.0, 0.35]

            renderer.update_scene(env.data, camera=camera)
            pixels = renderer.render()
            writer.append_data(pixels)

            sim_time = (step_idx + 1) * env.policy_dt
            if step_idx % 25 == 0 or step_idx == num_steps - 1:
                c_str = " ".join([f"{k}:{'●' if v else '○'}" for k, v in info["contacts"].items()])
                print(
                    f"[{sim_time:5.2f}s] Pos: ({info['pos'][0]:5.2f}, {info['pos'][1]:5.2f}, {info['pos'][2]:5.2f})m | "
                    f"Vx: {info['vel'][0]:5.2f}m/s | Pitch: {info['rpy'][1]:+5.1f}° | Feet: [{c_str}] | Tau: {info['peak_torque']:4.1f}Nm"
                )

        writer.close()
        renderer.close()
        print(f"\n[Done] Video saved successfully to {args.record_video}")
    elif args.headless:
        start_time = time.time()
        for step_idx in range(num_steps):
            if args.heading_mode:
                current_rpy = quat_to_euler_rpy(env.data.qpos[3:7])
                yaw_rad = np.deg2rad(current_rpy[2])
                yaw_err = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi
                cmd[2] = np.clip(args.heading_kp * yaw_err, -1.0, 1.0)

            info = env.step(cmd)
            sim_time = (step_idx + 1) * env.policy_dt

            if step_idx % 25 == 0 or step_idx == num_steps - 1:  # Print every 0.5s
                c_str = " ".join([f"{k}:{'●' if v else '○'}" for k, v in info["contacts"].items()])
                print(
                    f"[{sim_time:5.2f}s] Pos: x={info['pos'][0]:5.2f}, y={info['pos'][1]:5.2f}, z={info['pos'][2]:5.2f}m | "
                    f"Vel: vx={info['vel'][0]:5.2f}, vy={info['vel'][1]:5.2f}m/s | "
                    f"Pitch: {info['rpy'][1]:+5.1f}° | Feet: [{c_str}] | Tau: {info['peak_torque']:4.1f}Nm"
                )
        total_wall = time.time() - start_time
        print(f"\n[Done] Executed {num_steps} policy steps in {total_wall:.2f}s (Speedup: {args.duration / total_wall:.1f}x real-time)")
    else:
        print("[Viewer] Launching MuJoCo passive viewer window on DISPLAY=:0...")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            # Set camera viewpoint to rear follow chase view
            viewer.cam.distance = args.cam_distance
            viewer.cam.elevation = args.cam_elevation
            viewer.cam.azimuth = args.cam_azimuth
            viewer.cam.lookat[:] = [0.8, 0.0, 0.35]

            step_idx = 0
            wall_start = time.time()

            while viewer.is_running() and step_idx < num_steps:
                step_start = time.time()
                if args.heading_mode:
                    current_rpy = quat_to_euler_rpy(env.data.qpos[3:7])
                    yaw_rad = np.deg2rad(current_rpy[2])
                    yaw_err = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi
                    cmd[2] = np.clip(args.heading_kp * yaw_err, -1.0, 1.0)

                info = env.step(cmd)
                # Camera track robot with dynamic Z on stairs, decoupled Z on flat ground
                lookat_z = info["pos"][2] + 0.15 if "stair" in args.scene else 0.38
                cam_target = np.array([info["pos"][0] + 0.45, info["pos"][1], lookat_z])
                viewer.cam.lookat[:] = 0.90 * viewer.cam.lookat + 0.10 * cam_target
                viewer.sync()

                sim_time = (step_idx + 1) * env.policy_dt
                if step_idx % 25 == 0:
                    c_str = " ".join([f"{k}:{'●' if v else '○'}" for k, v in info["contacts"].items()])
                    print(
                        f"[{sim_time:5.2f}s] Pos: ({info['pos'][0]:5.2f}, {info['pos'][1]:5.2f}, {info['pos'][2]:5.2f})m | "
                        f"Vx: {info['vel'][0]:5.2f}m/s | Pitch: {info['rpy'][1]:+5.1f}° | Feet: [{c_str}] | Tau: {info['peak_torque']:4.1f}Nm"
                    )

                step_idx += 1

                # Auto-reset in stairs scene after climbing platform or after 30m on flat ground
                if "stair" in args.scene and info["pos"][0] > 6.0:
                    print("\n[Auto-Reset] Successfully climbed stair platform! Respawning at start position...\n")
                    env.reset()
                elif args.scene == "flat" and info["pos"][0] > 30.0:
                    print("\n[Auto-Reset] Ran 30m on flat ground! Respawning at start position...\n")
                    env.reset()

                # Sync with real-time if running faster than real-time
                elapsed = time.time() - step_start
                if elapsed < env.policy_dt:
                    time.sleep(env.policy_dt - elapsed)

        print("[Viewer] Window closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 MuJoCo Sim2Sim Runner")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to checkpoint directory")
    parser.add_argument("--assets_dir", type=str, default=DEFAULT_ASSETS_DIR, help="Path to Menagerie Go2 directory")
    parser.add_argument("--scene", type=str, choices=["flat", "stairs", "stairs_10cm", "stairs_5steps", "stairs_5steps_5cm", "stairs_5steps_15cm"], default="flat", help="Scene type")
    parser.add_argument("--vx", type=float, default=0.9, help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw angular velocity command (rad/s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (seconds)")
    parser.add_argument("--kp", type=float, default=25.0, help="Joint stiffness Kp")
    parser.add_argument("--kd", type=float, default=0.5, help="Joint damping Kd")
    parser.add_argument("--decimation", type=int, default=10, help="Physics substeps per policy step")
    parser.add_argument("--heading_mode", action="store_true", default=False, help="Enable automatic heading lock controller")
    parser.add_argument("--heading_kp", type=float, default=0.8, help="Heading lock P-gain (default 0.8)")
    parser.add_argument("--record_video", type=str, default=None, help="Path to save recorded MP4 video")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default 50)")
    parser.add_argument("--cam_azimuth", type=float, default=25.0, help="Camera azimuth angle (default 25.0 for rear-left chase view)")
    parser.add_argument("--cam_elevation", type=float, default=-18.0, help="Camera elevation angle (default -18.0)")
    parser.add_argument("--cam_distance", type=float, default=2.4, help="Camera distance (default 2.4)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical viewer")

    args = parser.parse_args()
    run_sim2sim(args)

