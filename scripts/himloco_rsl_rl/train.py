# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with HimLoco RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with HimLoco RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--max_lin_vel_x",
    type=float,
    default=None,
    help="Raise forward-speed ceiling: set curriculums_limit_ranges + low_vel_env_lin_x_ranges to (-X, X).",
)
parser.add_argument(
    "--reset_terrain",
    action="store_true",
    default=False,
    help="Reset terrain curriculum to easy (max_init_terrain_level=0).",
)
parser.add_argument(
    "--track_ang_vel_z_weight",
    type=float,
    default=None,
    help="Override track_ang_vel_z reward weight (veer fix: raise so the policy can't sacrifice heading for forward speed).",
)
parser.add_argument(
    "--heading_control_stiffness",
    type=float,
    default=None,
    help="Override the heading P-gain (default 0.5). Veer fix: raise so the heading controller commands a larger corrective yaw rate.",
)
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import hashlib
import json
import os
import shutil
import inspect
import torch
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from himloco_lab.utils import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Train with HimLoco RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # ★ speed-ladder overrides (p4): raise forward-speed ceiling + reset terrain to easy.
    #   --max_lin_vel_x X  -> dense forward exposure up to X m/s, WITHOUT relying on the
    #     lin_vel_cmd_levels auto-ramp (which never fires under desync'd envs — its low/high
    #     split reads only the reset batch, so one group is always empty -> reward=0 -> no ramp):
    #       low_vel_env_lin_x_ranges = (0, X)    # 80% anchor group: forward-dense 0..X
    #       ranges.lin_vel_x         = (-1, X)   # 20% frontier group: keeps backward + turning
    #       curriculums_limit_ranges = (-X, X)   # ceiling (informational; ramp disabled below)
    #     lin_vel_cmd_levels is disabled so it can't re-widen the forward-dense ranges.
    #   --reset_terrain    -> max_init_terrain_level = 0 (all envs re-climb terrain from level 0).
    if args_cli.max_lin_vel_x is not None:
        vmax = args_cli.max_lin_vel_x
        env_cfg.commands.base_velocity.curriculums_limit_ranges = (-vmax, vmax)
        env_cfg.commands.base_velocity.low_vel_env_lin_x_ranges = (0.0, vmax)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, vmax)
        env_cfg.curriculum.lin_vel_cmd_levels = None
    if args_cli.reset_terrain:
        env_cfg.scene.terrain.max_init_terrain_level = 0
    if args_cli.track_ang_vel_z_weight is not None:
        # ★ veer fix: raise the yaw-rate tracking weight. The default 1.5 (vs
        #   track_lin_vel_xy=3.0, the open-source 1:2 ratio) is too weak at 2.0 m/s —
        #   the policy sacrifices heading to hold forward speed, veering ~8.7m left / 50m.
        env_cfg.rewards.track_ang_vel_z.weight = args_cli.track_ang_vel_z_weight
    if args_cli.heading_control_stiffness is not None:
        # ★ veer fix #2: raise the heading P-gain. The default 0.5 is too weak at 2.0 m/s —
        #   a 25° heading error only commands 0.22 rad/s, which the reward (std=0.5) trivially
        #   ignores. Higher gain shrinks the heading-error deadband so the drift is corrected
        #   before it accumulates into a multi-metre lateral offset.
        env_cfg.commands.base_velocity.heading_control_stiffness = args_cli.heading_control_stiffness

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    if agent_cfg.ab_protocol_id:
        # The terrain generator otherwise warns that its cached geometry is not
        # reproducible even when the environment seed is set.
        if env_cfg.scene.terrain.terrain_generator is None:
            raise ValueError("Controlled A/B training requires a seeded terrain generator.")
        env_cfg.scene.terrain.terrain_generator.seed = agent_cfg.seed
        # Component-local constructors use fork_rng, so HIM-only modules cannot
        # consume random numbers that would shift rollout sampling.
        agent_cfg.policy.initialization_seed = agent_cfg.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length
    )

    print(f"[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
    print(f"[INFO] history_length: {env.history_length}")
    print(f"[INFO] num_obs (total): {env.num_obs}")
    if env.num_one_step_privileged_obs is not None:
        print(f"[INFO] num_one_step_privileged_obs: {env.num_one_step_privileged_obs}")
        print(f"[INFO] privileged_history_length: {env.privileged_history_length}")
        print(f"[INFO] num_privileged_obs (total): {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # we use the same logic as Isaac Lab to find the checkpoint
        from isaaclab_tasks.utils import get_checkpoint_path
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create runner from HimLoco RSL-RL
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # load the checkpoint if resuming
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # Export deployment configuration with history_length and use_encoder flag
    use_encoder = hasattr(runner.alg.actor_critic, "estimator")
    export_deploy_cfg(
        env.unwrapped, 
        log_dir,
        history_length=agent_cfg.history_length,
        use_encoder=use_encoder,
    )
     
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )
    _write_ab_manifest(
        log_dir=log_dir,
        task=args_cli.task,
        agent_entry=args_cli.agent,
        env=env,
        agent_cfg=agent_cfg,
        use_encoder=use_encoder,
        actor_critic=runner.alg.actor_critic,
    )
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_env_sha256(path):
    """Hash the environment YAML while ignoring only its per-run log path."""
    with open(path, "r", encoding="utf-8") as stream:
        lines = [line for line in stream if not line.startswith("log_dir:")]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _write_ab_manifest(log_dir, task, agent_entry, env, agent_cfg, use_encoder, actor_critic):
    """Write the machine-checkable identity of one controlled A/B run."""
    params_dir = os.path.join(log_dir, "params")
    env_path = os.path.join(params_dir, "env.yaml")
    agent_path = os.path.join(params_dir, "agent.yaml")
    deploy_path = os.path.join(params_dir, "deploy.yaml")
    common_identity = {
        "protocol_id": agent_cfg.ab_protocol_id,
        "task": task,
        "seed": agent_cfg.seed,
        "num_envs": env.num_envs,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "save_interval": agent_cfg.save_interval,
        "history_frames": agent_cfg.history_length + 1,
        "privileged_history_frames": agent_cfg.privileged_history_length + 1,
        "policy_observation_dim": env.num_obs,
        "single_step_policy_dim": env.num_one_step_obs,
        "critic_observation_dim": env.num_privileged_obs,
        "action_dim": env.num_actions,
        "actor_hidden_dims": agent_cfg.policy.actor_hidden_dims,
        "critic_hidden_dims": agent_cfg.policy.critic_hidden_dims,
        "init_noise_std": agent_cfg.policy.init_noise_std,
        "initialization_seed": agent_cfg.policy.initialization_seed,
        "algorithm": agent_cfg.algorithm.to_dict(),
        "normalized_environment_sha256": _normalized_env_sha256(env_path),
    }
    canonical_common = json.dumps(common_identity, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema_version": 1,
        "common_identity": common_identity,
        "common_identity_sha256": hashlib.sha256(canonical_common.encode("utf-8")).hexdigest(),
        "variant": agent_cfg.ab_variant,
        "agent_entry": agent_entry,
        "uses_him_encoder": use_encoder,
        "policy_class_name": agent_cfg.policy_class_name,
        "algorithm_class_name": agent_cfg.algorithm_class_name,
        "eligible_for_model_selection": "smoke" not in agent_cfg.run_name.lower(),
        "initial_critic_state_sha256": _state_dict_sha256(actor_critic.critic.state_dict()),
        "initial_action_std_sha256": _state_dict_sha256({"std": actor_critic.std}),
        "initial_shared_actor_layers_sha256": _state_dict_sha256(
            {
                name: tensor
                for name, tensor in actor_critic.actor.state_dict().items()
                if not name.startswith("0.")
            }
        ),
        "parameter_file_sha256": {
            "env.yaml": _sha256_file(env_path),
            "agent.yaml": _sha256_file(agent_path),
            "deploy.yaml": _sha256_file(deploy_path),
        },
    }
    path = os.path.join(params_dir, "ab_manifest.json")
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
