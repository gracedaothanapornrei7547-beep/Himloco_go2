#!/usr/bin/env python3
"""Run the locked Go2 history-MLP versus HIM experiment sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path("/home/rum/himloco_lab")
LAUNCHER = REPO_ROOT / "run_himloco.sh"
TRAIN_SCRIPT = "scripts/himloco_rsl_rl/train.py"
VALIDATOR = REPO_ROOT / "scripts/himloco_rsl_rl/validate_ab_pair.py"
PYTHON = Path("/home/rum/env_isaaclab_himloco/bin/python")
PROTOCOL_ID = "go2_him_vs_history_mlp_v1"
SOURCE_FILES = (
    "run_himloco.sh",
    "AB_TRAINING.md",
    "source/himloco_lab/pyproject.toml",
    "source/himloco_lab/setup.py",
    "scripts/himloco_rsl_rl/train.py",
    "scripts/himloco_rsl_rl/validate_ab_pair.py",
    "scripts/himloco_rsl_rl/run_ab_training.py",
    "source/himloco_lab/himloco_lab/rsl_rl/algorithms/him_ppo.py",
    "source/himloco_lab/himloco_lab/rsl_rl/config/rl_cfg.py",
    "source/himloco_lab/himloco_lab/rsl_rl/history.py",
    "source/himloco_lab/himloco_lab/rsl_rl/modules/him_actor_critic.py",
    "source/himloco_lab/himloco_lab/rsl_rl/modules/him_estimator.py",
    "source/himloco_lab/himloco_lab/rsl_rl/modules/history_mlp_actor_critic.py",
    "source/himloco_lab/himloco_lab/rsl_rl/runners/him_on_policy_runner.py",
    "source/himloco_lab/himloco_lab/rsl_rl/storage/him_rollout_storage.py",
    "source/himloco_lab/himloco_lab/rsl_rl/wrappers/himloco_vec_env_wrapper.py",
    "source/himloco_lab/himloco_lab/tasks/locomotion/agents/ab_rsl_rl_cfg.py",
    "source/himloco_lab/himloco_lab/tasks/locomotion/robots/go2/__init__.py",
    "source/himloco_lab/himloco_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py",
)
VARIANTS = {
    "mlp": {
        "agent": "ab_mlp_rsl_rl_cfg",
        "experiment": "go2_ab_history_mlp",
        "manifest_variant": "history_mlp_270",
    },
    "him": {
        "agent": "ab_him_rsl_rl_cfg",
        "experiment": "go2_ab_him",
        "manifest_variant": "him_release_encoder_270_to_19",
    },
}


class Interrupted(Exception):
    pass


ACTIVE_CHILD = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_hashes():
    result = {}
    relative_files = set(SOURCE_FILES)
    source_root = REPO_ROOT / "source" / "himloco_lab" / "himloco_lab"
    relative_files.update(str(path.relative_to(REPO_ROOT)) for path in source_root.rglob("*.py"))
    for relative in sorted(relative_files):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Locked source is missing: {path}")
        result[relative] = sha256_file(path)
    return result


def assert_sources(expected):
    actual = source_hashes()
    if actual != expected:
        changed = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise RuntimeError(f"Locked A/B source changed: {changed}")


def signal_handler(signum, _frame):
    global ACTIVE_CHILD
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        os.killpg(ACTIVE_CHILD.pid, signal.SIGTERM)
    raise Interrupted(f"received signal {signum}")


def build_plan(seeds):
    entries = []
    for index, seed in enumerate(seeds):
        order = ("mlp", "him") if index % 2 == 0 else ("him", "mlp")
        for variant in order:
            entries.append(
                {
                    "key": f"seed{seed}_{variant}",
                    "seed": seed,
                    "variant": variant,
                    "status": "pending",
                    "launches": 0,
                    "run_dir": None,
                    "checkpoint": None,
                    "checkpoint_sha256": None,
                }
            )
    return entries


def discover_new_run(experiment, run_name, before):
    root = REPO_ROOT / "logs" / "himloco_rsl_rl" / experiment
    after = {path.resolve() for path in root.glob(f"*_{run_name}") if path.is_dir()}
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one new run for {run_name}, found {created}")
    return created[0]


def latest_checkpoint(run_dir, expected_iteration):
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        try:
            iteration = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        checkpoints.append((iteration, path.resolve()))
    if not checkpoints:
        raise RuntimeError(f"No checkpoint in {run_dir}")
    iteration, checkpoint = max(checkpoints)
    if iteration != expected_iteration - 1:
        raise RuntimeError(f"Expected final iteration {expected_iteration - 1}, got {iteration}")
    return checkpoint


def validate_checkpoint(checkpoint, manifest, entry, expected_iteration):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("iter") != expected_iteration - 1:
        raise RuntimeError(f"{checkpoint}: internal iteration mismatch")
    metadata = state.get("ab_metadata")
    if metadata is None:
        raise RuntimeError(f"{checkpoint}: missing ab_metadata")
    common = manifest["common_identity"]
    expected_metadata = {
        "protocol_id": PROTOCOL_ID,
        "variant": manifest["variant"],
        "policy_class_name": manifest["policy_class_name"],
        "algorithm_class_name": manifest["algorithm_class_name"],
        "num_actor_obs": 270,
        "num_critic_obs": 238,
        "num_one_step_obs": 45,
        "num_actions": 12,
        "history_length": 5,
    }
    wrong = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if wrong:
        raise RuntimeError(f"{checkpoint}: checkpoint metadata mismatch: {wrong}")
    tensors = state.get("model_state_dict")
    if not isinstance(tensors, dict) or not tensors:
        raise RuntimeError(f"{checkpoint}: missing model state")
    if not all(torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise RuntimeError(f"{checkpoint}: non-finite model tensor")
    actor_input = 270 if entry["variant"] == "mlp" else 64
    if tuple(tensors["actor.0.weight"].shape) != (512, actor_input):
        raise RuntimeError(f"{checkpoint}: actor schema mismatch")
    if tuple(tensors["critic.0.weight"].shape) != (512, 238):
        raise RuntimeError(f"{checkpoint}: critic schema mismatch")


def validate_run_identity(run_dir, entry, args):
    manifest_path = run_dir / "params" / "ab_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    common = manifest["common_identity"]
    expected = VARIANTS[entry["variant"]]
    checks = {
        "protocol": (common["protocol_id"], PROTOCOL_ID),
        "variant": (manifest["variant"], expected["manifest_variant"]),
        "seed": (common["seed"], entry["seed"]),
        "num_envs": (common["num_envs"], args.num_envs),
        "iterations": (common["max_iterations"], args.iterations),
        "steps": (common["num_steps_per_env"], 100),
        "policy_dim": (common["policy_observation_dim"], 270),
        "critic_dim": (common["critic_observation_dim"], 238),
        "action_dim": (common["action_dim"], 12),
        "eligible": (manifest["eligible_for_model_selection"], not args.smoke),
    }
    wrong = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
    if wrong:
        raise RuntimeError(f"Run identity mismatch for {run_dir}: {wrong}")
    return manifest


def run_entry(entry, args, state, state_path, hashes):
    global ACTIVE_CHILD
    spec = VARIANTS[entry["variant"]]
    phase = "smoke" if args.smoke else "formal"
    run_name = f"ab_{phase}_v1_seed{entry['seed']}_{entry['variant']}_{args.num_envs}env_{args.iterations}iter"
    experiment_root = REPO_ROOT / "logs" / "himloco_rsl_rl" / spec["experiment"]
    before = {path.resolve() for path in experiment_root.glob(f"*_{run_name}") if path.is_dir()}
    entry["status"] = "running"
    entry["launches"] += 1
    entry["started_at"] = utc_now()
    log_path = state_path.parent / f"{entry['key']}.log"
    entry["log"] = str(log_path)
    atomic_json(state_path, state)
    command = [
        str(LAUNCHER),
        TRAIN_SCRIPT,
        "--task", "Unitree-Go2-Velocity",
        "--agent", spec["agent"],
        "--seed", str(entry["seed"]),
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.iterations),
        "--run_name", run_name,
        "--device", "cuda:0",
        "--headless",
    ]
    entry["command"] = command
    atomic_json(state_path, state)
    assert_sources(hashes)
    with log_path.open("ab", buffering=0) as log_stream:
        ACTIVE_CHILD = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return_code = ACTIVE_CHILD.wait()
        ACTIVE_CHILD = None
    assert_sources(hashes)
    if return_code != 0:
        raise RuntimeError(f"{entry['key']} exited with code {return_code}; see {log_path}")
    run_dir = discover_new_run(spec["experiment"], run_name, before)
    manifest = validate_run_identity(run_dir, entry, args)
    checkpoint = latest_checkpoint(run_dir, args.iterations)
    validate_checkpoint(checkpoint, manifest, entry, args.iterations)
    entry.update(
        status="completed",
        completed_at=utc_now(),
        run_dir=str(run_dir),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        common_identity_sha256=manifest["common_identity_sha256"],
    )
    atomic_json(state_path, state)


def validate_completed_pairs(state, output_dir, smoke):
    by_seed = {}
    for entry in state["entries"]:
        if entry["status"] == "completed":
            by_seed.setdefault(entry["seed"], {})[entry["variant"]] = entry
    for seed, pair in sorted(by_seed.items()):
        if set(pair) != {"mlp", "him"}:
            continue
        report = output_dir / f"pair_seed{seed}_validation.json"
        if report.exists():
            continue
        subprocess.run(
            [
                str(PYTHON), str(VALIDATOR), pair["mlp"]["run_dir"], pair["him"]["run_dir"],
                "--expect-eligibility", "ineligible" if smoke else "eligible", "--output", str(report),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--smoke", action="store_true", help="Mark all generated checkpoints ineligible.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "himloco_rsl_rl" / "ab_formal_v1",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if args.num_envs < 1 or args.iterations < 1:
        raise ValueError("num-envs and iterations must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "state.json"
    hashes = source_hashes()
    settings = {
        "protocol_id": PROTOCOL_ID,
        "seeds": args.seeds,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "num_steps_per_env": 100,
        "device": "cuda:0",
        "headless": True,
        "smoke": args.smoke,
        "source_sha256": hashes,
    }
    settings_text = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    settings_hash = hashlib.sha256(settings_text.encode("utf-8")).hexdigest()
    manifest_path = args.output_dir / "experiment_manifest.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as stream:
            state = json.load(stream)
        if state["settings_sha256"] != settings_hash:
            raise RuntimeError("Existing A/B state uses different locked settings")
        assert_sources(state["settings"]["source_sha256"])
        hashes = state["settings"]["source_sha256"]
    else:
        state = {
            "schema_version": 1,
            "status": "running",
            "created_at": utc_now(),
            "settings": settings,
            "settings_sha256": settings_hash,
            "entries": build_plan(args.seeds),
            "error": None,
        }
        atomic_json(manifest_path, settings)
        atomic_json(state_path, state)

    try:
        for entry in state["entries"]:
            if entry["status"] == "completed":
                continue
            if entry["launches"] >= 2:
                raise RuntimeError(f"Persistent launch budget exhausted for {entry['key']}")
            run_entry(entry, args, state, state_path, hashes)
            validate_completed_pairs(state, args.output_dir, args.smoke)
        state["status"] = "completed"
        state["completed_at"] = utc_now()
        atomic_json(state_path, state)
    except Interrupted as error:
        state["status"] = "interrupted"
        state["error"] = str(error)
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        return 130
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["updated_at"] = utc_now()
        for entry in state["entries"]:
            if entry["status"] == "running":
                entry["status"] = "failed"
                entry["error"] = repr(error)
        atomic_json(state_path, state)
        raise
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    sys.exit(main())
