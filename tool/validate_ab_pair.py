#!/usr/bin/env python3
"""Fail closed unless two HimLoco A/B runs share the locked common identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_VARIANTS = {"history_mlp_270", "him_release_encoder_270_to_19"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path):
    path = path.resolve()
    manifest_path = path / "params" / "ab_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    for filename, expected_hash in manifest["parameter_file_sha256"].items():
        actual_hash = sha256_file(path / "params" / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"{path}: {filename} hash mismatch")
    checkpoints = sorted(path.glob("model_*.pt"))
    if not checkpoints:
        raise ValueError(f"{path}: no checkpoint found")
    checkpoint = checkpoints[-1]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = state.get("ab_metadata")
    if metadata is None:
        raise ValueError(f"{checkpoint}: missing ab_metadata")
    return path, manifest, checkpoint, metadata, state


def validate_pair(left, right, expected_eligibility=None):
    runs = [load_run(left), load_run(right)]
    variants = {run[1]["variant"] for run in runs}
    if variants != EXPECTED_VARIANTS:
        raise ValueError(f"Expected one MLP and one HIM variant, got {sorted(variants)}")
    if runs[0][1]["common_identity_sha256"] != runs[1][1]["common_identity_sha256"]:
        raise ValueError("A/B common identities differ")
    if runs[0][1]["common_identity"] != runs[1][1]["common_identity"]:
        raise ValueError("A/B common identity payloads differ")

    for path, manifest, checkpoint, metadata, state in runs:
        common = manifest["common_identity"]
        required = {
            "protocol_id": common["protocol_id"],
            "variant": manifest["variant"],
            "policy_class_name": manifest["policy_class_name"],
            "algorithm_class_name": manifest["algorithm_class_name"],
            "num_actor_obs": common["policy_observation_dim"],
            "num_critic_obs": common["critic_observation_dim"],
            "num_one_step_obs": common["single_step_policy_dim"],
            "num_actions": common["action_dim"],
            "history_length": common["history_frames"] - 1,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"{checkpoint}: metadata {key} mismatch")
        tensors = state["model_state_dict"]
        first_actor = tensors["actor.0.weight"]
        first_critic = tensors["critic.0.weight"]
        expected_actor_in = 270 if manifest["variant"] == "history_mlp_270" else 64
        if tuple(first_actor.shape) != (512, expected_actor_in):
            raise ValueError(f"{checkpoint}: unexpected actor shape {tuple(first_actor.shape)}")
        if tuple(first_critic.shape) != (512, 238):
            raise ValueError(f"{checkpoint}: unexpected critic shape {tuple(first_critic.shape)}")
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f"{checkpoint}: non-finite tensor")
        if expected_eligibility is not None and manifest["eligible_for_model_selection"] != expected_eligibility:
            raise ValueError(
                f"{path}: expected eligibility={expected_eligibility}, got "
                f"{manifest['eligible_for_model_selection']}"
            )

    if runs[0][1]["initial_critic_state_sha256"] != runs[1][1]["initial_critic_state_sha256"]:
        raise ValueError("Initial critic state differs between variants")
    if runs[0][1]["initial_action_std_sha256"] != runs[1][1]["initial_action_std_sha256"]:
        raise ValueError("Initial action standard deviation differs between variants")
    if runs[0][1]["initial_shared_actor_layers_sha256"] != runs[1][1]["initial_shared_actor_layers_sha256"]:
        raise ValueError("Initial shared actor layers differ between variants")

    return {
        "status": "passed",
        "protocol_id": runs[0][1]["common_identity"]["protocol_id"],
        "common_identity_sha256": runs[0][1]["common_identity_sha256"],
        "runs": {
            run[1]["variant"]: {
                "path": str(run[0]),
                "checkpoint": str(run[2]),
                "checkpoint_sha256": sha256_file(run[2]),
            }
            for run in runs
        },
        "critic_initialization_exactly_equal": True,
        "shared_actor_layer_initialization_exactly_equal": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mlp_run", type=Path)
    parser.add_argument("him_run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expect-eligibility",
        choices=("eligible", "ineligible", "any"),
        default="any",
    )
    args = parser.parse_args()
    expected = {"eligible": True, "ineligible": False, "any": None}[args.expect_eligibility]
    report = validate_pair(args.mlp_run, args.him_run, expected_eligibility=expected)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
