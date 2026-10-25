# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Plain history-stacking MLP used as the controlled HimLoco A/B baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from contextlib import contextmanager, nullcontext


class HistoryMLPActorCritic(nn.Module):
    """Feed the complete policy history directly to a conventional MLP.

    Both variants use the same runner, rollout storage, PPO update, critic,
    action distribution, environment, and history buffer. The intended policy
    difference is 270->MLP here versus 270->HIM->(45+3+16)->MLP.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_actions,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        normalize_obs=False,
        initialization_seed=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "HistoryMLPActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(sorted(kwargs))
            )
        super().__init__()
        if num_actor_obs % num_one_step_obs != 0:
            raise ValueError(
                f"History dimension {num_actor_obs} is not divisible by single-step dimension {num_one_step_obs}."
            )
        self.history_size = num_actor_obs // num_one_step_obs
        self.normalize_obs = normalize_obs
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs
        self.actor = _build_mlp(
            num_actor_obs,
            actor_hidden_dims,
            num_actions,
            activation,
            initialization_seed=initialization_seed,
            seed_offset=100,
        )
        self.critic = _build_mlp(
            num_critic_obs,
            critic_hidden_dims,
            1,
            activation,
            initialization_seed=initialization_seed,
            seed_offset=200,
        )
        print(f"History MLP actor: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        del dones

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        if obs_history.shape[-1] != self.num_actor_obs:
            raise ValueError(f"Expected {self.num_actor_obs} observations, got {obs_history.shape[-1]}.")
        mean = self.actor(obs_history)
        if not torch.isfinite(mean).all():
            raise ValueError("NaN or Inf detected in HistoryMLPActorCritic output.")
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history=None, **kwargs):
        del kwargs
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        del observations
        return self.actor(obs_history)

    def test_inference(self, obs_history, observations=None):
        del observations
        return self.actor(obs_history), obs_history.new_zeros((obs_history.shape[0], 0))

    def evaluate(self, critic_observations, **kwargs):
        del kwargs
        return self.critic(critic_observations)


def _build_mlp(input_dim, hidden_dims, output_dim, activation, initialization_seed=None, seed_offset=0):
    if not hidden_dims:
        with _seed_scope(initialization_seed, seed_offset):
            return nn.Sequential(nn.Linear(input_dim, output_dim))
    with _seed_scope(initialization_seed, seed_offset):
        first = nn.Linear(input_dim, hidden_dims[0])
    layers = [first, _get_activation(activation)]
    for index, hidden_dim in enumerate(hidden_dims):
        if index == len(hidden_dims) - 1:
            with _seed_scope(initialization_seed, seed_offset + index + 1):
                layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            with _seed_scope(initialization_seed, seed_offset + index + 1):
                linear = nn.Linear(hidden_dim, hidden_dims[index + 1])
            layers.extend([linear, _get_activation(activation)])
    return nn.Sequential(*layers)


def _get_activation(name):
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


@contextmanager
def _forked_seed(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _seed_scope(seed, offset):
    if seed is None:
        return nullcontext()
    return _forked_seed(int(seed) + offset)
