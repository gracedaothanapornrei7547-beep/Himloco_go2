import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "himloco_lab"
    / "rsl_rl"
    / "modules"
    / "history_mlp_actor_critic.py"
)
SPEC = importlib.util.spec_from_file_location("history_mlp_actor_critic", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_history_mlp_has_controlled_ab_shapes_and_finite_outputs():
    model = module.HistoryMLPActorCritic(
        num_actor_obs=270,
        num_critic_obs=238,
        num_one_step_obs=45,
        num_actions=12,
        initialization_seed=42,
    )
    history = torch.randn(32, 270)
    critic = torch.randn(32, 238)

    action_mean = model.act_inference(history)
    value = model.evaluate(critic)

    assert model.history_size == 6
    assert model.actor[0].weight.shape == (512, 270)
    assert model.critic[0].weight.shape == (512, 238)
    assert action_mean.shape == (32, 12)
    assert value.shape == (32, 1)
    assert torch.isfinite(action_mean).all()
    assert torch.isfinite(value).all()


def test_component_seed_preserves_global_rng_and_repeats_critic_initialization():
    torch.manual_seed(1234)
    expected_next = torch.rand(5)
    torch.manual_seed(1234)
    first = module.HistoryMLPActorCritic(270, 238, 45, 12, initialization_seed=7)
    actual_next = torch.rand(5)
    second = module.HistoryMLPActorCritic(270, 238, 45, 12, initialization_seed=7)

    assert torch.equal(expected_next, actual_next)
    for left, right in zip(first.critic.state_dict().values(), second.critic.state_dict().values()):
        assert torch.equal(left, right)
    for left, right in zip(first.actor.state_dict().values(), second.actor.state_dict().values()):
        assert torch.equal(left, right)


def test_wrong_history_dimension_is_rejected():
    try:
        module.HistoryMLPActorCritic(269, 238, 45, 12)
    except ValueError as error:
        assert "not divisible" in str(error)
    else:
        raise AssertionError("Invalid history schema was accepted")
