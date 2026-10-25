import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "himloco_lab"
    / "rsl_rl"
    / "history.py"
)
SPEC = importlib.util.spec_from_file_location("himloco_history", MODULE_PATH)
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)


def test_repeat_frames_is_newest_first_frame_major():
    observation = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    history_buffer = history.repeat_frames(observation, frame_count=3)

    assert torch.equal(
        history_buffer,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
            ]
        ),
    )


def test_selected_reset_replaces_every_policy_and_critic_history_slot():
    policy_buffer = torch.full((2, 9), -1.0)
    critic_buffer = torch.full((2, 4), -2.0)
    policy = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    critic = torch.tensor([[30.0, 31.0], [40.0, 41.0]])
    env_ids = torch.tensor([1])

    history.initialize_history_rows(policy_buffer, policy, frame_count=3, env_ids=env_ids)
    history.initialize_history_rows(critic_buffer, critic, frame_count=2, env_ids=env_ids)

    assert torch.equal(policy_buffer[0], torch.full((9,), -1.0))
    assert torch.equal(
        policy_buffer[1],
        torch.tensor([20.0, 21.0, 22.0, 20.0, 21.0, 22.0, 20.0, 21.0, 22.0]),
    )
    assert torch.equal(critic_buffer[0], torch.full((4,), -2.0))
    assert torch.equal(critic_buffer[1], torch.tensor([40.0, 41.0, 40.0, 41.0]))
