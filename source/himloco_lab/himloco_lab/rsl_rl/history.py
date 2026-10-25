"""Pure Torch helpers for HIMLoco frame-major history buffers."""

from __future__ import annotations

import torch


def repeat_frames(observation: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Repeat one observation into a newest-first, frame-major history."""
    if observation.ndim != 2:
        raise ValueError(f"Expected a rank-2 observation tensor, got shape {tuple(observation.shape)}")
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    return observation.repeat(1, frame_count)


def initialize_history_rows(
    history_buffer: torch.Tensor,
    observation: torch.Tensor,
    frame_count: int,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Fill every history slot from a reset observation, in place."""
    if env_ids is None:
        history_buffer.copy_(repeat_frames(observation, frame_count))
    elif env_ids.numel() > 0:
        history_buffer[env_ids] = repeat_frames(observation[env_ids], frame_count)
