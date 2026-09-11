"""Stateful smoothing of policy targets before velocity interpolation."""
from __future__ import annotations

import numpy as np

from prometheus.policy.scheduler import ActionChunk


class PolicyActionSmoother:
    """EMA at policy action rate; alpha weights the previous smoothed target."""

    def __init__(self, *, enabled: bool = False, alpha: float = 0.6, gripper: bool = False):
        if not np.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("action smoothing alpha must be in [0, 1)")
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.gripper = bool(gripper)
        self._previous: np.ndarray | None = None

    def reset(self, current: np.ndarray) -> None:
        values = np.asarray(current, dtype=np.float64)
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError("policy smoothing requires finite 14-axis feedback")
        self._previous = values.copy()

    def apply(self, chunk: ActionChunk) -> ActionChunk:
        if not self.enabled:
            return chunk
        if chunk.action_space != "abs_qpos" or chunk.action_dim != 14:
            raise ValueError("policy smoothing supports only 14-axis abs_qpos")
        if self._previous is None:
            raise RuntimeError("seed policy smoothing from feedback before the first chunk")
        weights = np.full(14, self.alpha, dtype=np.float64)
        if not self.gripper:
            weights[[6, 13]] = 0.0
        previous = self._previous.copy()
        output = np.empty_like(chunk.actions)
        for index, target in enumerate(chunk.actions):
            previous = weights * previous + (1.0 - weights) * target
            output[index] = previous
        self._previous = previous
        return ActionChunk(output, action_space=chunk.action_space, hz=chunk.hz,
                           metadata={**chunk.metadata, "action_smoothing": self.status()})

    def status(self) -> dict:
        return {"enabled": self.enabled, "method": "ema", "alpha": self.alpha,
                "gripper": self.gripper}
