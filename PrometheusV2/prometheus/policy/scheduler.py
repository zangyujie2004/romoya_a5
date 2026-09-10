from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ActionChunk:
    """A finite command sequence in one robot action space."""

    actions: np.ndarray
    action_space: str
    hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f"actions must be 1D or 2D, got shape {actions.shape}")
        if actions.shape[0] <= 0 or actions.shape[1] <= 0:
            raise ValueError(f"actions must be non-empty, got shape {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain NaN or inf")
        if not str(self.action_space):
            raise ValueError("action_space must be non-empty")
        if float(self.hz) <= 0:
            raise ValueError("hz must be positive")

        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "action_space", str(self.action_space))
        object.__setattr__(self, "hz", float(self.hz))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def dt(self) -> float:
        return 1.0 / self.hz

    def __len__(self) -> int:
        return self.horizon


@dataclass(frozen=True)
class ActionContext:
    chunk: ActionChunk
    index: int
    previous_action: np.ndarray | None

    @property
    def dt(self) -> float:
        return self.chunk.dt

    @property
    def action_space(self) -> str:
        return self.chunk.action_space


class ActionFilter:
    """One independently switchable action transformation or guard."""

    def __init__(self, name: str, *, enabled: bool = True):
        if not name:
            raise ValueError("filter name must be non-empty")
        self.name = str(name)
        self.enabled = bool(enabled)

    def reset(self) -> None:
        pass

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        return action


class EmaFilter(ActionFilter):
    """Elementwise exponential smoothing over scheduled actions."""

    def __init__(self, alpha: float, *, name: str = "ema", enabled: bool = True):
        super().__init__(name, enabled=enabled)
        if not 0.0 <= float(alpha) < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        if self._state is None:
            self._state = action.copy()
            return action
        out = self.alpha * self._state + (1.0 - self.alpha) * action
        self._state = out.copy()
        return out.astype(np.float32)


class MaxDeltaFilter(ActionFilter):
    """Clamp elementwise action changes relative to the previous action."""

    def __init__(self, max_delta: float | np.ndarray, *, name: str = "max_delta", enabled: bool = True):
        super().__init__(name, enabled=enabled)
        value = np.asarray(max_delta, dtype=np.float32)
        if not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError("max_delta must be finite and positive")
        self.max_delta = value

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        previous = context.previous_action
        if previous is None:
            return action
        delta = np.clip(action - previous, -self.max_delta, self.max_delta)
        return (previous + delta).astype(np.float32)


class ActionScheduler:
    """Executes ActionChunk through a robot control client."""

    def __init__(
        self,
        robot_client: Any,
        *,
        filters: list[ActionFilter] | None = None,
        dry_run: bool = False,
        wait: bool = True,
        timeout_s: float = 3.0,
        history: int = 1024,
    ):
        if history <= 0:
            raise ValueError("history must be positive")
        self.robot_client = robot_client
        self.dry_run = bool(dry_run)
        self.wait = bool(wait)
        self.timeout_s = float(timeout_s)
        self.history: deque[np.ndarray] = deque(maxlen=int(history))
        self._filters: dict[str, ActionFilter] = {}
        self._previous_action: np.ndarray | None = None
        for item in filters or []:
            self.add_filter(item)

    @property
    def filters(self) -> tuple[ActionFilter, ...]:
        return tuple(self._filters.values())

    def add_filter(self, item: ActionFilter) -> None:
        if item.name in self._filters:
            raise ValueError(f"duplicate action filter {item.name!r}")
        self._filters[item.name] = item

    def set_filter_enabled(self, name: str, enabled: bool) -> None:
        item = self._filters[name]
        enabled = bool(enabled)
        if item.enabled != enabled:
            item.reset()
        item.enabled = enabled

    def reset(self) -> None:
        self._previous_action = None
        for item in self._filters.values():
            item.reset()

    def execute(
        self,
        chunk: ActionChunk,
        *,
        steps: int | None = None,
        on_action: Callable[[np.ndarray, ActionContext], None] | None = None,
    ) -> int:
        if chunk.action_space != str(self.robot_client.action_space):
            raise ValueError(
                f"chunk action_space={chunk.action_space!r} does not match "
                f"robot action_space={self.robot_client.action_space!r}"
            )
        if self._previous_action is None:
            self._previous_action = self._current_action_from_metadata(chunk)
        n_steps = chunk.horizon if steps is None else min(int(steps), chunk.horizon)
        if n_steps <= 0:
            raise ValueError("steps must be positive")

        sent = 0
        next_time = time.monotonic()
        for index in range(n_steps):
            action = chunk.actions[index].copy()
            context = ActionContext(chunk=chunk, index=index, previous_action=self._previous_action)
            action = self._apply_filters(action, context)
            self._send(action)
            if on_action is not None:
                on_action(action.copy(), context)
            self._previous_action = action.copy()
            self.history.append(action.copy())
            sent += 1

            next_time += chunk.dt
            delay = next_time - time.monotonic()
            if delay > 0 and index + 1 < n_steps:
                self._spin_sleep(delay)
        return sent

    def _apply_filters(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        for item in self._filters.values():
            if not item.enabled:
                continue
            action = np.asarray(item.apply(action, context), dtype=np.float32).reshape(-1)
            if action.shape != (context.chunk.action_dim,):
                raise ValueError(
                    f"filter {item.name!r} returned shape {action.shape}, "
                    f"expected {(context.chunk.action_dim,)}"
                )
            if not np.all(np.isfinite(action)):
                raise ValueError(f"filter {item.name!r} returned NaN or inf")
        return action

    def _current_action_from_metadata(self, chunk: ActionChunk) -> np.ndarray | None:
        value = chunk.metadata.get("current_action")
        if value is None:
            return None
        current = np.asarray(value, dtype=np.float32).reshape(-1)
        if current.shape != (chunk.action_dim,):
            raise ValueError(
                f"metadata current_action shape {current.shape} does not match "
                f"chunk action dim {(chunk.action_dim,)}"
            )
        if not np.all(np.isfinite(current)):
            raise ValueError("metadata current_action contains NaN or inf")
        return current.copy()

    def _send(self, action: np.ndarray) -> dict[str, Any] | None:
        if self.dry_run:
            print(f"[ActionScheduler] dry_run action={action.tolist()}", flush=True)
            return None
        result = self.robot_client.send_action(action, timeout=self.timeout_s, wait=self.wait)
        if self.wait:
            if not isinstance(result, Mapping):
                raise RuntimeError("send_action must return a mapping when wait=True")
            accepted = bool(result.get("accepted", False))
            if not accepted:
                raise RuntimeError(str(result.get("message", "robot rejected action")))
        return result

    def _spin_sleep(self, duration_s: float) -> None:
        deadline = time.monotonic() + float(duration_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.robot_client.spin_once(min(0.01, remaining))
