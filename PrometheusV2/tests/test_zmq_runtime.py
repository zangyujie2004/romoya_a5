from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from prometheus.runtime.scheduler import ActionScheduler, ActionTrace, ema_incoming
from prometheus.workflows.run_env import RunEnvSession, select_obs_keys, validate_reset_options


def test_queue_scheduler_sends_chunk_and_traces() -> None:
    target = _MemoryActionTarget()
    traces: list[ActionTrace] = []
    scheduler = ActionScheduler(
        action_target=target,
        action_hz=500.0,
        max_chunk_horizon=4,
        policy_low_watermark=0,
        trace_hook=traces.append,
    )

    scheduler.start()
    try:
        scheduler.submit(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        deadline = time.monotonic() + 1.0
        while len(target.actions) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(target.actions) == 2
        assert len(traces) == 2
        assert traces[-1].sent_action_count == 2
    finally:
        scheduler.stop()


def test_ema_incoming_smooths_from_last_sent_action() -> None:
    incoming = np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    smoothed = ema_incoming(
        pending=np.empty((0, 2), dtype=np.float32),
        incoming=incoming,
        last_sent_action=np.asarray([0.0, 0.0], dtype=np.float32),
        sent_action_count=10,
        alpha=0.5,
    )
    np.testing.assert_allclose(smoothed, [[1.0, 2.0], [3.5, 5.0]])


def test_run_env_session_steps_action_chunk_to_policy_boundary() -> None:
    obs = _FakeObsProvider()
    obs.put("robot/qpos", [0.0, 0.0], stamp_ns=1)
    robot_client = _FakeRobotClient(action_dim=2)
    session = RunEnvSession(
        robot_client=robot_client,
        obs_provider=obs,
        default_obs_keys=("robot/qpos",),
        action_hz=500.0,
        max_chunk_horizon=4,
        policy_low_watermark=0,
        reset_timeout_s=1.0,
        step_timeout_s=1.0,
    )

    reset = session.reset(obs_keys=())
    thread = threading.Thread(target=_publish_obs_after_delay, args=(obs, [3.0, 4.0], 2))
    thread.start()
    reply = session.step(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), obs_keys=())
    thread.join(timeout=1.0)
    session.close()

    assert not thread.is_alive()
    np.testing.assert_allclose(reset["obs"]["robot/qpos"], [0.0, 0.0])
    np.testing.assert_allclose(reply["obs"]["robot/qpos"], [3.0, 4.0])
    np.testing.assert_allclose(robot_client.actions, [[1.0, 2.0], [3.0, 4.0]])
    assert reply["info"]["sent_action_count"] == 2
    assert reply["info"]["scheduler_wait_ms"] >= 0.0
    assert reply["info"]["obs_wait_ms"] >= 0.0


def test_run_env_session_rejects_reset_expect_mismatch_before_scheduler_start() -> None:
    obs = _FakeObsProvider()
    obs.put("robot/qpos", [0.0, 0.0], stamp_ns=1)
    session = RunEnvSession(
        robot_client=_FakeRobotClient(action_dim=2),
        obs_provider=obs,
        default_obs_keys=("robot/qpos",),
        action_hz=500.0,
        max_chunk_horizon=4,
        policy_low_watermark=0,
        reset_timeout_s=1.0,
        step_timeout_s=1.0,
    )

    with pytest.raises(ValueError, match="action_hz mismatch"):
        session.reset(obs_keys=(), options={"expect": {"action_hz": 10.0}})
    assert session.closed


def test_run_env_session_reset_can_return_aligned_history() -> None:
    obs = _FakeObsProvider()
    obs.put("robot/qpos", [0.0, 0.0], stamp_ns=1)
    obs.put_history("robot/qpos", [[-1.0, -1.0], [0.0, 0.0]], stamps=[0, 1])
    session = RunEnvSession(
        robot_client=_FakeRobotClient(action_dim=2),
        obs_provider=obs,
        default_obs_keys=("robot/qpos",),
        action_hz=500.0,
        max_chunk_horizon=4,
        policy_low_watermark=0,
        reset_timeout_s=1.0,
        step_timeout_s=1.0,
    )

    reply = session.reset(
        obs_keys=(),
        options={"history_count": 2, "history_anchor": "robot/qpos", "history_slop_ms": 60.0},
    )
    session.close()

    assert len(reply["info"]["obs_history"]) == 2
    np.testing.assert_allclose(reply["info"]["obs_history"][0]["robot/qpos"], [-1.0, -1.0])
    assert reply["info"]["obs_history_timestamps"][1]["robot/qpos"] == 1
    assert reply["info"]["obs_history_frame_timestamps"] == [0, 1]


def test_select_obs_keys_and_validate_policy_boundary() -> None:
    assert select_obs_keys((), default_obs_keys=("robot/qpos",)) == ("robot/qpos",)
    assert select_obs_keys(("camera/base_0/rgb",), default_obs_keys=("robot/qpos",)) == ("camera/base_0/rgb",)
    validate_reset_options(
        {"expect": {"action_space": "abs_qpos", "action_dim": 14, "action_hz": 30.0, "action_chunk_horizon": 8}},
        action_space="abs_qpos",
        action_dim=14,
        action_hz=30.0,
        max_chunk_horizon=64,
        policy_low_watermark=0,
    )
    with pytest.raises(ValueError, match="policy_low_watermark"):
        validate_reset_options(
            {"expect": {"action_chunk_horizon": 4}},
            action_space="abs_qpos",
            action_dim=14,
            action_hz=30.0,
            max_chunk_horizon=64,
            policy_low_watermark=4,
        )


class _MemoryActionTarget:
    action_space = "abs_qpos"
    action_dim = 2

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def send_action(self, action: np.ndarray) -> None:
        self.actions.append(action.copy())


class _FakeRobotClient:
    action_space = "abs_qpos"

    def __init__(self, *, action_dim: int):
        self.action_dims = {self.action_space: int(action_dim)}
        self.actions: list[np.ndarray] = []

    def send_action(self, action: Any, *, timeout: float, wait: bool) -> dict[str, Any] | None:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        return {"accepted": True} if wait else None

    def spin_once(self, timeout_s: float = 0.0) -> None:
        pass


class _FakeObsProvider:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, SimpleNamespace] = {}

    def put(self, key: str, value: Any, *, stamp_ns: int) -> None:
        with self._lock:
            self._data[str(key)] = SimpleNamespace(
                value=np.asarray(value, dtype=np.float32),
                stamp_ns=int(stamp_ns),
            )
            self._history = getattr(self, "_history", {})
            self._history[str(key)] = [
                SimpleNamespace(value=np.asarray(value, dtype=np.float32), stamp_ns=int(stamp_ns))
            ]

    def put_history(self, key: str, values: Any, *, stamps: list[int]) -> None:
        with self._lock:
            self._history = getattr(self, "_history", {})
            self._history[str(key)] = [
                SimpleNamespace(value=np.asarray(value, dtype=np.float32), stamp_ns=int(stamp_ns))
                for value, stamp_ns in zip(values, stamps)
            ]

    def available(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._data))

    def latest(self, keys, *, timeout_s):
        return self._wait(keys, timeout_s=timeout_s, baselines=None)

    def next(self, keys, *, timeout_s, baselines=None):
        return self._wait(keys, timeout_s=timeout_s, baselines=baselines)

    def stamps(self, keys) -> dict[str, int]:
        with self._lock:
            return {str(key): int(self._data[str(key)].stamp_ns) for key in keys}

    def aligned_window(self, keys, *, anchor, count, slop_ms):
        selected = tuple(str(key) for key in keys)
        with self._lock:
            history = getattr(self, "_history", {})
            values = []
            stamps = []
            frame_stamps = []
            for index in range(int(count)):
                frame_values = {}
                frame_stamps_by_key = {}
                for key in selected:
                    sample = history[key][index]
                    frame_values[key] = sample.value.copy()
                    frame_stamps_by_key[key] = int(sample.stamp_ns)
                values.append(frame_values)
                stamps.append(frame_stamps_by_key)
                frame_stamps.append(int(history[str(anchor)][index].stamp_ns))
        return values, stamps, frame_stamps

    def _wait(self, keys, *, timeout_s, baselines):
        selected = tuple(str(key) for key in keys)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                if all(key in self._data for key in selected):
                    if baselines is None or all(self._data[key].stamp_ns != baselines.get(key) for key in selected):
                        return {key: self._data[key].value.copy() for key in selected}
            time.sleep(0.005)
        raise TimeoutError(f"obs timeout: {selected}")


def _publish_obs_after_delay(obs: _FakeObsProvider, value: list[float], stamp_ns: int) -> None:
    time.sleep(0.02)
    obs.put("robot/qpos", value, stamp_ns=stamp_ns)
