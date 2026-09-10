from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import time

import numpy as np
import pytest

from prometheus.policy.fm_policy import DEFAULT_STREAM_NAMES, FMPolicy
from prometheus.sessions.data import DataSession


class FakeFMRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.use_tactile = False

    def infer_from_window(self, frames, *, robot=None, num_inference_steps=None, solver=None):
        self.calls.append(
            {
                "frame_count": len(frames),
                "robot": dict(robot or {}),
                "num_inference_steps": num_inference_steps,
                "solver": solver,
            }
        )
        from dataclasses import dataclass, field

        @dataclass(frozen=True)
        class _Chunk:
            actions: np.ndarray
            action_space: str
            hz: float
            metadata: dict[str, Any] = field(default_factory=dict)

        return _Chunk(
            actions=np.ones((32, 14), dtype=np.float32),
            action_space="abs_eef",
            hz=30.0,
            metadata={"action_horizon": 32, "window_size": 8},
        )


def test_fm_policy_calls_runtime_and_returns_action_chunk():
    runtime = FakeFMRuntime()
    policy = FMPolicy(
        inputs={"window_size": 8, "stride": 1, "slop_ms": 50.0},
        inference={"run_dir": "unused"},
        robot={"gripper_width_m": 0.082},
        runtime=runtime,
    )
    data = _fm_data_session(frame_count=8)

    chunk = policy.infer(data)

    assert chunk.action_space == "abs_eef"
    assert chunk.hz == pytest.approx(30.0)
    assert chunk.actions.shape == (32, 14)
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["frame_count"] == 8
    assert runtime.calls[0]["robot"]["gripper_width_m"] == pytest.approx(0.082)


def test_fm_policy_action_hz_override():
    runtime = FakeFMRuntime()
    policy = FMPolicy(
        inputs={"window_size": 8, "stride": 1, "slop_ms": 50.0},
        inference={"run_dir": "unused"},
        action_hz=15.0,
        runtime=runtime,
    )
    data = _fm_data_session(frame_count=8)

    chunk = policy.infer(data)

    assert chunk.hz == pytest.approx(15.0)


def test_fm_policy_requires_enough_anchor_history():
    runtime = FakeFMRuntime()
    policy = FMPolicy(inputs={"window_size": 8}, inference={"run_dir": "unused"}, runtime=runtime)
    data = _fm_data_session(frame_count=4)

    with pytest.raises(ValueError, match="needs 8 samples"):
        policy.infer(data)


def test_fm_policy_slices_actions_by_n_action_steps():
    runtime = FakeFMRuntime()
    policy = FMPolicy(
        inputs={"window_size": 8},
        inference={"run_dir": "unused", "n_action_steps": 8},
        runtime=runtime,
    )
    data = _fm_data_session(frame_count=8)

    chunk = policy.infer(data)

    assert chunk.actions.shape == (8, 14)
    assert chunk.metadata["n_action_steps"] == 8
    assert chunk.metadata["n_action_steps_requested"] == 8


def test_fm_policy_default_stream_names_match_fm_layout():
    assert DEFAULT_STREAM_NAMES == (
        "base_0_color",
        "left_wrist_0_color",
        "right_wrist_0_color",
        "robot_state",
        "left_eef",
        "right_eef",
    )


def test_fm_policy_filters_tactile_streams_for_vision_only_runtime():
    runtime = FakeFMRuntime()
    policy = FMPolicy(
        inputs={
            "stream_names": list(DEFAULT_STREAM_NAMES),
            "tactile_stream_names": [
                "left_wrist_0_tactile_flow",
                "left_wrist_1_tactile_flow",
                "right_wrist_0_tactile_flow",
                "right_wrist_1_tactile_flow",
            ],
        },
        inference={"run_dir": "unused"},
        runtime=runtime,
    )
    assert policy._active_stream_names() == DEFAULT_STREAM_NAMES
    assert len(policy.stream_names) == len(DEFAULT_STREAM_NAMES) + 4


def _fm_data_session(*, frame_count: int) -> DataSession:
    stream_names = list(DEFAULT_STREAM_NAMES)
    topics = {
        name: {"topic": f"/{name}", "msg_type": "test_msgs/Fake"}
        for name in stream_names
    }
    node = DataSession(topics, history=max(32, frame_count + 4), bridge={"enabled": False}, numpy={"enabled": True})
    node.start()
    for idx in range(frame_count):
        stamp_ns = (idx + 1) * 33_000_000
        node.ingest(
            "robot_state",
            SimpleNamespace(
                name=["left_gripper", "right_gripper"],
                position=[0.01, 0.02],
                velocity=[0.0, 0.0],
                effort=[0.0, 0.0],
            ),
            stamp_ns=stamp_ns,
        )
        node.ingest("left_eef", _pose_msg(x=0.1 + idx * 0.01), stamp_ns=stamp_ns)
        node.ingest("right_eef", _pose_msg(x=-0.1 - idx * 0.01), stamp_ns=stamp_ns)
        for camera in ("base_0_color", "left_wrist_0_color", "right_wrist_0_color"):
            node.ingest(
                camera,
                _image_msg(np.zeros((4, 4, 3), dtype=np.uint8), "rgb8"),
                stamp_ns=stamp_ns,
            )
    _wait_numpy_ready(node, stream_names, frame_count)
    return node


def _wait_numpy_ready(data: DataSession, names: list[str], count: int) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        counts = data.numpy_counts()
        if all(counts.get(name, 0) >= count for name in names):
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for numpy counts: {data.numpy_counts()}")


def _image_msg(array: np.ndarray, encoding: str) -> Any:
    array = np.ascontiguousarray(array)
    return SimpleNamespace(
        height=array.shape[0],
        width=array.shape[1],
        encoding=encoding,
        step=array.strides[0],
        data=array.tobytes(),
    )


def _pose_msg(*, x: float) -> Any:
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=0.0, z=0.2),
            orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
