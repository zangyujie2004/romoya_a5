from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from prometheus.policy.humanego_policy import HumanEgoPolicy
from prometheus.sessions.data import DataSession


class FakeHumanEgoClient:
    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        return {
            "ok": True,
            "action_chunk": {
                "actions": np.ones((2, 14), dtype=np.float32),
                "action_space": "abs_eef",
                "hz": 10.0,
                "metadata": {"done": False},
            },
        }

    def close(self) -> None:
        self.closed = True


def test_humanego_policy_sends_init_then_base_payloads():
    client = FakeHumanEgoClient()
    policy = HumanEgoPolicy(remote={"server": "ws://localhost:2171"}, client=client)
    data = _data_session()

    first = policy.infer(data)
    second = policy.infer(data)

    assert first.action_space == "abs_eef"
    assert second.actions.shape == (2, 14)
    assert len(client.requests) == 2
    assert client.requests[0]["cmd"] == "infer"
    assert client.requests[0]["reset_episode"] is True
    assert "stereo" in client.requests[0]["observation"]
    assert "reset_episode" not in client.requests[1]
    assert "stereo" not in client.requests[1]["observation"]

    obs = client.requests[0]["observation"]
    assert obs["rgb_bgr"].shape == (2, 3, 3)
    assert obs["rgb_bgr"][0, 0].tolist() == [2, 1, 0]
    assert obs["stereo"]["left"].shape == (2, 3)
    assert obs["gripper_widths"] == {"left": 0.01, "right": 0.02}
    np.testing.assert_array_equal(obs["K"], np.eye(3))


def test_humanego_policy_fails_when_gripper_joint_is_missing():
    policy = HumanEgoPolicy(remote={"server": "ws://localhost:2171"}, client=FakeHumanEgoClient())
    data = _data_session(joint_names=["left_gripper"])

    with pytest.raises(ValueError, match="right_gripper"):
        policy.infer(data)


def _data_session(joint_names: list[str] | None = None) -> DataSession:
    stream_names = [
        "robot_state",
        "left_eef",
        "right_eef",
        "d405_color",
        "d405_color_info",
        "d405_infra1",
        "d405_infra2",
        "d405_infra1_info",
        "d405_infra2_info",
    ]
    topics = {
        name: {"topic": f"/{name}", "msg_type": "test_msgs/Fake"}
        for name in stream_names
    }
    node = DataSession(topics, history=4)
    stamp_ns = 123_000_000
    names = ["left_gripper", "right_gripper"] if joint_names is None else joint_names
    positions = [0.01, 0.02][: len(names)]
    node.ingest("robot_state", SimpleNamespace(name=names, position=positions), stamp_ns=stamp_ns)
    node.ingest("left_eef", _pose_msg(x=0.1), stamp_ns=stamp_ns)
    node.ingest("right_eef", _pose_msg(x=-0.1), stamp_ns=stamp_ns)
    node.ingest("d405_color", _image_msg(np.arange(18, dtype=np.uint8).reshape(2, 3, 3), "rgb8"), stamp_ns=stamp_ns)
    node.ingest("d405_color_info", _camera_info(), stamp_ns=stamp_ns)
    node.ingest("d405_infra1", _image_msg(np.arange(6, dtype=np.uint8).reshape(2, 3), "mono8"), stamp_ns=stamp_ns)
    node.ingest("d405_infra2", _image_msg(np.arange(6, dtype=np.uint8).reshape(2, 3), "mono8"), stamp_ns=stamp_ns)
    node.ingest("d405_infra1_info", _camera_info(), stamp_ns=stamp_ns)
    node.ingest("d405_infra2_info", _camera_info(tx=-0.05), stamp_ns=stamp_ns)
    return node


def _image_msg(array: np.ndarray, encoding: str) -> Any:
    array = np.ascontiguousarray(array)
    return SimpleNamespace(
        height=array.shape[0],
        width=array.shape[1],
        encoding=encoding,
        step=array.strides[0],
        data=array.tobytes(),
    )


def _camera_info(tx: float = 0.0) -> Any:
    K = np.eye(3, dtype=np.float64)
    P = np.zeros((3, 4), dtype=np.float64)
    P[:3, :3] = K
    P[0, 3] = tx
    return SimpleNamespace(k=K.reshape(-1).tolist(), p=P.reshape(-1).tolist())


def _pose_msg(*, x: float) -> Any:
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=0.0, z=0.2),
            orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
