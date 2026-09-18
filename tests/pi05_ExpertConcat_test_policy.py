"""Offline tests for prometheus.policy.pi05_ExpertConcat_policy (no robot, no model server).

    python -m pytest -q tests/pi05_ExpertConcat_test_policy.py
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import cv2
import msgpack
import numpy as np
import pytest

from prometheus.policy.pi05_ExpertConcat_policy import (
    DEFAULT_CAMERA_STREAMS,
    DEFAULT_TACTILE_STREAMS,
    JOINT_NAMES,
    OpenPIWebsocketClient,
    Pi05ExpertConcatPolicy,
)
from prometheus.policy.scheduler import ActionChunk

WINDOW = 8
SERVER_METADATA = {
    "policy": "pi05_ExpertConcat",
    "action_horizon": 50,
    "action_dim": 14,
    "tactile_history": WINDOW,
    "image_size": 224,
    "fps": 30,
    "prompt": "make a sandwich",
}


def _sample(name: str, data: Any, stamp_ns: int, encoding: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, data=data, stamp_ns=stamp_ns, recv_ns=stamp_ns, encoding=encoding)


def _frames(rng: np.random.Generator, count: int = WINDOW) -> list[SimpleNamespace]:
    frames = []
    for index in range(count):
        stamp = 1_000_000_000 + index * 33_333_333
        samples = {
            stream: _sample(stream, rng.integers(0, 256, (240, 320, 3), dtype=np.uint8), stamp, "rgb8")
            for stream in DEFAULT_CAMERA_STREAMS.values()
        }
        samples["robot_state"] = _sample(
            "robot_state",
            {
                "position": rng.normal(size=14).astype(np.float32),
                "velocity": np.zeros(14, dtype=np.float32),
                "effort": np.zeros(14, dtype=np.float32),
                "name": np.asarray(JOINT_NAMES, dtype=object),
            },
            stamp,
            "joint_state",
        )
        for stream in DEFAULT_TACTILE_STREAMS:
            samples[stream] = _sample(stream, rng.normal(size=(700, 6)).astype(np.float32), stamp, "pointcloud2")
        frames.append(SimpleNamespace(stamp_ns=stamp, samples=samples, skew_ms={name: 0.0 for name in samples}))
    return frames


class FakeData:
    def __init__(self, frames: list[SimpleNamespace]):
        self.frames = frames
        self.calls: list[dict[str, Any]] = []

    def window_numpy(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return self.frames


class FakeClient:
    def __init__(self, metadata: dict[str, Any] | None = None, actions: np.ndarray | None = None):
        self.metadata = dict(SERVER_METADATA if metadata is None else metadata)
        self.actions = np.arange(50 * 14, dtype=np.float32).reshape(50, 14) / 1000.0 if actions is None else actions
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def connect(self, *, timeout_s: float) -> dict[str, Any]:
        return dict(self.metadata)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(obs)
        return {"actions": self.actions, "policy_timing": {"infer_ms": 80.0}, "server_timing": {"infer_ms": 85.0}}

    def close(self) -> None:
        self.closed = True


def _policy(client: Any, **kwargs: Any) -> Pi05ExpertConcatPolicy:
    return Pi05ExpertConcatPolicy(inputs={"window_size": WINDOW}, client=client, **kwargs)


def test_observation_matches_preprocess_pi05_layout():
    rng = np.random.default_rng(0)
    frames = _frames(rng)
    obs = _policy(FakeClient()).build_observation(frames)

    latest = frames[-1].samples
    for bundle, stream in DEFAULT_CAMERA_STREAMS.items():
        expected = cv2.resize(latest[stream].data, (224, 224), interpolation=cv2.INTER_AREA)
        assert obs["images"][bundle].dtype == np.uint8
        np.testing.assert_array_equal(obs["images"][bundle], expected)
    np.testing.assert_array_equal(obs["state"], latest["robot_state"].data["position"])

    tactile = obs["tactile"]
    assert tactile.shape == (WINDOW, 35, 20, 24) and tactile.dtype == np.float32
    for t in (0, WINDOW - 1):
        for sensor, stream in enumerate(DEFAULT_TACTILE_STREAMS):
            points = frames[t].samples[stream].data  # (700, 6): point = row * 20 + col
            np.testing.assert_array_equal(tactile[t, 3, 7, sensor * 6 : sensor * 6 + 6], points[3 * 20 + 7])
            np.testing.assert_array_equal(tactile[t, ..., sensor * 6 : sensor * 6 + 6], points.reshape(35, 20, 6))
    assert "prompt" not in obs
    assert _policy(FakeClient(), prompt="make a sandwich").build_observation(frames)["prompt"] == "make a sandwich"


def test_infer_returns_abs_qpos_chunk():
    frames = _frames(np.random.default_rng(1))
    data = FakeData(frames)
    client = FakeClient()
    policy = _policy(client, n_action_steps=25)

    chunk = policy.infer(data)

    assert isinstance(chunk, ActionChunk)
    assert chunk.action_space == "abs_qpos" and chunk.hz == pytest.approx(30.0)
    np.testing.assert_array_equal(chunk.actions, client.actions[:25])
    assert chunk.metadata["action_horizon"] == 50 and chunk.metadata["n_action_steps"] == 25
    np.testing.assert_array_equal(chunk.metadata["timestamp"]["frame"], [frame.stamp_ns for frame in frames])
    call = data.calls[0]
    assert call["anchor"] == "base_0_color" and call["count"] == WINDOW and call["stride"] == 1
    assert set(call["names"]) == {*DEFAULT_CAMERA_STREAMS.values(), "robot_state", *DEFAULT_TACTILE_STREAMS}
    assert policy.status()["connected"] is True and policy.status()["chunks"] == 1


@pytest.mark.parametrize(
    ("metadata_update", "kwargs", "message"),
    [
        ({"tactile_history": 4}, {}, "tactile_history"),
        ({"fps": 50}, {}, "fps"),
        ({"action_horizon": 20}, {"n_action_steps": 50}, "action_horizon"),
        ({"prompt": None}, {}, "prompt"),
    ],
)
def test_server_metadata_mismatch_fails_before_inference(metadata_update, kwargs, message):
    client = FakeClient({**SERVER_METADATA, **metadata_update})
    policy = _policy(client, **kwargs)
    with pytest.raises(ValueError, match=message):
        policy.wait_ready(timeout_s=1.0)
    assert client.closed and not client.requests


def test_rejects_joint_order_and_resolution_that_differ_from_training():
    frames = _frames(np.random.default_rng(2))
    state = frames[-1].samples["robot_state"].data
    state["name"] = np.asarray([*JOINT_NAMES[7:], *JOINT_NAMES[:7]], dtype=object)
    with pytest.raises(ValueError, match="joint order"):
        _policy(FakeClient()).build_observation(frames)

    frames = _frames(np.random.default_rng(3))
    frames[-1].samples["base_0_color"] = _sample("base_0_color", np.zeros((480, 640, 3), np.uint8), 0, "rgb8")
    with pytest.raises(ValueError, match="source_image_hw"):
        _policy(FakeClient()).build_observation(frames)


def test_openpi_wire_format_over_websocket():
    """The fake server decodes and encodes with plain msgpack in openpi_client.msgpack_numpy's format."""
    from websockets.sync.server import serve

    actions = np.linspace(-1.0, 1.0, 50 * 14, dtype=np.float32).reshape(50, 14)
    received: list[dict] = []

    def handler(ws):
        ws.send(msgpack.packb(SERVER_METADATA))
        obs = msgpack.unpackb(ws.recv())
        received.append(obs)
        ws.send(
            msgpack.packb(
                {
                    "actions": {
                        b"__ndarray__": True,
                        b"data": actions.tobytes(),
                        b"dtype": actions.dtype.str,
                        b"shape": list(actions.shape),
                    },
                    "policy_timing": {"infer_ms": 12.5},
                }
            )
        )
        ws.recv()  # second request: answer with a server-side error
        ws.send("Traceback: boom")

    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenPIWebsocketClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
        assert client.connect(timeout_s=5.0) == SERVER_METADATA
        obs = _policy(FakeClient()).build_observation(_frames(np.random.default_rng(4)))
        reply = client.infer(obs)
        np.testing.assert_array_equal(reply["actions"], actions)
        assert reply["policy_timing"]["infer_ms"] == pytest.approx(12.5)

        state = received[0]["state"]
        assert state[b"__ndarray__"] is True and state[b"dtype"] == "<f4" and list(state[b"shape"]) == [14]
        image = received[0]["images"]["base_0"]
        assert image[b"dtype"] == "|u1" and list(image[b"shape"]) == [224, 224, 3]
        assert list(received[0]["tactile"][b"shape"]) == [WINDOW, 35, 20, 24]

        with pytest.raises(RuntimeError, match="boom"):
            client.infer(obs)
        client.close()
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
