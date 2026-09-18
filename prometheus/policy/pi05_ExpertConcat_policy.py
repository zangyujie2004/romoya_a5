"""pi0.5 + tactile (expert_concat) policy adapter for rollout_sync.

The model runs in a separate openpi process (tactile_vla_post_training/expert_concat/scripts/
pi05_ExpertConcat_serve_policy.py). This adapter builds one observation from the data session, sends
it over openpi's websocket protocol and returns the predicted absolute joint targets as an
ActionChunk for the existing abs_qpos path (scheduler limits, smoothing and gripper handling apply).

The observation repeats the per-frame processing preprocess_pi05 used to build the training data:
- images: the latest base_0 / left_wrist_0 / right_wrist_0 RGB frames resized to 224 x 224 with
  cv2.INTER_AREA;
- state: robot_state positions in stream order (left joint 1-6, left gripper, right joint 1-6, right
  gripper), the layout the recorder writes to robot_state_dict.pkl;
- tactile: the Xense flow point clouds of the last `window_size` anchor frames, each (700, 6)
  [x, y, z, dx, dy, dz] reshaped to (35, 20, 6) and concatenated over the sensors in the
  preprocess_pi05 order (left_wrist_0, left_wrist_1, right_wrist_0, right_wrist_1) -> (T, 35, 20, 24).
Normalization, prompt tokenization and the tactile channel selection run on the server, as in training.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Mapping, Sequence

import numpy as np

from prometheus.policy.scheduler import ActionChunk

DEFAULT_CAMERA_STREAMS = {
    "base_0": "base_0_color",
    "left_wrist_0": "left_wrist_0_color",
    "right_wrist_0": "right_wrist_0_color",
}
DEFAULT_STATE_STREAM = "robot_state"
DEFAULT_TACTILE_STREAMS = (
    "left_wrist_0_tactile_flow",
    "left_wrist_1_tactile_flow",
    "right_wrist_0_tactile_flow",
    "right_wrist_1_tactile_flow",
)
JOINT_NAMES = (
    *(f"left_joint_{index}" for index in range(1, 7)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(1, 7)),
    "right_gripper",
)
NUM_JOINTS = len(JOINT_NAMES)
TACTILE_GRID = (35, 20, 6)


class Pi05ExpertConcatPolicy:
    """Remote pi0.5 expert_concat policy returning `abs_qpos` action chunks."""

    def __init__(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        remote: Mapping[str, Any] | None = None,
        prompt: str | None = None,
        n_action_steps: int | None = None,
        action_hz: float = 30.0,
        action_space: str = "abs_qpos",
        client: Any | None = None,
    ):
        inputs = dict(inputs or {})
        remote = dict(remote or {})
        self.anchor = str(inputs.get("anchor", "base_0_color"))
        self.slop_ms = float(inputs.get("slop_ms", 60.0))
        self.window_size = int(inputs.get("window_size", 8))
        self.stride = int(inputs.get("stride", 1))
        self.wait_latest = bool(inputs.get("wait_latest", True))
        self.wait_latest_timeout_ms = float(inputs.get("wait_latest_timeout_ms", 5.0))
        self.image_size = int(inputs.get("image_size", 224))
        source_hw = inputs.get("source_image_hw", (240, 320))
        self.source_image_hw = None if source_hw is None else tuple(int(v) for v in source_hw)
        self.camera_streams = {
            str(bundle): str(stream)
            for bundle, stream in dict(inputs.get("camera_streams", DEFAULT_CAMERA_STREAMS)).items()
        }
        self.state_stream = str(inputs.get("state_stream", DEFAULT_STATE_STREAM))
        self.tactile_streams = _names(inputs.get("tactile_streams", DEFAULT_TACTILE_STREAMS))
        self.stream_names = tuple(
            dict.fromkeys([*self.camera_streams.values(), self.state_stream, *self.tactile_streams])
        )
        if self.window_size < 1 or self.stride < 1:
            raise ValueError("inputs.window_size and inputs.stride must be positive")
        if self.anchor not in self.stream_names:
            raise ValueError(f"inputs.anchor {self.anchor!r} must be one of the input streams {self.stream_names}")

        self.prompt = None if prompt is None or not str(prompt).strip() else str(prompt)
        self.n_action_steps = None if n_action_steps is None else int(n_action_steps)
        if self.n_action_steps is not None and self.n_action_steps < 1:
            raise ValueError("n_action_steps must be positive")
        self.action_hz = float(action_hz)
        self.action_space = str(action_space)

        self.server = str(remote.get("server", "ws://127.0.0.1:8000"))
        self.connect_timeout_s = float(remote.get("connect_timeout_s", 10.0))
        self.request_timeout_s = float(remote.get("request_timeout_s", 10.0))
        self._client = client if client is not None else OpenPIWebsocketClient(
            self.server, request_timeout_s=self.request_timeout_s
        )
        self.server_metadata: dict[str, Any] | None = None
        self._chunks = 0
        self._last_timing_ms: dict[str, float] = {}

    def wait_ready(self, timeout_s: float) -> None:
        self._ensure_connected(timeout_s)

    def infer(self, data: Any) -> ActionChunk:
        self._ensure_connected(self.connect_timeout_s)
        frames = data.window_numpy(
            anchor=self.anchor,
            names=self.stream_names,
            count=self.window_size,
            stride=self.stride,
            slop_ms=self.slop_ms,
            wait_latest=self.wait_latest,
            timeout_ms=self.wait_latest_timeout_ms,
        )
        obs = self.build_observation(frames)

        start = time.perf_counter()
        response = self._client.infer(obs)
        round_trip_ms = (time.perf_counter() - start) * 1000.0
        actions, horizon = self._actions_from_response(response)

        self._last_timing_ms = {"round_trip": round_trip_ms}
        for key in ("policy_timing", "server_timing"):
            infer_ms = dict(response.get(key) or {}).get("infer_ms")
            if infer_ms is not None:
                self._last_timing_ms[key.split("_")[0]] = float(infer_ms)
        self._chunks += 1
        return ActionChunk(
            actions,
            action_space=self.action_space,
            hz=self.action_hz,
            metadata={
                "timestamp": _timestamp_dict(frames),
                "action_horizon": horizon,
                "n_action_steps": int(actions.shape[0]),
                "timing_ms": dict(self._last_timing_ms),
            },
        )

    def build_observation(self, frames: Sequence[Any]) -> dict[str, Any]:
        if len(frames) != self.window_size:
            raise ValueError(f"expected {self.window_size} frames, got {len(frames)}")
        latest = _samples(frames[-1])
        images = {bundle: self._image(_sample(latest, stream)) for bundle, stream in self.camera_streams.items()}
        tactile = np.stack(
            [
                np.concatenate([_tactile_grid(_sample(_samples(frame), n)) for n in self.tactile_streams], axis=-1)
                for frame in frames
            ],
            axis=0,
        )
        obs: dict[str, Any] = {
            "images": images,
            "state": _joint_positions(_sample(latest, self.state_stream)),
            "tactile": np.ascontiguousarray(tactile, dtype=np.float32),
        }
        if self.prompt is not None:
            obs["prompt"] = self.prompt
        return obs

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self._client.close()

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "server": self.server,
            "connected": self.server_metadata is not None,
            "server_metadata": _json_scalars(self.server_metadata or {}),
            "anchor": self.anchor,
            "slop_ms": self.slop_ms,
            "window_size": self.window_size,
            "stride": self.stride,
            "stream_names": list(self.stream_names),
            "prompt": self.prompt,
            "n_action_steps": self.n_action_steps,
            "action_hz": self.action_hz,
            "action_space": self.action_space,
            "chunks": self._chunks,
            "last_timing_ms": dict(self._last_timing_ms),
        }

    def _ensure_connected(self, timeout_s: float) -> None:
        if self.server_metadata is not None:
            return
        metadata = dict(self._client.connect(timeout_s=timeout_s) or {})
        try:
            self._check_server_metadata(metadata)
        except Exception:
            self._client.close()
            raise
        self.server_metadata = metadata
        print(f"[pi05_ExpertConcat] connected to {self.server}: {_json_scalars(metadata)}", flush=True)

    def _check_server_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Fail before the robot moves if the server was trained with a different input/output layout."""
        expected = {
            "tactile_history": self.window_size,
            "action_dim": NUM_JOINTS,
            "image_size": self.image_size,
        }
        for key, value in expected.items():
            if metadata.get(key) is not None and int(metadata[key]) != value:
                raise ValueError(f"policy server {key}={metadata[key]} but this adapter uses {value}")
        fps = metadata.get("fps")
        if fps is not None and abs(float(fps) - self.action_hz) > 1e-6:
            raise ValueError(f"policy server was trained at {fps} fps but action_hz={self.action_hz}")
        horizon = metadata.get("action_horizon")
        if horizon is not None and self.n_action_steps is not None and self.n_action_steps > int(horizon):
            raise ValueError(f"n_action_steps={self.n_action_steps} exceeds the server action_horizon={horizon}")
        if self.prompt is None and not metadata.get("prompt"):
            raise ValueError("policy server has no default prompt; set policy.prompt to the training prompt")

    def _actions_from_response(self, response: Any) -> tuple[np.ndarray, int]:
        if not isinstance(response, Mapping) or "actions" not in response:
            raise ValueError("policy server response has no 'actions'")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != NUM_JOINTS:
            raise ValueError(f"policy server actions must be (horizon, {NUM_JOINTS}), got {actions.shape}")
        horizon = int(actions.shape[0])
        count = horizon if self.n_action_steps is None else self.n_action_steps
        if count > horizon:
            raise ValueError(f"n_action_steps={count} exceeds the returned horizon {horizon}")
        return actions[:count].copy(), horizon

    def _image(self, sample: Any) -> np.ndarray:
        import cv2

        encoding = str(getattr(sample, "encoding", "rgb8")).lower()
        rgb = np.asarray(sample.data)
        if encoding != "rgb8" or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"{_sample_name(sample)}: expected rgb8 HxWx3 uint8, got {encoding} {rgb.dtype} {rgb.shape}"
            )
        if self.source_image_hw is not None and rgb.shape[:2] != self.source_image_hw:
            raise ValueError(
                f"{_sample_name(sample)}: camera frames are {rgb.shape[:2]} but the training data was "
                f"recorded at {self.source_image_hw} (inputs.source_image_hw)"
            )
        if rgb.shape[:2] != (self.image_size, self.image_size):
            # Same call as preprocess_pi05 (pipeline/process.py, interpolation "area").
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb)


class OpenPIWebsocketClient:
    """Client for openpi's WebsocketPolicyServer (same protocol as openpi_client.WebsocketClientPolicy).

    The server sends its metadata as soon as the connection opens. Each request and reply is one
    msgpack message with numpy arrays encoded as in openpi_client.msgpack_numpy (bytes keys). A text
    reply carries the server-side traceback.
    """

    def __init__(self, server: str, *, request_timeout_s: float | None = 10.0):
        self.server = server if server.startswith("ws") else f"ws://{server}"
        self.request_timeout_s = request_timeout_s
        self._ws: Any | None = None
        self._stack: contextlib.ExitStack | None = None

    def connect(self, *, timeout_s: float) -> dict[str, Any]:
        from websockets.sync.client import connect

        self.close()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            wait_s = max(1.0, deadline - time.monotonic())
            stack = contextlib.ExitStack()
            try:
                # Entered as a context manager (kept open by the stack): websockets >= 17 warns otherwise.
                ws = stack.enter_context(connect(self.server, compression=None, max_size=None, open_timeout=wait_s))
            except OSError as exc:  # refused / unreachable / timed out: the server may still be loading
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"cannot reach policy server {self.server}: {exc}") from exc
                time.sleep(0.5)
                continue
            try:
                raw = ws.recv(timeout=wait_s)
                if isinstance(raw, str):
                    raise RuntimeError(f"policy server sent text instead of metadata:\n{raw}")
                metadata = unpackb(raw)
            except BaseException:
                stack.close()
                raise
            self._ws, self._stack = ws, stack
            return dict(metadata)

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("policy server is not connected")
        try:
            self._ws.send(packb(obs))
            raw = self._ws.recv(timeout=self.request_timeout_s)
        except BaseException:
            # A half-finished request leaves the connection in an unknown state.
            self.close()
            raise
        if isinstance(raw, str):
            raise RuntimeError(f"error in the policy server:\n{raw}")
        return unpackb(raw)

    def close(self) -> None:
        stack, self._stack, self._ws = self._stack, None, None
        if stack is not None:
            try:
                stack.close()
            except Exception:
                pass


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    import msgpack

    return msgpack.packb(obj, default=_pack_array)


def unpackb(data: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(data, object_hook=_unpack_array)


def _samples(frame: Any) -> Mapping[str, Any]:
    samples = getattr(frame, "samples", None)
    if not isinstance(samples, Mapping):
        raise TypeError("frame must expose a samples mapping")
    return samples


def _sample(samples: Mapping[str, Any], name: str) -> Any:
    if name not in samples:
        raise KeyError(f"frame is missing stream {name!r}")
    return samples[name]


def _sample_name(sample: Any) -> str:
    return str(getattr(sample, "name", "stream"))


def _joint_positions(sample: Any) -> np.ndarray:
    data = sample.data
    if isinstance(data, Mapping):
        positions = np.asarray(data["position"], dtype=np.float32).reshape(-1)
        names = data.get("name")
        if names is not None:
            names = tuple(str(name) for name in np.asarray(names, dtype=object).reshape(-1).tolist())
            if names != JOINT_NAMES:
                raise ValueError(f"{_sample_name(sample)} joint order {names} != training layout {JOINT_NAMES}")
    else:
        positions = np.asarray(data, dtype=np.float32).reshape(-1)
    if positions.shape != (NUM_JOINTS,) or not np.all(np.isfinite(positions)):
        raise ValueError(f"{_sample_name(sample)} must hold {NUM_JOINTS} finite positions, got {positions}")
    return positions


def _tactile_grid(sample: Any) -> np.ndarray:
    height, width, channels = TACTILE_GRID
    points = np.asarray(sample.data, dtype=np.float32)
    if points.ndim == 1 and points.size == height * width * channels:
        points = points.reshape(height * width, channels)
    if points.shape != (height * width, channels):
        raise ValueError(
            f"{_sample_name(sample)}: expected {height * width} points x [x, y, z, dx, dy, dz], got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{_sample_name(sample)}: tactile point cloud contains NaN or inf")
    return points.reshape(height, width, channels)


def _timestamp_dict(frames: Sequence[Any]) -> dict[str, Any]:
    names = tuple(_samples(frames[-1]))
    return {
        "frame": np.asarray([int(frame.stamp_ns) for frame in frames], dtype=np.int64),
        "samples": {
            name: np.asarray([int(_samples(frame)[name].stamp_ns) for frame in frames], dtype=np.int64)
            for name in names
        },
    }


def _json_scalars(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
        for key, value in values.items()
    }


def _names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError("stream names must be a sequence, not a string")
    return tuple(str(item) for item in value)
