from __future__ import annotations

from typing import Any

import numpy as np


class ZmqEnvClient:
    """Small client SDK for third-party policy code."""

    def __init__(self, socket: Any):
        self.socket = socket
        self._request_count = 0

    def reset(
        self,
        *,
        expect: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        obs_keys: tuple[str, ...] = (),
        history_count: int | None = None,
        history_anchor: str | None = None,
        history_slop_ms: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_options = dict(options or {})
        if expect is not None:
            if "expect" in reset_options:
                raise ValueError("reset() received both expect= and options['expect']")
            reset_options["expect"] = dict(expect)
        if history_count is not None:
            reset_options["history_count"] = int(history_count)
        if history_anchor is not None:
            reset_options["history_anchor"] = str(history_anchor)
        if history_slop_ms is not None:
            reset_options["history_slop_ms"] = float(history_slop_ms)
        reply = self._request({"type": "reset", "options": reset_options, "obs_keys": list(obs_keys)})
        return reply["obs"], dict(reply.get("info", {}))

    def step(
        self,
        actions: np.ndarray,
        *,
        obs_keys: tuple[str, ...] = (),
        history_count: int | None = None,
        history_anchor: str | None = None,
        history_slop_ms: float | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        payload: dict[str, Any] = {
            "type": "step",
            "actions": np.asarray(actions, dtype=np.float32),
            "obs_keys": list(obs_keys),
        }
        if history_count is not None:
            payload["history_count"] = int(history_count)
        if history_anchor is not None:
            payload["history_anchor"] = str(history_anchor)
        if history_slop_ms is not None:
            payload["history_slop_ms"] = float(history_slop_ms)
        reply = self._request(payload)
        return (
            reply["obs"],
            float(reply.get("reward", 0.0)),
            bool(reply.get("terminated", False)),
            bool(reply.get("truncated", False)),
            dict(reply.get("info", {})),
        )

    def close(self) -> None:
        try:
            self._request({"type": "close"})
        finally:
            self.socket.close(linger=0)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._request_count += 1
        request_id = f"client:{self._request_count}"
        self.socket.send_pyobj({"request_id": request_id, **payload})
        reply = self.socket.recv_pyobj()
        if not isinstance(reply, dict):
            raise TypeError(f"ZMQ env reply must be dict, got {type(reply).__name__}")
        if reply.get("request_id") != request_id:
            raise RuntimeError(f"ZMQ env reply request_id mismatch: expected {request_id}, got {reply.get('request_id')}")
        if not reply.get("ok", False):
            raise RuntimeError(str(reply.get("message", "request failed")))
        return reply


def connect(addr: str) -> ZmqEnvClient:
    import zmq

    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(str(addr))
    return ZmqEnvClient(socket)
