from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prometheus.policy.scheduler import ActionChunk

DEFAULT_STREAM_NAMES = (
    "base_0_color",
    "left_wrist_0_color",
    "right_wrist_0_color",
    "robot_state",
    "left_eef",
    "right_eef",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FLOW_MATCHING_ROOT = _PROJECT_ROOT / "third_party" / "fm"


def _ensure_flow_matching_importable() -> Path:
    root = _FLOW_MATCHING_ROOT.resolve()
    if not root.is_dir():
        raise ImportError(f"flow_matching vendor not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_run_dir(path: str | Path) -> Path:
    run_dir = Path(path).expanduser()
    if not run_dir.is_absolute():
        run_dir = (_PROJECT_ROOT / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    return run_dir


TACTILE_STREAM_SUFFIX = "_tactile_flow"


class MemoryHealthError(RuntimeError):
    """Raised when realtime Memory is stale or unavailable."""


class FMPolicy:
    """In-process Flow Matching policy adapter for Prometheus rollout."""

    def __init__(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        inference: Mapping[str, Any] | None = None,
        robot: Mapping[str, Any] | None = None,
        action_hz: float | None = None,
        runtime: Any | None = None,
    ):
        self.inputs = dict(inputs or {})
        self.inference = dict(inference or {})
        self.robot_cfg = dict(robot or {})
        self.anchor = str(self.inputs.get("anchor", "left_wrist_0_color"))
        self.slop_ms = float(self.inputs.get("slop_ms", 100.0))
        self.window_size = int(self.inputs.get("window_size", 8))
        self.stride = int(self.inputs.get("stride", 1))
        self.wait_latest = bool(self.inputs.get("wait_latest", True))
        self.wait_latest_timeout_ms = float(self.inputs.get("wait_latest_timeout_ms", 5.0))
        self.stream_names = _resolve_stream_names(self.inputs)
        self.n_action_steps = _optional_int(self.inference.get("n_action_steps"))
        self.action_hz = _optional_float(action_hz)
        if self.action_hz is None:
            self.action_hz = _optional_float(self.inference.get("action_hz"))
        if self.action_hz is None:
            self.action_hz = _optional_float(self.inputs.get("action_hz"))
        self.runtime = runtime if runtime is not None else self._build_runtime()
        self.use_memory = _optional_bool(self.inference.get("use_memory"), default=True)
        self.memory_feed_hz = (
            _optional_float(self.inference.get("memory_feed_hz"))
            or self.action_hz
            or 30.0
        )
        self.memory_warning_lag_ms = float(
            self.inference.get("memory_warning_lag_ms", 300.0)
        )
        self.memory_max_lag_ms = float(self.inference.get("memory_max_lag_ms", 500.0))
        self.memory_ready_timeout_ms = float(
            self.inference.get("memory_ready_timeout_ms", 5000.0)
        )
        self.memory_source_max_lag_ms = float(
            self.inference.get("memory_source_max_lag_ms", self.memory_max_lag_ms)
        )
        self._memory_stop_event = threading.Event()
        self._memory_thread: threading.Thread | None = None
        self._memory_preprocess_cfg = None
        self._memory_stream_names_cache: tuple[str, ...] | None = None
        self._memory_frame_id = 0
        self._memory_last_anchor_stamp_ns: int | None = None
        self._memory_last_good_frame = None
        self._memory_last_good_state: np.ndarray | None = None
        self._memory_last_fresh_time: float | None = None
        self._memory_last_error = ""
        self._memory_last_warning_time = 0.0
        self._memory_last_health: dict[str, Any] = {}
        self._validate_memory_mode()

    @property
    def use_tactile(self) -> bool:
        return bool(getattr(self.runtime, "use_tactile", False))

    def infer(self, data: Any) -> ActionChunk:
        if self._runtime_memory_enabled():
            self._ensure_memory_started(data)
        frames = data.window_numpy(
            anchor=self.anchor,
            names=self._active_stream_names(),
            count=self.window_size,
            stride=self.stride,
            slop_ms=self.slop_ms,
            wait_latest=self.wait_latest,
            timeout_ms=self.wait_latest_timeout_ms,
        )
        if self._runtime_memory_enabled():
            self._check_memory_health()

        num_inference_steps = _optional_int(self.inference.get("num_inference_steps"))
        solver = _optional_str(self.inference.get("solver"))
        if hasattr(self.runtime, "infer_from_window"):
            chunk = self.runtime.infer_from_window(
                frames,
                robot=self.robot_cfg,
                num_inference_steps=num_inference_steps,
                solver=solver,
            )
            return self._to_action_chunk(chunk)
        chunk = self._infer_from_numpy_window(
            frames,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        return self._to_action_chunk(chunk)

    def _to_action_chunk(self, chunk: Any) -> ActionChunk:
        actions = _slice_actions(chunk.actions, self.n_action_steps)
        metadata = dict(chunk.metadata)
        metadata["n_action_steps"] = int(actions.shape[0])
        if self.n_action_steps is not None:
            metadata["n_action_steps_requested"] = int(self.n_action_steps)
        if self._memory_last_health:
            metadata["memory"] = dict(self._memory_last_health)
        return ActionChunk(
            actions,
            action_space=str(chunk.action_space),
            hz=self._action_hz(default=chunk.hz),
            metadata=metadata,
        )

    def stop(self) -> None:
        self._stop_memory_feeder()

    def close(self) -> None:
        self.stop()

    def status(self) -> dict[str, Any]:
        runtime = self.runtime
        device = getattr(runtime, "device", None)
        return {
            "type": type(self).__name__,
            "anchor": self.anchor,
            "slop_ms": self.slop_ms,
            "window_size": self.window_size,
            "stride": self.stride,
            "wait_latest": self.wait_latest,
            "wait_latest_timeout_ms": self.wait_latest_timeout_ms,
            "stream_names": list(self.stream_names),
            "active_stream_names": list(self._active_stream_names()),
            "use_tactile": self.use_tactile,
            "n_action_steps": self.n_action_steps,
            "run_dir": str(getattr(runtime, "run_dir", "")),
            "checkpoint": str(getattr(runtime, "checkpoint_path", "")),
            "device": str(device) if device is not None else "",
            "action_hz": self._action_hz(default=None),
            "use_memory": self.use_memory,
            "runtime_memory_enabled": self._runtime_memory_enabled(),
            "memory_thread_alive": (
                self._memory_thread is not None and self._memory_thread.is_alive()
            ),
            "memory_health": dict(self._memory_last_health),
        }

    def _build_runtime(self) -> Any:
        _ensure_flow_matching_importable()
        from infer import FMInferenceRuntime

        if "run_dir" not in self.inference:
            raise ValueError("inference.run_dir is required for FMPolicy")
        run_dir = resolve_run_dir(self.inference["run_dir"])
        checkpoint = self.inference.get("checkpoint")
        return FMInferenceRuntime(
            run_dir,
            checkpoint=checkpoint,
            device=_optional_str(self.inference.get("device")),
            warmup=bool(self.inference.get("warmup", True)),
        )

    def _validate_memory_mode(self) -> None:
        if self.memory_feed_hz <= 0:
            raise ValueError("inference.memory_feed_hz must be positive")
        memory_enabled = self._runtime_memory_enabled()
        if memory_enabled and not self.use_memory:
            raise ValueError(
                "checkpoint enables Memory but policy.inference.use_memory=false. "
                "Set use_memory=true or use a non-Memory checkpoint."
            )
        if self.use_memory and not memory_enabled:
            print(
                "[FMPolicy][Memory] INFO: use_memory=true but checkpoint Memory is "
                "disabled; running without Memory.",
                flush=True,
            )

    def _runtime_memory_enabled(self) -> bool:
        policy = getattr(self.runtime, "policy", None)
        return bool(getattr(policy, "memory_enabled", False))

    def _memory_stream_names(self) -> tuple[str, ...]:
        if self._memory_stream_names_cache is not None:
            return self._memory_stream_names_cache
        if self._memory_preprocess_cfg is None:
            from infer.preprocess import parse_preprocess_config

            self._memory_preprocess_cfg = parse_preprocess_config(
                self.runtime.cfg,
                robot=self.robot_cfg,
            )
        names = [
            *tuple(self._memory_preprocess_cfg.camera_views),
            *tuple(self._memory_preprocess_cfg.state_stream_names),
        ]
        self._memory_stream_names_cache = tuple(dict.fromkeys(str(name) for name in names))
        return self._memory_stream_names_cache

    def _ensure_memory_started(self, data: Any) -> None:
        if not self.use_memory or not self._runtime_memory_enabled():
            return
        if self._memory_thread is not None and self._memory_thread.is_alive():
            return
        if not hasattr(self.runtime, "start_async_dino") or not hasattr(
            self.runtime,
            "submit_async_dino_frame",
        ):
            raise MemoryHealthError(
                "runtime Memory is enabled but async DINO APIs are unavailable"
            )
        from infer.preprocess import parse_preprocess_config

        self._memory_preprocess_cfg = parse_preprocess_config(
            self.runtime.cfg,
            robot=self.robot_cfg,
        )
        self._memory_stream_names_cache = None
        self.runtime.start_async_dino(preprocess=self._memory_preprocess_cfg)
        self._memory_stop_event.clear()
        self._memory_thread = threading.Thread(
            target=self._memory_feeder_loop,
            args=(data,),
            name="fm_memory_feeder",
            daemon=True,
        )
        self._memory_thread.start()
        print(
            "[FMPolicy][Memory] started async feeder: "
            f"feed_hz={self.memory_feed_hz:.1f}, "
            f"warning_lag_ms={self.memory_warning_lag_ms:.1f}, "
            f"max_lag_ms={self.memory_max_lag_ms:.1f}",
            flush=True,
        )

    def _memory_feeder_loop(self, data: Any) -> None:
        from infer.preprocess import build_state_frame

        period_s = 1.0 / float(self.memory_feed_hz)
        next_time = time.monotonic()
        while not self._memory_stop_event.is_set():
            capture_time = time.perf_counter()
            frame = None
            state = None
            try:
                frame = data.frame_numpy(
                    anchor=self.anchor,
                    names=self._memory_stream_names(),
                    slop_ms=self.slop_ms,
                )
                state = build_state_frame(frame, self._memory_preprocess_cfg)
                stamp_ns = int(getattr(frame, "stamp_ns", 0))
                if stamp_ns != self._memory_last_anchor_stamp_ns:
                    self._memory_last_anchor_stamp_ns = stamp_ns
                    self._memory_last_fresh_time = capture_time
                    self._memory_last_good_frame = frame
                    self._memory_last_good_state = np.asarray(
                        state,
                        dtype=np.float32,
                    ).copy()
                    self._memory_last_error = ""
            except Exception as exc:
                self._memory_last_error = f"{type(exc).__name__}: {exc}"
                self._maybe_print_memory_warning(
                    "waiting for synchronized memory frame; "
                    f"using previous frame if available. last_error={self._memory_last_error}",
                    interval_s=1.0,
                )

            frame_to_submit = self._memory_last_good_frame
            state_to_submit = self._memory_last_good_state
            if frame_to_submit is not None and state_to_submit is not None:
                try:
                    self.runtime.submit_async_dino_frame(
                        self._memory_frame_id,
                        frame_to_submit,
                        state_to_submit,
                        capture_time=capture_time,
                    )
                    self._memory_frame_id += 1
                except Exception as exc:
                    self._memory_last_error = f"{type(exc).__name__}: {exc}"
                    self._maybe_print_memory_warning(
                        "failed to submit async DINO memory frame: "
                        f"{self._memory_last_error}",
                        interval_s=1.0,
                    )

            next_time += period_s
            delay = next_time - time.monotonic()
            if delay <= 0:
                next_time = time.monotonic()
                delay = 0.0
            self._memory_stop_event.wait(delay)

    def _check_memory_health(self) -> None:
        deadline = time.monotonic() + max(0.0, self.memory_ready_timeout_ms) / 1000.0
        while True:
            status = self._memory_status()
            ready = bool(status.get("memory_ready")) and bool(status.get("latest_has_state"))
            if ready:
                break
            if time.monotonic() >= deadline:
                self._raise_memory_error(
                    "memory not ready before inference; stopping rollout without "
                    "sending a new action chunk",
                    status,
                )
            time.sleep(0.02)

        memory_lag_ms = _optional_float(status.get("memory_lag_ms"))
        source_lag_ms = _optional_float(status.get("source_lag_ms"))
        max_lag = float(self.memory_max_lag_ms)
        warn_lag = float(self.memory_warning_lag_ms)
        if memory_lag_ms is None:
            self._raise_memory_error("memory lag is unavailable", status)
        if memory_lag_ms > max_lag:
            self._raise_memory_error(
                f"memory lag too large: {memory_lag_ms:.1f} ms > {max_lag:.1f} ms",
                status,
            )
        if source_lag_ms is not None and source_lag_ms > self.memory_source_max_lag_ms:
            self._raise_memory_error(
                "memory source frame is stale: "
                f"{source_lag_ms:.1f} ms > {self.memory_source_max_lag_ms:.1f} ms",
                status,
            )
        if memory_lag_ms > warn_lag or (
            source_lag_ms is not None and source_lag_ms > warn_lag
        ):
            self._maybe_print_memory_warning(
                "memory lag warning: "
                f"memory_lag_ms={_format_ms(memory_lag_ms)}, "
                f"source_lag_ms={_format_ms(source_lag_ms)}",
                interval_s=1.0,
            )

    def _memory_status(self) -> dict[str, Any]:
        status_fn = getattr(self.runtime, "async_memory_status", None)
        if callable(status_fn):
            status = dict(status_fn())
        else:
            buffer = getattr(self.runtime, "async_dino_buffer", None)
            status = {"running": buffer is not None, "memory_ready": False}
            if buffer is not None:
                status.update(buffer.get_stats())
                capture_time = status.get("latest_capture_time")
                status["memory_lag_ms"] = (
                    None
                    if capture_time is None
                    else (time.perf_counter() - float(capture_time)) * 1000.0
                )
        source_lag_ms = (
            None
            if self._memory_last_fresh_time is None
            else (time.perf_counter() - self._memory_last_fresh_time) * 1000.0
        )
        status["source_lag_ms"] = source_lag_ms
        status["last_error"] = self._memory_last_error
        self._memory_last_health = self._compact_memory_status(status)
        return status

    def _compact_memory_status(self, status: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "running",
            "memory_ready",
            "memory_lag_ms",
            "source_lag_ms",
            "processed_count",
            "dropped_count",
            "deadline_miss_count",
            "buffer_length",
            "latest_frame_id",
            "latest_has_state",
            "last_error",
        )
        out: dict[str, Any] = {}
        for key in keys:
            value = status.get(key)
            if isinstance(value, float):
                value = round(value, 3)
            out[key] = value
        return out

    def _maybe_print_memory_warning(self, message: str, *, interval_s: float) -> None:
        now = time.monotonic()
        if now - self._memory_last_warning_time < float(interval_s):
            return
        self._memory_last_warning_time = now
        print(f"[FMPolicy][Memory] WARNING: {message}", flush=True)

    def _raise_memory_error(self, message: str, status: Mapping[str, Any]) -> None:
        compact = self._compact_memory_status(status)
        self._memory_last_health = compact
        print(
            "[FMPolicy][Memory] ERROR: "
            f"{message}. Stopping rollout without sending a new action chunk. "
            f"status={compact}",
            flush=True,
        )
        raise MemoryHealthError(message)

    def _stop_memory_feeder(self) -> None:
        self._memory_stop_event.set()
        if self._memory_thread is not None and self._memory_thread.is_alive():
            self._memory_thread.join(timeout=1.0)
        self._memory_thread = None
        stop_async = getattr(self.runtime, "stop_async_dino", None)
        if callable(stop_async):
            stop_async()

    def _infer_from_numpy_window(
        self,
        frames: Any,
        *,
        num_inference_steps: int | None,
        solver: str | None,
    ) -> Any:
        from infer.postprocess import apply_action_process
        from infer.preprocess import parse_preprocess_config
        from infer.types import InferenceChunk

        from prometheus.policy.numpy_preprocess import build_obs_from_numpy_frames

        preprocess_cfg = parse_preprocess_config(self.runtime.cfg, robot=self.robot_cfg)
        if len(preprocess_cfg.camera_views) != self.runtime.n_image_views:
            raise ValueError(
                "preprocess camera_views count does not match model n_image_views: "
                f"{len(preprocess_cfg.camera_views)} != {self.runtime.n_image_views}"
            )
        if bool(self.runtime.use_tactile) != bool(preprocess_cfg.use_tactile):
            raise ValueError(
                "runtime.use_tactile does not match resolved_config data.use_tactile: "
                f"{self.runtime.use_tactile} != {preprocess_cfg.use_tactile}"
            )
        obs, state_raw = build_obs_from_numpy_frames(
            frames,
            preprocess_cfg,
            self.runtime.normalizer,
            window_size=self.runtime.window_size,
        )
        pred_rot6d = self.runtime.predict_rot6d_abs(
            obs,
            state_raw=state_raw,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        actions = apply_action_process(pred_rot6d, self.runtime.deploy.action_process)
        expected = (self.runtime.action_horizon, 14)
        if actions.shape != expected:
            raise ValueError(f"postprocess shape {actions.shape} != expected {expected}")
        return InferenceChunk(
            actions=actions,
            action_space=self.runtime.deploy.action_process,
            hz=self._action_hz(default=self.runtime.deploy.action_hz),
            metadata={
                "action_horizon": self.runtime.action_horizon,
                "window_size": self.runtime.window_size,
                "timestamp": obs["timestamp"],
            },
        )

    def _action_hz(self, *, default: Any) -> float:
        hz = self.action_hz
        if hz is None:
            if default is None:
                default = getattr(getattr(self.runtime, "deploy", None), "action_hz", None)
            hz = default
        if hz is None:
            raise ValueError("policy action_hz is not configured")
        if float(hz) <= 0:
            raise ValueError("policy action_hz must be positive")
        return float(hz)


    def _active_stream_names(self) -> tuple[str, ...]:
        if self.use_tactile:
            return self.stream_names
        return tuple(
            name for name in self.stream_names if not str(name).endswith(TACTILE_STREAM_SUFFIX)
        )


def _resolve_stream_names(inputs: Mapping[str, Any]) -> tuple[str, ...]:
    names = list(inputs.get("stream_names", DEFAULT_STREAM_NAMES))
    tactile_names = inputs.get("tactile_stream_names")
    if tactile_names is not None:
        if isinstance(tactile_names, str):
            raise TypeError("tactile_stream_names must be a sequence, not a string")
        names.extend(str(item) for item in tactile_names)
    return _names(names)


def _names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError("stream names must be a sequence, not a string")
    return tuple(str(item) for item in value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _slice_actions(actions: Any, n_action_steps: int | None) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if n_action_steps is None:
        return arr
    n = int(n_action_steps)
    if n < 1:
        raise ValueError(f"n_action_steps must be positive, got {n}")
    if n > arr.shape[0]:
        raise ValueError(
            f"n_action_steps={n} exceeds predicted action horizon {arr.shape[0]}"
        )
    return arr[:n].copy()


def _format_ms(value: float | None) -> str:
    return "None" if value is None else f"{float(value):.1f} ms"
