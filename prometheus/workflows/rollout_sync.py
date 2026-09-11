from __future__ import annotations

from prometheus.transport.local import configure as configure_transport

import os
import time
import threading
from typing import Any

import numpy as np

from prometheus.policy.fm_policy import MemoryHealthError
from prometheus.policy.scheduler import ActionChunk
from prometheus.policy.smoothing import PolicyActionSmoother
from prometheus.sessions.policy import PolicySession
from prometheus.utils.cpu_affinity import configure_process_cpu_affinity
from prometheus.utils.workflow import (
    build_data_session,
    build_hardware_session,
    config_dict,
    instantiate_config,
    prepare_run,
    section,
    write_json,
)


def run_from_config(cfg: Any) -> int:
    # Step 1: Resolve Hydra config into plain runtime sections.
    data = config_dict(cfg, "rollout_sync")
    configure_transport(data.get("transport"))
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    task_cfg = section(data, "task")
    hardware_cfg = section(data, "hardware")
    data_cfg = section(data, "data")
    policy_cfg = section(data, "policy")
    policy_session_cfg = section(data, "policy_session")
    scheduler_cfg = section(data, "scheduler")
    workflow_cfg = section(data, "workflow")
    action_smoother = PolicyActionSmoother(**workflow_cfg.get("action_smoothing", {}))
    # Apply before model loading and session creation so their workers inherit
    # the data CPU set. A5 connect separately assigns its SDK control threads.
    data_cpu_affinity = workflow_cfg.get("data_cpu_affinity")
    configure_process_cpu_affinity(data_cpu_affinity)
    if data_cpu_affinity is not None:
        print(f"[rollout_sync] data / inference CPU affinity: {sorted(os.sched_getaffinity(0))}", flush=True)
    run = prepare_run(run_cfg)
    run.runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_CAMERA_TIMING_TRACE"] = str(run.runtime_dir / "camera_publish_timing.jsonl")

    # Step 2: Build workflow-owned sessions and policy object.
    hardware = build_hardware_session(run=run, hardware_cfg=hardware_cfg, robot_cfg=robot_cfg)
    data_session = build_data_session(data_cfg=data_cfg, hardware=hardware)
    policy = instantiate_config(policy_cfg, "policy")
    policy_session = PolicySession(
        policy=policy,
        name=str(policy_session_cfg["name"]),
        mode=str(policy_session_cfg["mode"]),
    )
    robot_client = None
    scheduler = None

    status = "running"
    stop_reason = "not_started"
    wait_ready_timeout_s = float(run_cfg["wait_ready_timeout_s"])
    steps = int(run_cfg["steps"])
    if steps <= 0:
        raise ValueError("run.steps must be positive")
    stop_on_done = bool(workflow_cfg["stop_on_done"])
    chunks = 0
    actions_sent = 0
    return_code = 0
    try:
        # Step 3: Start hardware and policy resources, then wait for readiness.
        hardware.start()
        data_session.start()
        policy_session.start()
        hardware.wait_ready(wait_ready_timeout_s)
        data_session.wait_ready(wait_ready_timeout_s)
        policy_session.wait_ready(wait_ready_timeout_s)
        _wait_policy_inputs_ready(data_session, policy_session.policy, wait_ready_timeout_s)

        # Step 4: Bind the robot client and scheduler after hardware is ready.
        robot_client = hardware.robot_client
        scheduler = instantiate_config(scheduler_cfg, "scheduler", robot_client=robot_client)
        scheduler.reset()
        _open_grippers_before_policy(hardware, scheduler, workflow_cfg)
        if data_session.recording_enabled:
            recording_cfg = section(data_cfg, "recording")
            data_session.start_episode(
                str(recording_cfg["episode_id"]),
                metadata={"workflow": "rollout_sync", "run_id": run.run_id, "task": task_cfg},
            )
        
        # Step 5: Run synchronous data -> policy -> action chunks.
        while chunks < steps:
            robot_client.spin_once(0.0)
            chunk = policy_session.policy.infer(data_session) 
            if chunks == 0 and (scheduler.max_velocity is not None or action_smoother.enabled):
                current = hardware.robot_driver.read_state().qpos
                if scheduler.max_velocity is not None:
                    scheduler.seed_current_action(current)
                if action_smoother.enabled:
                    action_smoother.reset(current)
                    print(f"[rollout_sync] Policy action smoothing: {action_smoother.status()}", flush=True)
            data_session.record_policy_action_chunk(chunk, metadata={"chunk_index": chunks})
            chunk = _prepare_policy_chunk(hardware, chunk)
            chunk = action_smoother.apply(chunk)
            actions_sent += scheduler.execute(
                chunk,
                on_action=_record_action(data_session, chunks) if data_session.recording_enabled else None,
            )
            chunks += 1
            if stop_on_done and bool(chunk.metadata.get("done", False)):
                stop_reason = "policy_done"
                break
        else:
            stop_reason = "steps_completed"

        status = "completed"
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        return_code = 130
        return return_code
    except MemoryHealthError as exc:
        status = "stopped"
        stop_reason = "memory_unhealthy"
        return_code = 2
        print(f"[rollout_sync] memory unhealthy: {exc}", flush=True)
        return return_code
    except Exception:
        status = "error"
        stop_reason = "exception"
        return_code = 1
        raise
    finally:
        shutdown_gripper_error = ""
        if scheduler is not None and status in {"completed", "interrupted"}:
            try:
                # The ZMQ owner remains running until the opening command has
                # completed and both grippers have settled at their targets.
                _open_grippers_after_policy(hardware, scheduler, workflow_cfg)
            except (Exception, KeyboardInterrupt) as exc:
                shutdown_gripper_error = repr(exc)
                status = "error"
                stop_reason = "shutdown_gripper_open_failed"
                print(f"[rollout_sync] Gripper opening failed: {exc!r}; skipping return motion.", flush=True)
        shutdown_pose = _return_to_shutdown_pose(hardware, workflow_cfg, scheduler, status)
        if shutdown_pose.get("status") == "error":
            status = "error"
            stop_reason = "shutdown_return_failed"
        if scheduler is not None:
            actions_sent = scheduler.actions_processed
        # Step 6: Freeze data ingress, then let robot cleanup/reset and data finalization overlap.
        try:
            data_session.stop_ingress()
        except Exception:
            pass
        if data_session.recording_enabled:
            data_session.stop_episode(accept=status == "completed")
        try:
            policy_session.stop()
        finally:
            policy_session.close()
        _stop_data_and_hardware(data_session, hardware)
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "shutdown_pose": shutdown_pose,
                "shutdown_gripper_error": shutdown_gripper_error,
                "action_smoothing": action_smoother.status(),
                "chunks": chunks,
                "actions_sent": actions_sent,
                "time_ns": time.time_ns(),
                "output_dir": str(run.output_dir),
                "runtime_dir": str(run.runtime_dir),
                "task": task_cfg,
                "hardware": hardware.status().as_dict(),
                "data": data_session.status().as_dict(),
                "policy": policy_session.status().as_dict(),
                "robot_client": None if robot_client is None else robot_client.status(),
                "scheduler": {
                    "dry_run": None if scheduler is None else scheduler.dry_run,
                    "max_acceleration": (
                        None if scheduler is None or scheduler.max_acceleration is None
                        else [float(value) if np.isfinite(value) else None for value in scheduler.max_acceleration]
                    ),
                    "history": 0 if scheduler is None else len(scheduler.history),
                    "filters": [] if scheduler is None else [item.name for item in scheduler.filters],
                },
            },
        )
        if shutdown_pose.get("status") == "error":
            raise RuntimeError("shutdown return failed: " + shutdown_pose["error"])
        if shutdown_gripper_error:
            raise RuntimeError("shutdown gripper opening failed: " + shutdown_gripper_error)


def _prepare_policy_chunk(hardware, chunk):
    if not hardware.robot.get("driver"):
        return chunk
    prepare = getattr(hardware.robot_driver, "prepare_policy_actions", None)
    if not callable(prepare):
        return chunk
    return ActionChunk(
        prepare(chunk.actions), action_space=chunk.action_space, hz=chunk.hz,
        metadata=dict(chunk.metadata),
    )


def _open_grippers_before_policy(hardware, scheduler, workflow_cfg):
    _open_grippers(hardware, scheduler, workflow_cfg.get("open_grippers_before_policy", {}), "before inference")


def _open_grippers_after_policy(hardware, scheduler, workflow_cfg):
    _open_grippers(hardware, scheduler, workflow_cfg.get("open_grippers_after_policy", {}), "after inference")


def _open_grippers(hardware, scheduler, opening, phase):
    if not opening.get("enabled", False):
        return
    if scheduler.dry_run:
        print("[rollout_sync] dry_run: skipping physical gripper opening.", flush=True)
        return
    targets = np.asarray(opening["positions"], dtype=np.float32)
    tolerance = float(opening.get("tolerance", 0.1))
    timeout = float(opening.get("timeout_s", 5.0))
    settle = float(opening.get("settle_s", 0.5))
    if targets.shape != (2,) or not np.all(np.isfinite(targets)) or np.any(targets < -3.14) or np.any(targets > 0):
        raise ValueError("A5 gripper opening requires two SDK positions in [-3.14, 0]")
    if not all(np.isfinite(v) and v > 0 for v in (tolerance, timeout, settle)):
        raise ValueError("gripper opening tolerance, timeout and settle must be finite and positive")
    if scheduler.max_velocity is None:
        raise ValueError("gripper opening requires velocity interpolation")
    driver = hardware.robot_driver
    current = np.asarray(driver.read_state().qpos, dtype=np.float32)
    if current.shape != (14,) or not np.all(np.isfinite(current)):
        raise ValueError("gripper opening requires finite 14-axis A5 feedback")
    goal = current.copy()
    goal[[6, 13]] = targets
    scheduler.seed_current_action(current)
    print(f"[rollout_sync] Opening grippers to {targets.tolist()} {phase}...", flush=True)
    scheduler.execute(ActionChunk([goal], action_space="abs_qpos", hz=scheduler.interpolation_hz))
    deadline = time.monotonic() + timeout
    settled_since = None
    while True:
        actual = np.asarray(driver.read_state().qpos, dtype=np.float32)[[6, 13]]
        now = time.monotonic()
        if np.all(np.isfinite(actual)) and np.all(np.abs(actual - targets) <= tolerance):
            if settled_since is None:
                settled_since = now
            if now - settled_since >= settle:
                break
        else:
            settled_since = None
        if now >= deadline:
            raise TimeoutError(f"grippers did not reach open position: actual={actual.tolist()}, target={targets.tolist()}, tolerance={tolerance}")
        scheduler._spin_sleep(min(0.02, deadline - now))
    print(f"[rollout_sync] Grippers within opening tolerance ({phase}): actual={actual.tolist()}, target={targets.tolist()}, tolerance={tolerance}.", flush=True)


def _return_to_shutdown_pose(hardware, workflow_cfg, scheduler, status):
    if not workflow_cfg.get("return_to_shutdown_pose", False):
        return {"status": "disabled"}
    if scheduler is None or scheduler.dry_run or status not in {"completed", "interrupted"}:
        return {"status": "skipped"}
    try:
        driver = hardware.robot_driver
        if not driver.is_connected:
            return {"status": "skipped", "reason": "driver disconnected"}
        # Gripper opening has finished through ZMQ; now stop command processing
        # before directly using the owned driver for the return motion.
        hardware.robot_owner.stop()
        print("[rollout_sync] Returning slowly to shutdown_safe_position...", flush=True)
        driver.move_to_shutdown_safe_position()
        print("[rollout_sync] Shutdown pose reached; closing hardware.", flush=True)
        return {"status": "completed"}
    except (Exception, KeyboardInterrupt) as exc:
        print(f"[rollout_sync] Shutdown return failed: {exc!r}; closing hardware.", flush=True)
        return {"status": "error", "error": repr(exc)}


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="rollout_sync")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


def _record_action(data_session: Any, chunk_index: int):
    def _callback(action, context) -> None:
        ok = data_session.record_action(
            action,
            action_space=context.action_space,
            hz=context.chunk.hz,
            metadata={"chunk_index": chunk_index, "action_index": context.index},
        )
        if not ok:
            raise RuntimeError("failed to enqueue executed action for recording")

    return _callback


def _stop_data_and_hardware(data_session: Any, hardware: Any) -> None:
    errors: list[str] = []

    def _run(label: str, fn: Any) -> None:
        try:
            fn()
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    data_thread = threading.Thread(
        target=_run,
        args=("data.stop", data_session.stop),
        name="rollout_data_stop",
        daemon=False,
    )
    hardware_thread = threading.Thread(
        target=_run,
        args=("hardware.stop", hardware.stop),
        name="rollout_hardware_stop",
        daemon=False,
    )
    data_thread.start()
    hardware_thread.start()
    data_thread.join()
    hardware_thread.join()
    _run("data.close", data_session.close)
    _run("hardware.close", hardware.close)
    if errors:
        print("cleanup errors: " + "; ".join(errors), flush=True)


def _wait_policy_inputs_ready(data_session: Any, policy: Any, timeout_s: float) -> None:
    anchor = getattr(policy, "anchor", None)
    stream_names = getattr(policy, "stream_names", None)
    window_size = int(getattr(policy, "window_size", 1))
    stride = int(getattr(policy, "stride", 1))
    if not anchor or not stream_names:
        return
    needed = (window_size - 1) * stride + 1
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    while time.monotonic() < deadline:
        if data_session.ready([anchor], count=needed) and data_session.ready(stream_names, count=1):
            try:
                frames = data_session.window_numpy(
                    anchor=anchor,
                    names=stream_names,
                    count=window_size,
                    stride=stride,
                    slop_ms=float(getattr(policy, "slop_ms", 100.0)),
                )
                _validate_policy_window(policy, frames)
                return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(0.05)
    skew_hint = ""
    try:
        skew = data_session.latest_skew_ms(anchor=anchor, names=stream_names)
        skew_hint = f" latest_header_skew_ms={skew}"
    except Exception:
        pass
    raise TimeoutError(
        "timed out waiting for policy input window: "
        f"anchor={anchor!r} needed={needed} counts={data_session.counts()} "
        f"numpy_counts={data_session.numpy_counts()} "
        f"last_error={last_error!r}{skew_hint}"
    )


def _validate_policy_window(policy: Any, frames: list[Any]) -> None:
    runtime = getattr(policy, "runtime", None)
    if runtime is None or not runtime.__class__.__module__.startswith("infer."):
        return
    from infer.preprocess import parse_preprocess_config
    from prometheus.policy.numpy_preprocess import build_obs_from_numpy_frames

    preprocess_cfg = parse_preprocess_config(
        runtime.cfg,
        robot=getattr(policy, "robot_cfg", {}),
    )
    build_obs_from_numpy_frames(
        frames,
        preprocess_cfg,
        runtime.normalizer,
        window_size=int(getattr(runtime, "window_size")),
    )


if __name__ == "__main__":
    main()
