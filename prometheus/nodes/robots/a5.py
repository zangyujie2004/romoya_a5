from __future__ import annotations

import gc
import math
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from prometheus.nodes.robots.robot_node import RobotState
from prometheus.utils.cpu_affinity import inherit_cpu_affinity, normalize_cpu_affinity


A5_ACTION_SPACES = ("abs_qpos",)
ACTION_DIMS = {"abs_qpos": 14}
ROBOT_TOPIC_SUFFIXES = {
    "state": "state",
    "left_eef": "left_eef_pose",
    "right_eef": "right_eef_pose",
    "control": "control",
    "result": "control_result",
    "status": "status",
}
SIDE_VECTOR_START = {"left": 0, "right": 7}
MAX_ABS_TEACH_GRIPPER_TORQUE = 0.1
A5_SAFE_JOINT_POSITION_LIMIT = math.pi
# Positive offsets close the grippers slightly in native SDK position units.
# Applied to policy targets before interpolation, not feedback or lifecycle moves.
GIPPER_OFFSET = 0.1

@dataclass(frozen=True, kw_only=True)
class A5RobotState(RobotState):
    eef_wxyz: np.ndarray


class A5DragDriver:
    """A5 bimanual driver for hand-guided collection and joint replay.

    Each side uses six joint angles followed by the native SDK gripper position,
    matching read_state().qpos. Gripper positions are not widths in metres.
    """

    sides = ("left", "right")
    action_dim = 14

    def __init__(
        self,
        *,
        robot_id: str = "a5_drag",
        left_can: str = "can1",
        right_can: str = "can3",
        hz: float = 100.0,
        action_space: str = "abs_qpos",
        urdf_name: str = "a5.urdf",
        link6_mass_kg: Mapping[str, float] | None = None,
        teach_gripper_torque: Mapping[str, float] | None = None,
        shutdown_safe_position: Mapping[str, Any] | None = None,
        shutdown_safe_max_velocity: float = 0.25,
        shutdown_safe_timeout_s: float = 30.0,
        shutdown_safe_settle_time_s: float = 0.3,
        shutdown_safe_position_tolerance: float = 0.01,
        shutdown_safe_velocity_tolerance: float = 0.05,
        shutdown_safe_poll_interval_s: float = 0.02,
        dt: float = 0.01,
        feedback_timeout_s: float = 2.0,
        home_timeout_s: float = 25.0,
        home_command_grace_s: float = 0.5,
        home_settle_time_s: float = 0.5,
        home_velocity_tolerance: float = 0.05,
        home_position_tolerance: float = 0.005,
        home_poll_interval_s: float = 0.05,
        sdk_root: str | None = None,
        eef_topics: Mapping[str, str] | None = None,
        control_cpu_affinity: list[int] | None = None,
    ):
        if action_space not in A5_ACTION_SPACES:
            raise ValueError(f"A5 action_space must be one of {A5_ACTION_SPACES}")
        if float(hz) <= 0:
            raise ValueError("A5 hz must be positive")
        if float(dt) <= 0:
            raise ValueError("A5 dt must be positive")
        if float(feedback_timeout_s) <= 0:
            raise ValueError("A5 feedback_timeout_s must be positive")
        shutdown_safe_values = {
            "shutdown_safe_max_velocity": shutdown_safe_max_velocity,
            "shutdown_safe_timeout_s": shutdown_safe_timeout_s,
            "shutdown_safe_settle_time_s": shutdown_safe_settle_time_s,
            "shutdown_safe_position_tolerance": shutdown_safe_position_tolerance,
            "shutdown_safe_velocity_tolerance": shutdown_safe_velocity_tolerance,
            "shutdown_safe_poll_interval_s": shutdown_safe_poll_interval_s,
        }
        for name, value in shutdown_safe_values.items():
            if float(value) <= 0:
                raise ValueError(f"A5 {name} must be positive")
        if float(shutdown_safe_settle_time_s) >= float(shutdown_safe_timeout_s):
            raise ValueError("A5 shutdown_safe_timeout_s must exceed shutdown_safe_settle_time_s")
        home_values = {
            "home_timeout_s": home_timeout_s,
            "home_command_grace_s": home_command_grace_s,
            "home_settle_time_s": home_settle_time_s,
            "home_velocity_tolerance": home_velocity_tolerance,
            "home_position_tolerance": home_position_tolerance,
            "home_poll_interval_s": home_poll_interval_s,
        }
        for name, value in home_values.items():
            if float(value) <= 0:
                raise ValueError(f"A5 {name} must be positive")
        if float(home_command_grace_s) + float(home_settle_time_s) >= float(home_timeout_s):
            raise ValueError("A5 home_timeout_s must exceed home_command_grace_s + home_settle_time_s")
        self.robot_id = str(robot_id)
        self.left_can = str(left_can)
        self.right_can = str(right_can)
        self.hz = float(hz)
        self.action_space = str(action_space)
        self.urdf_name = str(urdf_name)
        self.link6_mass_kg = _optional_side_floats(
            link6_mass_kg,
            "link6_mass_kg",
            positive=True,
        )
        self.teach_gripper_torque = _optional_side_floats(
            teach_gripper_torque,
            "teach_gripper_torque",
            default=0.0,
            max_abs=MAX_ABS_TEACH_GRIPPER_TORQUE,
        )
        self.shutdown_safe_position = _optional_side_vectors(
            shutdown_safe_position,
            "shutdown_safe_position",
            size=6,
            max_abs=A5_SAFE_JOINT_POSITION_LIMIT,
        )
        self.shutdown_safe_max_velocity = float(shutdown_safe_max_velocity)
        self.shutdown_safe_timeout_s = float(shutdown_safe_timeout_s)
        self.shutdown_safe_settle_time_s = float(shutdown_safe_settle_time_s)
        self.shutdown_safe_position_tolerance = float(shutdown_safe_position_tolerance)
        self.shutdown_safe_velocity_tolerance = float(shutdown_safe_velocity_tolerance)
        self.shutdown_safe_poll_interval_s = float(shutdown_safe_poll_interval_s)
        self.dt = float(dt)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.home_timeout_s = float(home_timeout_s)
        self.home_command_grace_s = float(home_command_grace_s)
        self.home_settle_time_s = float(home_settle_time_s)
        self.home_velocity_tolerance = float(home_velocity_tolerance)
        self.home_position_tolerance = float(home_position_tolerance)
        self.home_poll_interval_s = float(home_poll_interval_s)
        self.sdk_root = Path(sdk_root).expanduser() if sdk_root else _default_sdk_root()
        self.eef_topics = dict(eef_topics or {})
        self.control_cpu_affinity = normalize_cpu_affinity(control_cpu_affinity)
        self.PoseStamped: Any | None = None
        self.eef_pubs: dict[str, Any] = {}
        self._arms: dict[str, Any] = {}
        self._generated_urdf_dir: Any | None = None
        self._shutdown_requested = threading.Event()
        self._closed = True
        self.control_mode = ""

    @property
    def joint_names(self) -> list[str]:
        names: list[str] = []
        for side in self.sides:
            names.extend([f"{side}_joint_{index}" for index in range(1, 7)])
            names.append(f"{side}_gripper")
        return names

    @property
    def is_connected(self) -> bool:
        return len(self._arms) == 2 and not self._closed

    def connect(self) -> None:
        if self.is_connected:
            return
        single_arm = self._import_single_arm()
        self._shutdown_requested.clear()
        self._closed = False
        try:
            urdf_names = self._prepare_urdfs()
            configs = {
                "left": self._arm_config(self.left_can, urdf_names["left"]),
                "right": self._arm_config(self.right_can, urdf_names["right"]),
            }
            with inherit_cpu_affinity(self.control_cpu_affinity):
                for side in self.sides:
                    self._arms[side] = single_arm(configs[side])
            self._wait_for_feedback()
            self.reset_home()
        except Exception:
            self.close()
            raise

    def read_state(self) -> A5RobotState:
        self._require_connected()

        def read_side(side: str) -> dict[str, np.ndarray]:
            arm = self._arms[side]
            return {
                "qpos": _state_vector(arm.get_joint_positions(), f"{side} qpos"),
                "qvel": _state_vector(arm.get_joint_velocities(), f"{side} qvel"),
                "effort": _state_vector(arm.get_joint_torques(), f"{side} effort"),
                "eef_wxyz": _state_vector(arm.get_ee_pose(), f"{side} eef pose"),
            }

        side_states = self._run_sides(read_side)
        return A5RobotState(
            joint_names=self.joint_names,
            qpos=_concat_sides(side_states, "qpos"),
            qvel=_concat_sides(side_states, "qvel"),
            effort=_concat_sides(side_states, "effort"),
            eef_wxyz=_concat_sides(side_states, "eef_wxyz"),
        )

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        self._require_connected()
        if action_space not in A5_ACTION_SPACES:
            raise ValueError(f"A5 action_space must be one of {A5_ACTION_SPACES}, got {action_space!r}")
        self.send_joint_positions(action)

    def prepare_policy_actions(self, actions: np.ndarray) -> np.ndarray:
        """Offset only policy gripper targets; preserve raw predictions and joints."""
        command = np.array(actions, dtype=np.float32, copy=True)
        if command.ndim != 2 or command.shape[1] != 14 or not np.all(np.isfinite(command)):
            raise ValueError("A5 policy actions must have finite shape (T, 14)")
        if not np.isfinite(GIPPER_OFFSET):
            raise ValueError("GIPPER_OFFSET must be finite")
        command[:, [6, 13]] = np.clip(command[:, [6, 13]] + GIPPER_OFFSET, -3.14, 0.0)
        return command

    def send_joint_positions(self, qpos: np.ndarray) -> np.ndarray:
        self._require_motion_available()
        command = _joint_command(qpos)
        if self.control_mode != "position":
            self.set_position_mode()
        try:
            self._run_sides(lambda side: self._send_side_joint_positions(side, _side_vector(command, side)))
        except Exception:
            self.set_protect_mode()
            raise
        return command

    def set_teach_mode(self) -> None:
        self._require_connected()
        self._run_sides(self._set_teach_side)
        self.control_mode = "teach"

    def set_position_mode(self) -> None:
        self._require_motion_available()
        if self.control_mode == "position":
            return
        try:
            # Validate both sides before the SDK switches either arm to position
            # control. Seed targets from feedback so mode changes hold the pose.
            current = _joint_command(
                np.concatenate(
                    [
                        _state_vector(self._arms[side].get_joint_positions(), f"{side} position-mode qpos")
                        for side in self.sides
                    ]
                )
            )
            self._clear_gripper_torque()
            self._run_sides(lambda side: self._send_side_joint_positions(side, _side_vector(current, side)))
            self.control_mode = "position"
        except Exception:
            self.set_protect_mode()
            raise

    def set_protect_mode(self) -> None:
        self._require_connected()
        errors = self._protect_existing_arms()
        self.control_mode = "protect"
        if errors:
            raise RuntimeError("A5 protect_mode failed: " + "; ".join(errors))

    def reset_home(self) -> None:
        self._require_connected()
        self.control_mode = "homing"
        try:
            self._clear_gripper_torque()
            results = self._run_sides(lambda side: self._arms[side].go_home())
            rejected = [side for side, result in results.items() if result is False]
            if rejected:
                raise RuntimeError(f"A5 go_home was rejected for sides: {rejected}")
            # The vendor home state accepts the catch target independently of
            # the six arm joints. Zero is the A5 closed-gripper position.
            self._run_sides(lambda side: self._arms[side].set_gripper_pos(0.0))
            self._wait_for_home_settle()
            if self._shutdown_requested.is_set():
                raise RuntimeError("A5 home reset was cancelled by shutdown")
            self.set_teach_mode()
        except Exception:
            self._protect_after_reset_failure()
            raise

    def move_to_shutdown_safe_position(self) -> None:
        self._require_connected()
        if self.shutdown_safe_position is None:
            raise RuntimeError("A5 shutdown_safe_position is not configured")

        try:
            self._clear_gripper_torque()
            self.control_mode = "shutdown_positioning"
            start = {
                side: _state_vector(
                    self._arms[side].get_joint_positions(),
                    f"{side} shutdown-start qpos",
                )[:6].astype(np.float64)
                for side in self.sides
            }
            if any(not np.all(np.isfinite(values)) for values in start.values()):
                raise RuntimeError("A5 shutdown-start feedback contains NaN or inf values")

            max_distance = max(
                float(np.max(np.abs(self.shutdown_safe_position[side] - start[side])))
                for side in self.sides
            )
            # A half-cosine profile starts and stops with zero velocity. Its
            # peak slope is pi/2, so scale duration to honor the configured cap.
            trajectory_duration = (
                max_distance * (math.pi / 2.0) / self.shutdown_safe_max_velocity
            )
            required_time = trajectory_duration + self.shutdown_safe_settle_time_s
            if required_time >= self.shutdown_safe_timeout_s:
                raise TimeoutError(
                    "A5 shutdown-safe trajectory needs "
                    f"{required_time:.1f}s but timeout is {self.shutdown_safe_timeout_s:.1f}s"
                )

            deadline = time.monotonic() + self.shutdown_safe_timeout_s
            command_interval_s = self.dt
            steps = max(1, int(math.ceil(trajectory_duration / command_interval_s)))
            started = time.monotonic()
            for index in range(steps + 1):
                if self._shutdown_requested.is_set():
                    raise RuntimeError("A5 shutdown-safe move was cancelled by shutdown")
                phase = index / steps
                alpha = 0.5 - 0.5 * math.cos(math.pi * phase)
                for side in self.sides:
                    command = start[side] + alpha * (
                        self.shutdown_safe_position[side] - start[side]
                    )
                    self._arms[side].set_joint_positions(command)
                next_tick = started + (index + 1) * self.dt
                if index < steps:
                    time.sleep(max(0.0, next_tick - time.monotonic()))

            self._wait_for_shutdown_safe_settle(deadline)
            self.control_mode = "shutdown_safe"
        except Exception:
            self._protect_after_reset_failure()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._shutdown_requested.set()
        protect_errors: list[str] = []
        try:
            if self._arms:
                protect_errors = self._protect_existing_arms()
        finally:
            self.control_mode = "protect"
            self._arms.clear()
            gc.collect()
            self._cleanup_generated_urdfs()
            self._closed = True
        if protect_errors:
            raise RuntimeError("A5 protect_mode failed: " + "; ".join(protect_errors))

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "connected": self.is_connected,
            "action_space": self.action_space,
            "control_mode": self.control_mode,
            "hz": self.hz,
            "feedback_timeout_s": self.feedback_timeout_s,
            "home_timeout_s": self.home_timeout_s,
            "home_command_grace_s": self.home_command_grace_s,
            "home_settle_time_s": self.home_settle_time_s,
            "home_velocity_tolerance": self.home_velocity_tolerance,
            "home_position_tolerance": self.home_position_tolerance,
            "left_can": self.left_can,
            "right_can": self.right_can,
            "sdk_root": str(self.sdk_root),
            "urdf_name": self.urdf_name,
            "link6_mass_kg": dict(self.link6_mass_kg) if self.link6_mass_kg is not None else None,
            "teach_gripper_torque": dict(self.teach_gripper_torque),
            "shutdown_safe_position": (
                {side: values.tolist() for side, values in self.shutdown_safe_position.items()}
                if self.shutdown_safe_position is not None
                else None
            ),
            "shutdown_safe_max_velocity": self.shutdown_safe_max_velocity,
            "shutdown_safe_timeout_s": self.shutdown_safe_timeout_s,
            "eef_topics": dict(self.eef_topics),
            "motion_commands_enabled": True,
            "control_cpu_affinity": self.control_cpu_affinity,
            "reset_home_enabled": True,
        }

    def setup_transport(self, *, node: Any, qos_depth: int) -> None:
        if not self.eef_topics:
            return
        from prometheus.transport.messages import PoseStamped

        missing = set(self.sides).difference(self.eef_topics)
        if missing:
            raise ValueError(f"A5 eef_topics missing sides: {sorted(missing)}")
        self.PoseStamped = PoseStamped
        self.eef_pubs = {
            side: node.create_publisher(PoseStamped, self.eef_topics[side], int(qos_depth))
            for side in self.sides
        }

    def publish_transport_state(self, *, node: Any, stamp: Any, state: RobotState) -> None:
        del node
        if not self.eef_pubs:
            return
        values = np.asarray(getattr(state, "eef_wxyz"), dtype=np.float32).reshape(14)
        for side in self.sides:
            if self.PoseStamped is None:
                raise RuntimeError("A5 EEF publishers are not initialized")
            pose = _side_vector(values, side)
            if not np.all(np.isfinite(pose)):
                continue
            msg = self.PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = f"{self.robot_id}/{side}_eef"
            msg.pose.position.x = float(pose[0])
            msg.pose.position.y = float(pose[1])
            msg.pose.position.z = float(pose[2])
            msg.pose.orientation.w = float(pose[3])
            msg.pose.orientation.x = float(pose[4])
            msg.pose.orientation.y = float(pose[5])
            msg.pose.orientation.z = float(pose[6])
            self.eef_pubs[side].publish(msg)

    def _arm_config(self, can_port: str, urdf_name: str) -> dict[str, Any]:
        return {
            "can_port": can_port,
            "urdf_name": urdf_name,
            "num_joints": 7,
            "dt": self.dt,
        }

    def _prepare_urdfs(self) -> dict[str, str]:
        if self.link6_mass_kg is None:
            return {side: self.urdf_name for side in self.sides}

        source = Path(self.urdf_name).expanduser()
        if not source.is_absolute():
            source = self.sdk_root / "bimanual" / "script" / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"A5 URDF not found: {source}")

        self._cleanup_generated_urdfs()
        self._generated_urdf_dir = tempfile.TemporaryDirectory(prefix="prometheus-a5-urdf-")
        output_dir = Path(self._generated_urdf_dir.name)
        paths: dict[str, str] = {}
        for side in self.sides:
            tree = ET.parse(source)
            mass = _link6_mass_element(tree, source)
            mass.set("value", format(self.link6_mass_kg[side], ".15g"))
            output = output_dir / f"a5_{side}.urdf"
            tree.write(output, encoding="utf-8", xml_declaration=True)
            paths[side] = str(output)
        return paths

    def _cleanup_generated_urdfs(self) -> None:
        if self._generated_urdf_dir is not None:
            self._generated_urdf_dir.cleanup()
            self._generated_urdf_dir = None

    def _set_teach_side(self, side: str) -> None:
        arm = self._arms[side]
        arm.gravity_compensation()
        arm.set_gripper_ext_torque(self.teach_gripper_torque[side])

    def _clear_gripper_torque(self) -> None:
        self._run_sides(lambda side: self._arms[side].set_gripper_ext_torque(0.0))

    def _import_single_arm(self) -> Any:
        sdk_root = self.sdk_root.resolve()
        package_dir = sdk_root / "bimanual"
        api_dir = package_dir / "api"
        extension_dir = api_dir / "arx_r5_python"
        required = (
            package_dir / "__init__.py",
            api_dir / "arx_r5_src" / "libarx_r5a_src.so",
            api_dir / "libkinematic_solver.so",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "A5 SDK is not built; missing " + ", ".join(missing) + ". Run third_party/A5/build.sh first."
            )
        for path in (sdk_root, extension_dir, api_dir):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        try:
            from bimanual import SingleArm
        except (ImportError, OSError) as exc:
            raise ImportError(
                f"failed to import A5 SingleArm from {sdk_root}: {exc}. "
                "Install the A5 shared-library links into the active environment lib directory."
            ) from exc
        return SingleArm

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("A5 controller is not connected")

    def _require_motion_available(self) -> None:
        self._require_connected()
        if self._shutdown_requested.is_set():
            raise RuntimeError("A5 motion was cancelled by shutdown")

    def _send_side_joint_positions(self, side: str, target: np.ndarray) -> None:
        self._require_motion_available()
        arm = self._arms[side]
        # set_joint_positions() enables SDK state 5. Set the catch target first
        # so enabling position control cannot reuse an old gripper target.
        if arm.set_gripper_pos(float(target[6])) is False:
            raise RuntimeError(f"A5 {side} gripper position command was rejected")
        if arm.set_joint_positions(target[:6].copy()) is False:
            raise RuntimeError(f"A5 {side} joint position command was rejected")

    def _wait_for_feedback(self) -> None:
        deadline = time.monotonic() + self.feedback_timeout_s
        missing = list(self.sides)
        while time.monotonic() < deadline:
            missing = [side for side in self.sides if not _has_feedback(self._arms[side])]
            if not missing:
                return
            time.sleep(0.05)
        interfaces = [self.left_can if side == "left" else self.right_can for side in missing]
        raise RuntimeError(
            "A5 received no motor feedback on "
            + ", ".join(interfaces)
            + "; check 24V power, emergency stop, and CAN signal cables"
        )

    def _wait_for_home_settle(self) -> None:
        started = time.monotonic()
        deadline = started + self.home_timeout_s
        evaluate_after = started + self.home_command_grace_s
        settled_since: float | None = None
        previous_positions: np.ndarray | None = None

        while time.monotonic() < deadline:
            if self._shutdown_requested.is_set():
                raise RuntimeError("A5 home reset was cancelled by shutdown")
            positions = np.concatenate(
                [
                    _state_vector(self._arms[side].get_joint_positions(), f"{side} home qpos")
                    for side in self.sides
                ]
            )
            velocities = np.concatenate(
                [
                    _state_vector(self._arms[side].get_joint_velocities(), f"{side} home qvel")
                    for side in self.sides
                ]
            )
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
                raise RuntimeError("A5 home feedback contains NaN or inf values")

            now = time.monotonic()
            feedback_ok = all(_has_feedback(self._arms[side]) for side in self.sides)
            velocity_settled = bool(np.max(np.abs(velocities)) <= self.home_velocity_tolerance)
            position_settled = bool(
                previous_positions is not None
                and np.max(np.abs(positions - previous_positions)) <= self.home_position_tolerance
            )
            if now >= evaluate_after and feedback_ok and velocity_settled and position_settled:
                if settled_since is None:
                    settled_since = now
                elif now - settled_since >= self.home_settle_time_s:
                    return
            else:
                settled_since = None

            previous_positions = positions
            time.sleep(min(self.home_poll_interval_s, max(0.0, deadline - time.monotonic())))

        raise TimeoutError(
            f"A5 did not settle at home within {self.home_timeout_s:.1f}s; entering protect mode"
        )

    def _wait_for_shutdown_safe_settle(self, deadline: float) -> None:
        if self.shutdown_safe_position is None:
            raise RuntimeError("A5 shutdown_safe_position is not configured")
        settled_since: float | None = None
        last_error = float("inf")
        last_velocity = float("inf")
        while time.monotonic() < deadline:
            if self._shutdown_requested.is_set():
                raise RuntimeError("A5 shutdown-safe move was cancelled by shutdown")
            positions = {
                side: _state_vector(
                    self._arms[side].get_joint_positions(),
                    f"{side} shutdown-safe qpos",
                )[:6]
                for side in self.sides
            }
            velocities = {
                side: _state_vector(
                    self._arms[side].get_joint_velocities(),
                    f"{side} shutdown-safe qvel",
                )[:6]
                for side in self.sides
            }
            if any(
                not np.all(np.isfinite(values))
                for values in (*positions.values(), *velocities.values())
            ):
                raise RuntimeError("A5 shutdown-safe feedback contains NaN or inf values")
            last_error = max(
                float(np.max(np.abs(positions[side] - self.shutdown_safe_position[side])))
                for side in self.sides
            )
            last_velocity = max(
                float(np.max(np.abs(velocities[side]))) for side in self.sides
            )
            now = time.monotonic()
            if (
                last_error <= self.shutdown_safe_position_tolerance
                and last_velocity <= self.shutdown_safe_velocity_tolerance
            ):
                if settled_since is None:
                    settled_since = now
                elif now - settled_since >= self.shutdown_safe_settle_time_s:
                    return
            else:
                settled_since = None
            time.sleep(
                min(
                    self.shutdown_safe_poll_interval_s,
                    max(0.0, deadline - time.monotonic()),
                )
            )
        raise TimeoutError(
            "A5 did not settle at the shutdown-safe position within "
            f"{self.shutdown_safe_timeout_s:.1f}s "
            f"(max position error={last_error:.4f}, max velocity={last_velocity:.4f}); "
            "entering protect mode"
        )

    def _protect_after_reset_failure(self) -> None:
        self._protect_existing_arms()
        self.control_mode = "protect"

    def _protect_existing_arms(self) -> list[str]:
        errors: list[str] = []
        for side, arm in list(self._arms.items()):
            try:
                arm.set_gripper_ext_torque(0.0)
            except Exception as exc:
                errors.append(f"{side} gripper torque clear: {exc}")
            try:
                arm.protect_mode()
            except Exception as exc:
                errors.append(f"{side}: {exc}")
        return errors

    def _run_sides(self, fn: Callable[[str], Any]) -> dict[str, Any]:
        return {side: fn(side) for side in self.sides}

def a5_topics(robot_id: str) -> dict[str, str]:
    prefix = f"/prometheus/robots/{robot_id}"
    return {name: f"{prefix}/{suffix}" for name, suffix in ROBOT_TOPIC_SUFFIXES.items()}


def resolve_robot(**robot: Any) -> dict[str, Any]:
    config = dict(robot)
    config["type"] = "a5"
    config.update(runtime_config(config))
    config["sensors"] = sensor_config(config)
    config["streams"] = streams(config)
    return config


def runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    robot_id = str(robot["id"])
    robot_cfg = _robot_config(robot)
    can = _side_mapping(robot_cfg, "can")
    topics = a5_topics(robot_id)
    action_space = str(robot_cfg["action_space"])
    state_publish_hz = float(robot_cfg["state_publish_hz"])
    control_domain_id = int(robot["control_domain_id"])
    return {
        "driver": {
            "_target_": "prometheus.nodes.robots.a5.A5DragDriver",
            "robot_id": robot_id,
            "left_can": can["left"],
            "right_can": can["right"],
            "hz": state_publish_hz,
            "action_space": action_space,
            "urdf_name": str(robot_cfg.get("urdf_name", "a5.urdf")),
            "link6_mass_kg": _optional_side_config(robot_cfg, "link6_mass_kg"),
            "teach_gripper_torque": _optional_side_config(robot_cfg, "teach_gripper_torque"),
            "shutdown_safe_position": _optional_side_config(
                robot_cfg, "shutdown_safe_position"
            ),
            "shutdown_safe_max_velocity": float(
                robot_cfg.get("shutdown_safe_max_velocity", 0.25)
            ),
            "shutdown_safe_timeout_s": float(robot_cfg.get("shutdown_safe_timeout_s", 30.0)),
            "shutdown_safe_settle_time_s": float(
                robot_cfg.get("shutdown_safe_settle_time_s", 0.3)
            ),
            "shutdown_safe_position_tolerance": float(
                robot_cfg.get("shutdown_safe_position_tolerance", 0.01)
            ),
            "shutdown_safe_velocity_tolerance": float(
                robot_cfg.get("shutdown_safe_velocity_tolerance", 0.05)
            ),
            "shutdown_safe_poll_interval_s": float(
                robot_cfg.get("shutdown_safe_poll_interval_s", 0.02)
            ),
            "dt": float(robot_cfg.get("dt", 0.01)),
            "feedback_timeout_s": float(robot_cfg.get("feedback_timeout_s", 2.0)),
            "home_timeout_s": float(robot_cfg.get("home_timeout_s", 25.0)),
            "home_command_grace_s": float(robot_cfg.get("home_command_grace_s", 0.5)),
            "home_settle_time_s": float(robot_cfg.get("home_settle_time_s", 0.5)),
            "home_velocity_tolerance": float(robot_cfg.get("home_velocity_tolerance", 0.05)),
            "home_position_tolerance": float(robot_cfg.get("home_position_tolerance", 0.005)),
            "home_poll_interval_s": float(robot_cfg.get("home_poll_interval_s", 0.05)),
            "sdk_root": robot_cfg.get("sdk_root"),
            "control_cpu_affinity": robot_cfg.get("control_cpu_affinity"),
            "eef_topics": {"left": topics["left_eef"], "right": topics["right_eef"]},
        },
        "owner": {
            "_target_": "prometheus.nodes.robots.owner.RobotOwnerNode",
            "robot_id": robot_id,
            "topics": {
                "state": topics["state"],
                "control": topics["control"],
                "result": topics["result"],
                "status": topics["status"],
            },
            "hz": state_publish_hz,
            "ros_domain_id": control_domain_id,
            "action_dims": ACTION_DIMS,
        },
        "client": {
            "_target_": "prometheus.nodes.robots.client.RobotControlClient",
            "action_space": action_space,
            "action_dim": 14,
            "control_topic": topics["control"],
            "result_topic": topics["result"],
            "command_source": f"{robot_id}_workflow",
            "ros_domain_id": control_domain_id,
        },
    }


def sensor_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    from prometheus.nodes.sensors.realsense import runtime_config as build_camera_runtime
    from prometheus.nodes.sensors.xense import runtime_config as build_tactile_runtime

    sensors = build_camera_runtime(robot)
    sensors.update(build_tactile_runtime(robot))
    return sensors


def streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from prometheus.nodes.sensors.realsense import streams as build_camera_streams
    from prometheus.nodes.sensors.xense import streams as build_tactile_streams

    items = robot_streams(robot)
    items.update(build_camera_streams(robot))
    items.update(build_tactile_streams(robot))
    return items


def robot_streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    control_domain_id = int(robot["control_domain_id"])
    topics = a5_topics(str(robot["id"]))
    return {
        "robot_state": {
            "topic": topics["state"],
            "msg_type": "prometheus.transport.messages.JointState",
            "domain": "robot",
            "ros_domain_id": control_domain_id,
        },
        "left_eef": {
            "topic": topics["left_eef"],
            "msg_type": "prometheus.transport.messages.PoseStamped",
            "domain": "robot",
            "ros_domain_id": control_domain_id,
        },
        "right_eef": {
            "topic": topics["right_eef"],
            "msg_type": "prometheus.transport.messages.PoseStamped",
            "domain": "robot",
            "ros_domain_id": control_domain_id,
        },
    }


def _default_sdk_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "A5"


def _robot_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    value = robot.get("robot")
    if not isinstance(value, Mapping):
        raise ValueError("A5 config requires a robot mapping")
    config = dict(value)
    missing = {"state_publish_hz", "action_space", "can"}.difference(config)
    if missing:
        raise ValueError(f"A5 robot config missing required keys: {sorted(missing)}")
    if str(config["action_space"]) not in A5_ACTION_SPACES:
        raise ValueError(f"A5 action_space must be one of {A5_ACTION_SPACES}")
    return config


def _side_mapping(robot: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = robot.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"A5 robot config requires {name}.left and {name}.right")
    missing = {"left", "right"}.difference(value)
    if missing:
        raise ValueError(f"A5 robot config {name} missing sides: {sorted(missing)}")
    return {"left": value["left"], "right": value["right"]}


def _optional_side_config(robot: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    if robot.get(name) is None:
        return None
    return _side_mapping(robot, name)


def _optional_side_floats(
    value: Mapping[str, float] | None,
    name: str,
    *,
    default: float | None = None,
    positive: bool = False,
    max_abs: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, float] | None:
    if value is None:
        if default is None:
            return None
        return {side: float(default) for side in A5DragDriver.sides}
    if not isinstance(value, Mapping):
        raise ValueError(f"A5 {name} must contain left and right values")
    missing = set(A5DragDriver.sides).difference(value)
    if missing:
        raise ValueError(f"A5 {name} missing sides: {sorted(missing)}")

    result = {side: float(value[side]) for side in A5DragDriver.sides}
    for side, number in result.items():
        if not math.isfinite(number):
            raise ValueError(f"A5 {name}.{side} must be finite")
        if positive and number <= 0:
            raise ValueError(f"A5 {name}.{side} must be positive")
        if max_abs is not None and abs(number) > max_abs:
            raise ValueError(f"A5 {name}.{side} must be between {-max_abs} and {max_abs}")
        if minimum is not None and number < minimum:
            raise ValueError(f"A5 {name}.{side} must be between {minimum} and {maximum}")
        if maximum is not None and number > maximum:
            raise ValueError(f"A5 {name}.{side} must be between {minimum} and {maximum}")
    return result


def _optional_side_vectors(
    value: Mapping[str, Any] | None,
    name: str,
    *,
    size: int,
    max_abs: float,
) -> dict[str, np.ndarray] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"A5 {name} must contain left and right values")
    missing = set(A5DragDriver.sides).difference(value)
    if missing:
        raise ValueError(f"A5 {name} missing sides: {sorted(missing)}")

    result: dict[str, np.ndarray] = {}
    for side in A5DragDriver.sides:
        vector = np.asarray(value[side], dtype=np.float64).reshape(-1)
        if vector.shape != (size,):
            raise ValueError(f"A5 {name}.{side} must contain {size} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"A5 {name}.{side} must contain finite values")
        if float(np.max(np.abs(vector))) > max_abs:
            raise ValueError(
                f"A5 {name}.{side} values must be between {-max_abs} and {max_abs}"
            )
        result[side] = vector.copy()
    return result


def _link6_mass_element(tree: ET.ElementTree, source: Path) -> ET.Element:
    for link in tree.getroot().findall("link"):
        if link.get("name") != "link6":
            continue
        mass = link.find("./inertial/mass")
        if mass is not None and mass.get("value") is not None:
            return mass
        break
    raise ValueError(f"A5 URDF has no link6 inertial mass: {source}")


def _state_vector(value: Any, name: str) -> np.ndarray:
    # Native feedback can contain a transient huge sentinel while CAN motors
    # come online. Preserve it as inf without emitting a NumPy overflow warning;
    # the caller's finite-value check will wait for valid feedback or fail safely.
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float64).reshape(-1).astype(np.float32)
    if array.shape != (7,):
        raise ValueError(f"A5 {name} must contain 7 values, got shape {array.shape}")
    return array.copy()


def _joint_command(value: Any) -> np.ndarray:
    command = np.asarray(value, dtype=np.float64)
    if command.shape != (14,):
        raise ValueError(f"A5 qpos command must have shape (14,), got {command.shape}")
    if not np.all(np.isfinite(command)):
        raise ValueError("A5 qpos command must contain finite values")
    return command.copy()


def _has_feedback(arm: Any) -> bool:
    values = np.concatenate(
        [
            _state_vector(arm.get_joint_positions(), "feedback qpos"),
            _state_vector(arm.get_joint_torques(), "feedback effort"),
            _state_vector(arm.get_ee_pose(), "feedback eef pose"),
        ]
    )
    return bool(np.all(np.isfinite(values)) and np.any(np.abs(values) > 1e-6))


def _concat_sides(states: Mapping[str, Mapping[str, np.ndarray]], name: str) -> np.ndarray:
    return np.concatenate([states[side][name] for side in A5DragDriver.sides]).astype(np.float32)


def _side_vector(values: np.ndarray, side: str) -> np.ndarray:
    start = SIDE_VECTOR_START[side]
    return values[start : start + 7]
