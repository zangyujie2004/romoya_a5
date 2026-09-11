from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from prometheus.nodes.robots.a5 import A5DragDriver


class FakeSingleArm:
    instances: list["FakeSingleArm"] = []

    def __init__(self, config):
        self.config = dict(config)
        self.gravity_calls = 0
        self.gripper_torque_calls = []
        self.gripper_position_calls = []
        self.arm_position_calls = []
        self.protect_calls = 0
        self.go_home_calls = 0
        self.homing = False
        self.positioning = False
        self.offset = 0.0 if config["can_port"] == "can1" else 10.0
        self.joint_positions = np.arange(6, dtype=np.float64) + self.offset
        self.gripper_position = 6.0 + self.offset
        self.instances.append(self)

    def gravity_compensation(self):
        self.gravity_calls += 1
        self.homing = False

    def set_gripper_ext_torque(self, torque):
        self.gripper_torque_calls.append(float(torque))

    def set_gripper_pos(self, position):
        self.gripper_position_calls.append(float(position))
        self.gripper_position = float(position)

    def set_joint_positions(self, positions):
        values = np.asarray(positions, dtype=np.float64).copy()
        self.arm_position_calls.append(values)
        self.joint_positions = values
        self.positioning = True

    def protect_mode(self):
        self.protect_calls += 1

    def go_home(self):
        self.go_home_calls += 1
        self.homing = True
        return True

    def get_joint_positions(self):
        positions = np.concatenate((self.joint_positions, [self.gripper_position])).astype(np.float32)
        return positions

    def get_joint_velocities(self):
        if self.homing or self.positioning:
            return np.zeros(7, dtype=np.float32)
        return np.arange(7, dtype=np.float32) + self.offset + 20

    def get_joint_torques(self):
        return np.arange(7, dtype=np.float32) + self.offset + 40

    def get_ee_pose(self):
        return np.arange(7, dtype=np.float32) + self.offset + 60


@pytest.fixture(autouse=True)
def clear_fake_arms():
    FakeSingleArm.instances.clear()


def _driver(monkeypatch, **kwargs) -> A5DragDriver:
    kwargs.setdefault("home_timeout_s", 0.1)
    kwargs.setdefault("home_command_grace_s", 0.001)
    kwargs.setdefault("home_settle_time_s", 0.001)
    kwargs.setdefault("home_poll_interval_s", 0.001)
    driver = A5DragDriver(left_can="can1", right_can="can3", **kwargs)
    monkeypatch.setattr(driver, "_import_single_arm", lambda: FakeSingleArm)
    return driver


def test_a5_policy_gripper_offset_clips_both_bounds_without_mutating_predictions(monkeypatch):
    from prometheus.nodes.robots import a5

    monkeypatch.setattr(a5, "GIPPER_OFFSET", 0.1)
    driver = _driver(monkeypatch)
    raw = np.tile(np.arange(14, dtype=np.float32), (3, 1))
    raw[:, [6, 13]] = [[-1.0, -0.05], [-3.5, 0.2], [-3.14, -1.5]]
    original = raw.copy()
    actual = driver.prepare_policy_actions(raw)

    np.testing.assert_allclose(actual[:, [6, 13]], [[-0.9, 0], [-3.14, 0], [-3.04, -1.4]], atol=1e-6)
    joints = [i for i in range(14) if i not in (6, 13)]
    np.testing.assert_array_equal(actual[:, joints], original[:, joints])
    np.testing.assert_array_equal(raw, original)
    assert not FakeSingleArm.instances


def test_a5_drag_driver_reads_left_then_right(monkeypatch):
    driver = _driver(monkeypatch)
    driver.connect()
    try:
        state = driver.read_state()

        assert state.joint_names[:7] == [
            "left_joint_1",
            "left_joint_2",
            "left_joint_3",
            "left_joint_4",
            "left_joint_5",
            "left_joint_6",
            "left_gripper",
        ]
        assert state.joint_names[7] == "right_joint_1"
        assert np.allclose(state.qpos[:6], np.arange(6))
        assert state.qpos[6] == 0.0
        assert np.allclose(state.qpos[7:13], np.arange(6) + 10)
        assert state.qpos[13] == 0.0
        assert np.allclose(state.qvel[:7], np.arange(7) + 20)
        assert np.allclose(state.effort[7:], np.arange(7) + 50)
        assert np.allclose(state.eef_wxyz[:7], np.arange(7) + 60)
        assert driver.status()["motion_commands_enabled"] is True
    finally:
        driver.close()


def test_a5_drag_driver_enters_gravity_compensation_and_protects_on_close(monkeypatch):
    driver = _driver(monkeypatch)
    driver.connect()
    arms = list(FakeSingleArm.instances)

    assert driver.control_mode == "teach"
    assert all(arm.gravity_calls == 1 for arm in arms)
    assert all(arm.go_home_calls == 1 for arm in arms)
    assert all(arm.gripper_position_calls == [0.0] for arm in arms)
    assert all(arm.gripper_torque_calls == [0.0, 0.0] for arm in arms)

    driver.read_state()
    assert all(arm.gravity_calls == 1 for arm in arms)
    assert all(arm.gripper_torque_calls == [0.0, 0.0] for arm in arms)

    driver.close()
    assert all(arm.protect_calls == 1 for arm in arms)
    assert all(arm.gripper_torque_calls == [0.0, 0.0, 0.0] for arm in arms)
    assert driver.is_connected is False


def test_a5_drag_driver_applies_side_gripper_torque_in_teach_mode(monkeypatch):
    driver = _driver(
        monkeypatch,
        teach_gripper_torque={"left": 0.01, "right": -0.02},
    )
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        assert arms[0].gripper_torque_calls == [0.0, 0.01]
        assert arms[1].gripper_torque_calls == [0.0, -0.02]
    finally:
        driver.close()

    assert arms[0].gripper_torque_calls[-1] == 0.0
    assert arms[1].gripper_torque_calls[-1] == 0.0


def test_a5_drag_driver_generates_per_side_payload_urdfs(monkeypatch, tmp_path):
    source = tmp_path / "a5.urdf"
    source.write_text(
        '<robot name="a5"><link name="link6"><inertial><mass value="0.6" />'
        "</inertial></link></robot>",
        encoding="ascii",
    )
    driver = _driver(
        monkeypatch,
        urdf_name=str(source),
        link6_mass_kg={"left": 0.75, "right": 0.8},
    )
    driver.connect()
    try:
        generated = [Path(arm.config["urdf_name"]) for arm in FakeSingleArm.instances]
        assert all(path.is_file() for path in generated)
        masses = [
            float(ET.parse(path).find("./link[@name='link6']/inertial/mass").get("value"))
            for path in generated
        ]
        assert masses == [0.75, 0.8]
    finally:
        driver.close()

    assert all(not path.exists() for path in generated)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"link6_mass_kg": {"left": 0.0, "right": 0.6}}, "must be positive"),
        ({"teach_gripper_torque": {"left": 0.11, "right": 0.0}}, "between -0.1 and 0.1"),
    ],
)
def test_a5_drag_driver_rejects_unsafe_teach_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        A5DragDriver(**kwargs)


def test_a5_drag_driver_resets_home_with_closed_grippers(monkeypatch):
    driver = _driver(
        monkeypatch,
    )
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        driver.reset_home()

        assert all(arm.go_home_calls == 2 for arm in arms)
        assert all(arm.gravity_calls == 2 for arm in arms)
        assert all(arm.gripper_position_calls == [0.0, 0.0] for arm in arms)
        assert driver.control_mode == "teach"
        assert driver.status()["reset_home_enabled"] is True
    finally:
        driver.close()


def test_a5_drag_driver_protects_both_arms_when_home_times_out(monkeypatch):
    class MovingArm(FakeSingleArm):
        fail_homing = False

        def get_joint_velocities(self):
            if self.fail_homing:
                return np.ones(7, dtype=np.float32)
            return super().get_joint_velocities()

    driver = A5DragDriver(
        home_timeout_s=0.01,
        home_command_grace_s=0.001,
        home_settle_time_s=0.001,
        home_poll_interval_s=0.001,
    )
    monkeypatch.setattr(driver, "_import_single_arm", lambda: MovingArm)
    driver.connect()
    arms = list(MovingArm.instances)
    try:
        for arm in arms:
            arm.fail_homing = True
        with pytest.raises(TimeoutError, match="did not settle at home"):
            driver.reset_home()

        assert all(arm.protect_calls == 1 for arm in arms)
        assert driver.control_mode == "protect"
    finally:
        driver.close()


def test_a5_drag_driver_moves_smoothly_to_shutdown_safe_position(monkeypatch):
    target = {
        "left": [-0.01, 0.0, -0.01, -0.17, 0.0, 0.03],
        "right": [0.0, 0.0, -0.02, -0.07, -0.01, -0.02],
    }
    driver = _driver(
        monkeypatch,
        shutdown_safe_position=target,
        shutdown_safe_max_velocity=100.0,
        shutdown_safe_timeout_s=0.5,
        shutdown_safe_settle_time_s=0.001,
        shutdown_safe_position_tolerance=0.001,
        shutdown_safe_poll_interval_s=0.001,
        dt=0.001,
    )
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        driver.move_to_shutdown_safe_position()

        assert driver.control_mode == "shutdown_safe"
        assert all(len(arm.arm_position_calls) > 1 for arm in arms)
        np.testing.assert_allclose(arms[0].arm_position_calls[-1], target["left"])
        np.testing.assert_allclose(arms[1].arm_position_calls[-1], target["right"])
    finally:
        driver.close()


def test_a5_drag_driver_requires_configured_shutdown_safe_position(monkeypatch):
    driver = _driver(monkeypatch)
    driver.connect()
    try:
        with pytest.raises(RuntimeError, match="shutdown_safe_position is not configured"):
            driver.move_to_shutdown_safe_position()
    finally:
        driver.close()


def test_a5_drag_driver_does_not_restore_teach_mode_after_shutdown_request(monkeypatch):
    driver = _driver(monkeypatch)
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        driver._shutdown_requested.set()
        with pytest.raises(RuntimeError, match="cancelled by shutdown"):
            driver.reset_home()

        assert all(arm.gravity_calls == 1 for arm in arms)
        assert all(arm.protect_calls == 1 for arm in arms)
        assert driver.control_mode == "protect"
    finally:
        driver.close()


def test_a5_drag_driver_attempts_to_protect_both_arms_when_one_fails(monkeypatch):
    class ProtectFailArm(FakeSingleArm):
        def protect_mode(self):
            super().protect_mode()
            if self.config["can_port"] == "can1":
                raise RuntimeError("left protect failed")

    driver = A5DragDriver()
    monkeypatch.setattr(driver, "_import_single_arm", lambda: ProtectFailArm)
    driver.connect()
    arms = list(ProtectFailArm.instances)

    with pytest.raises(RuntimeError, match="left: left protect failed"):
        driver.close()

    assert all(arm.protect_calls == 1 for arm in arms)
    assert driver.is_connected is False


def test_a5_drag_driver_requires_connection_for_motion():
    driver = A5DragDriver()

    with pytest.raises(RuntimeError, match="not connected"):
        driver.send_action("abs_qpos", np.zeros(14, dtype=np.float32))

    with pytest.raises(RuntimeError, match="not connected"):
        driver.set_position_mode()


def test_a5_position_mode_holds_current_pose_and_clears_teach_torque(monkeypatch):
    driver = _driver(monkeypatch, teach_gripper_torque={"left": 0.01, "right": -0.02})
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        for arm, gripper in zip(arms, (-2.7, -3.3)):
            arm.gripper_position = gripper
        current = driver.read_state().qpos.copy()
        driver.set_position_mode()
        driver.set_position_mode()

        assert driver.control_mode == "position"
        for arm, target in zip(arms, current.reshape(2, 7)):
            assert len(arm.arm_position_calls) == 1
            np.testing.assert_array_equal(arm.arm_position_calls[0], target[:6])
            assert arm.gripper_position_calls[-1] == pytest.approx(target[6])
            assert arm.gripper_torque_calls[-1] == 0.0

        driver.reset_home()
        assert driver.control_mode == "teach"
        assert [arm.gripper_torque_calls[-1] for arm in arms] == [0.01, -0.02]
    finally:
        driver.close()


def test_a5_joint_actions_preserve_side_order_and_native_gripper_units(monkeypatch):
    driver = _driver(monkeypatch)
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        command = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
        command[[6, 13]] = [-3.3, -1.2]
        driver.send_action("abs_qpos", command)
        driver.send_action("abs_qpos", command * 0.5)

        assert driver.control_mode == "position"
        for arm, target in zip(arms, command.reshape(2, 7)):
            # One hold command on entering position mode, then two actions.
            assert len(arm.arm_position_calls) == 3
            np.testing.assert_array_equal(arm.arm_position_calls[1], target[:6])
            np.testing.assert_array_equal(arm.arm_position_calls[2], target[:6] * 0.5)
            assert arm.gripper_position_calls[-2:] == pytest.approx([target[6], target[6] * 0.5])
        np.testing.assert_array_equal(driver.read_state().qpos, command * 0.5)
    finally:
        driver.close()


@pytest.mark.parametrize(
    ("action_space", "command", "message"),
    [
        ("abs_eef", np.zeros(14), "action_space"),
        ("delta_qpos", np.zeros(14), "action_space"),
        ("abs_qpos", np.zeros(13), "shape"),
        ("abs_qpos", np.zeros((2, 7)), "shape"),
        ("abs_qpos", np.array([0.0] * 13 + [np.nan]), "finite"),
        ("abs_qpos", np.array([0.0] * 13 + [np.inf]), "finite"),
    ],
)
def test_a5_rejects_invalid_actions_before_changing_either_arm(monkeypatch, action_space, command, message):
    driver = _driver(monkeypatch)
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        with pytest.raises(ValueError, match=message):
            driver.send_action(action_space, command)

        assert driver.control_mode == "teach"
        assert all(arm.arm_position_calls == [] for arm in arms)
        assert all(arm.gripper_position_calls == [0.0] for arm in arms)
    finally:
        driver.close()


def test_a5_position_mode_rejects_invalid_right_feedback_before_moving_left(monkeypatch):
    driver = _driver(monkeypatch)
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        arms[1].joint_positions[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            driver.set_position_mode()

        assert all(arm.arm_position_calls == [] for arm in arms)
        assert all(arm.protect_calls == 1 for arm in arms)
        assert driver.control_mode == "protect"
    finally:
        driver.close()


@pytest.mark.parametrize("method", ["set_joint_positions", "set_gripper_pos"])
@pytest.mark.parametrize("failure", ["exception", "rejection"])
@pytest.mark.parametrize("enter_position_mode", [True, False])
def test_a5_protects_both_arms_on_sdk_command_failure(monkeypatch, method, failure, enter_position_mode):
    driver = _driver(monkeypatch)
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        if not enter_position_mode:
            driver.set_position_mode()

        def fail(value):
            if failure == "exception":
                raise RuntimeError("SDK command failed")
            return False

        monkeypatch.setattr(arms[1], method, fail)
        with pytest.raises(RuntimeError, match="SDK command failed|command was rejected"):
            driver.send_action("abs_qpos", np.zeros(14))

        assert all(arm.protect_calls == 1 for arm in arms)
        assert all(arm.gripper_torque_calls[-1] == 0.0 for arm in arms)
        assert driver.control_mode == "protect"
    finally:
        driver.close()


def test_a5_motion_does_not_resume_during_shutdown(monkeypatch):
    driver = _driver(monkeypatch)
    driver.connect()
    arms = list(FakeSingleArm.instances)
    try:
        driver._shutdown_requested.set()
        with pytest.raises(RuntimeError, match="cancelled by shutdown"):
            driver.send_action("abs_qpos", np.zeros(14))
        with pytest.raises(RuntimeError, match="cancelled by shutdown"):
            driver.set_position_mode()
        assert all(arm.arm_position_calls == [] for arm in arms)
    finally:
        driver.close()


def test_a5_drag_driver_rejects_missing_motor_feedback(monkeypatch):
    class SilentArm(FakeSingleArm):
        def get_joint_positions(self):
            return np.zeros(7, dtype=np.float32)

        def get_joint_torques(self):
            return np.zeros(7, dtype=np.float32)

        def get_ee_pose(self):
            return np.zeros(7, dtype=np.float32)

    driver = A5DragDriver(feedback_timeout_s=0.01)
    monkeypatch.setattr(driver, "_import_single_arm", lambda: SilentArm)

    with pytest.raises(RuntimeError, match="no motor feedback"):
        driver.connect()

    assert driver.is_connected is False
