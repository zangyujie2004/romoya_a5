from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_a5_dual_arm_lifecycle.py"


def load_script():
    spec = importlib.util.spec_from_file_location("test_a5_dual_arm_lifecycle_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDriver:
    def __init__(self, *, read_error=None, reset_error=None):
        self.read_error = read_error
        self.reset_error = reset_error
        self.calls = []

    def status(self):
        return {
            "link6_mass_kg": {"left": 0.6, "right": 0.6},
            "teach_gripper_torque": {"left": 0.0, "right": 0.0},
        }

    def connect(self):
        self.calls.append("connect")

    def read_state(self):
        self.calls.append("read_state")
        raise self.read_error

    def reset_home(self):
        self.calls.append("reset_home")
        if self.reset_error is not None:
            raise self.reset_error

    def close(self):
        self.calls.append("close")


def test_ctrl_c_homes_then_closes():
    script = load_script()
    driver = FakeDriver(read_error=KeyboardInterrupt())

    result = script.run_lifecycle(driver, poll_hz=50.0, countdown_s=0)

    assert result == 0
    assert driver.calls == ["connect", "read_state", "reset_home", "close"]


def test_second_ctrl_c_cancels_home_then_closes():
    script = load_script()
    driver = FakeDriver(read_error=KeyboardInterrupt(), reset_error=KeyboardInterrupt())

    result = script.run_lifecycle(driver, poll_hz=50.0, countdown_s=0)

    assert result == 130
    assert driver.calls == ["connect", "read_state", "reset_home", "close"]


def test_runtime_error_skips_home_then_closes():
    script = load_script()
    driver = FakeDriver(read_error=RuntimeError("feedback lost"))

    result = script.run_lifecycle(driver, poll_hz=50.0, countdown_s=0)

    assert result == 1
    assert driver.calls == ["connect", "read_state", "close"]


def test_print_state_keeps_left_then_right_order(capsys):
    script = load_script()
    state = SimpleNamespace(qpos=__import__("numpy").arange(14, dtype=float))

    script.print_state(state)

    output = capsys.readouterr().out
    assert "left=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]" in output
    assert "right=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]" in output
