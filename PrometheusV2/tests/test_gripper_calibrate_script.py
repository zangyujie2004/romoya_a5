from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gripper_calibrate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("gripper_calibrate_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeArm:
    def __init__(self, readouts):
        self.readouts = iter(readouts)

    def get_joint_positions(self):
        joints = np.zeros(7, dtype=np.float64)
        joints[6] = next(self.readouts)
        return joints


def test_sample_a5_gripper_returns_median_and_spread():
    script = load_script()

    median, spread = script.sample_a5_gripper(
        FakeArm([-3.30, -3.32, -3.31]),
        samples=3,
        interval_s=0.0,
    )

    assert median == pytest.approx(-3.31)
    assert spread == pytest.approx(0.02)


def test_sample_a5_gripper_rejects_invalid_count():
    script = load_script()

    with pytest.raises(ValueError, match="samples must be positive"):
        script.sample_a5_gripper(FakeArm([]), samples=0, interval_s=0.0)
