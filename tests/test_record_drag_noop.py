from __future__ import annotations

import json
import pickle

from omegaconf import OmegaConf

from prometheus.workflows.record_drag import (
    _move_robot_to_shutdown_safe_position,
    _reset_robot_after_episode,
    run_from_config,
)


def test_record_drag_noop_records_event_episode(tmp_path):
    cfg = OmegaConf.load("tests/configs/record_drag_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.data.recording.output_dir = str(tmp_path / "output" / "episodes")

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["stop_reason"] == "max_episodes"
    assert manifest["episodes_started"] == 1
    assert manifest["episodes_saved"] == 1
    assert manifest["events_ignored"] == 0
    assert manifest["task"]["id"] == "noop_task"

    event_path = tmp_path / "output" / "episodes" / "0000" / "event" / "event_dict.pkl"
    assert not (tmp_path / "output" / "episodes" / "0000" / "action").exists()
    with event_path.open("rb") as file:
        events = pickle.load(file)

    assert events["names"] == ["record_start", "event_mark", "record_stop"]
    assert events["data"] == [True, True, True]


def test_record_drag_noop_appends_after_existing_episode_dir(tmp_path):
    cfg = OmegaConf.load("tests/configs/record_drag_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.data.recording.output_dir = str(tmp_path / "output" / "episodes")
    (tmp_path / "output" / "episodes" / "0000").mkdir(parents=True)

    assert run_from_config(cfg) == 0

    assert (tmp_path / "output" / "episodes" / "0000").is_dir()
    assert (tmp_path / "output" / "episodes" / "0001" / "event" / "event_dict.pkl").is_file()


def test_reset_after_episode_can_skip_shutdown_motion():
    class RobotClient:
        def __init__(self):
            self.calls = []

        def reset_home(self, *, timeout, wait):
            self.calls.append((timeout, wait))
            return {"accepted": True}

    class Hardware:
        robot_client = RobotClient()

    hardware = Hardware()
    workflow = {
        "reset_after_episode": {
            "enabled": True,
            "timeout_s": 30.0,
            "on_shutdown": False,
        }
    }

    _reset_robot_after_episode(hardware, workflow, "0000", during_shutdown=True)
    assert hardware.robot_client.calls == []

    _reset_robot_after_episode(hardware, workflow, "0000")
    assert hardware.robot_client.calls == [(30.0, True)]


def test_shutdown_safe_pose_uses_dedicated_robot_command():
    class RobotClient:
        def __init__(self):
            self.calls = []

        def move_to_shutdown_safe_position(self, *, timeout, wait):
            self.calls.append((timeout, wait))
            return {"accepted": True}

    class Hardware:
        robot_client = RobotClient()

    hardware = Hardware()
    workflow = {"shutdown_safe_pose": {"enabled": True, "timeout_s": 35.0}}

    _move_robot_to_shutdown_safe_position(hardware, workflow)

    assert hardware.robot_client.calls == [(35.0, True)]

