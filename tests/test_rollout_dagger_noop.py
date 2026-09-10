from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from prometheus.workflows.rollout_dagger import run_from_config


def _load_noop_cfg(tmp_path: Path, config_name: str = "rollout_dagger_noop.yaml") -> OmegaConf:
    cfg = OmegaConf.load(f"tests/configs/{config_name}")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.data.recording.output_dir = str(tmp_path / "output" / "episodes")
    return cfg


def test_rollout_dagger_noop_bootstrap_cycle_and_teaching(tmp_path):
    cfg = _load_noop_cfg(tmp_path)

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["workflow"] == "rollout_dagger"
    assert manifest["status"] == "completed"
    assert manifest["stop_reason"] == "max_cycles"
    assert manifest["cycles_completed"] == 1
    assert manifest["actions_sent"] > 0
    assert manifest["task"]["id"] == "noop_task"
    assert manifest["controller"]["phase"] == "autonomous"
    assert manifest["controller"]["event_level"] == 0

    robot_client = manifest["robot_client"]
    assert robot_client["reset_count"] == 1
    assert robot_client["position_count"] == 2  # bootstrap + post-cycle
    assert robot_client["teach_count"] == 1

    episode_zero = tmp_path / "output" / "episodes" / "0000"
    episode_one = tmp_path / "output" / "episodes" / "0001"
    assert episode_zero.is_dir()
    assert episode_one.is_dir()
    assert not (episode_zero / "action").exists()

    with (episode_zero / "event" / "event_dict.pkl").open("rb") as file:
        events = pickle.load(file)
    assert "event_label" in events["names"]
    assert 0 in events["data"]
    assert 1 in events["data"]


def test_rollout_dagger_noop_pause_hold_and_cycle(tmp_path):
    cfg = _load_noop_cfg(tmp_path, "rollout_dagger_noop_pause.yaml")

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["stop_reason"] == "max_cycles"
    assert manifest["cycles_completed"] == 1
    assert manifest["actions_sent"] > 0

    robot_client = manifest["robot_client"]
    assert robot_client["reset_count"] == 1
    assert robot_client["position_count"] == 2
    assert robot_client["teach_count"] == 1
    # hold during PAUSED should still send actions after the pause step
    assert robot_client["sent_actions"] > manifest["chunk_steps"]

    episode_zero = tmp_path / "output" / "episodes" / "0000"
    with (episode_zero / "event" / "event_dict.pkl").open("rb") as file:
        events = pickle.load(file)
    assert 0 in events["data"]
    assert 1 in events["data"]


def test_rollout_dagger_noop_rejects_recording_disabled(tmp_path):
    cfg = _load_noop_cfg(tmp_path)
    cfg.data.recording.enabled = False

    with pytest.raises(ValueError, match="recording.enabled=true"):
        run_from_config(cfg)


def test_rollout_dagger_noop_realtime_vis_enabled(tmp_path):
    cfg = _load_noop_cfg(tmp_path)
    cfg.workflow.realtime_vis = True

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["cycles_completed"] == 1
