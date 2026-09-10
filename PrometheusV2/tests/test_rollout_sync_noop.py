from __future__ import annotations

import json
import pickle

from omegaconf import OmegaConf

from prometheus.workflows.rollout_sync import run_from_config


def test_rollout_sync_noop(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["chunks"] == 3
    assert manifest["actions_sent"] == 3
    assert manifest["task"]["id"] == "noop_task"
    assert manifest["robot_client"]["sent_actions"] == 3


def test_rollout_sync_records_action_stream_when_enabled(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.data.recording = {
        "enabled": True,
        "output_dir": str(tmp_path / "output" / "episodes"),
        "episode_id": "0001",
        "queue_size": 16,
        "fps": 30.0,
    }

    assert run_from_config(cfg) == 0

    episode_dir = tmp_path / "output" / "episodes" / "0001"
    manifest = json.loads((episode_dir / "manifests" / "action.json").read_text())
    with (episode_dir / "action" / "action_chunk_dict.pkl").open("rb") as file:
        chunks = pickle.load(file)
    with (episode_dir / "action" / "executed_action_dict.pkl").open("rb") as file:
        actions = pickle.load(file)

    action_manifest = manifest["prometheus_action_recorder"]
    assert action_manifest["status"] == "saved"
    assert action_manifest["counts"]["action.policy_chunk"] == 3
    assert action_manifest["counts"]["action.executed"] == 3
    assert action_manifest["policy_action_chunks"] == 3
    assert action_manifest["executed_actions"] == 3
    assert len(chunks["chunks"]) == 3
    assert len(actions["actions"]) == 3
    assert actions["action_space"] == ["noop_action", "noop_action", "noop_action"]
    assert actions["hz"] == [20.0, 20.0, 20.0]
