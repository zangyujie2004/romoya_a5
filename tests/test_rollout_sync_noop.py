from __future__ import annotations

import json
import pickle
import pytest

from omegaconf import OmegaConf

from prometheus.workflows.rollout_sync import run_from_config


@pytest.mark.parametrize("cpus", [None, [8, 9, 10, 11, 12, 13, 14, 15]])
def test_cpu_affinity_precedes_model_and_hardware_creation(tmp_path, monkeypatch, cpus):
    from prometheus.workflows import rollout_sync
    from prometheus.sessions.hardware import HardwareSession

    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    if cpus is not None:
        cfg.workflow.data_cpu_affinity = cpus
    events = []
    build_hardware = rollout_sync.build_hardware_session
    instantiate = rollout_sync.instantiate_config
    start = HardwareSession.start

    def affinity(value):
        assert value == cpus
        events.append("affinity")

    def hardware(**kwargs):
        events.append("build_hardware")
        return build_hardware(**kwargs)

    def create(config, name, **kwargs):
        events.append(name)
        return instantiate(config, name, **kwargs)

    def start_hardware(self):
        events.append("start_hardware")
        return start(self)

    monkeypatch.setattr(rollout_sync, "configure_process_cpu_affinity", affinity)
    monkeypatch.setattr(rollout_sync, "build_hardware_session", hardware)
    monkeypatch.setattr(rollout_sync, "instantiate_config", create)
    monkeypatch.setattr(HardwareSession, "start", start_hardware)
    assert run_from_config(cfg) == 0
    assert events == ["affinity", "build_hardware", "policy", "start_hardware", "scheduler"]


@pytest.mark.parametrize('stop', ['completed', 'interrupted', 'error'])
def test_end_opening_precedes_return_on_completion_and_interrupt(tmp_path, monkeypatch, stop):
    from prometheus.workflows import rollout_sync
    from tests.noop import NoopPolicy

    cfg = OmegaConf.load('tests/configs/rollout_sync_noop.yaml')
    cfg.run.output_dir = str(tmp_path / 'output')
    cfg.run.runtime_dir = str(tmp_path / 'output' / 'runtime')
    cfg.run.steps = 1
    events = []
    original_infer = NoopPolicy.infer

    def infer(self, data):
        events.append('infer')
        if stop == 'interrupted':
            raise KeyboardInterrupt
        if stop == 'error':
            raise ValueError('policy failed')
        return original_infer(self, data)

    def return_pose(*args):
        events.append('return')
        return {'status': 'disabled'}

    monkeypatch.setattr(NoopPolicy, 'infer', infer)
    monkeypatch.setattr(rollout_sync, '_open_grippers_after_policy', lambda *args: events.append('open'))
    monkeypatch.setattr(rollout_sync, '_return_to_shutdown_pose', return_pose)
    if stop == 'error':
        with pytest.raises(ValueError, match='policy failed'):
            run_from_config(cfg)
    else:
        assert run_from_config(cfg) == (0 if stop == 'completed' else 130)
    assert events == (['infer', 'open', 'return'] if stop in {'completed', 'interrupted'} else ['infer', 'return'])


@pytest.mark.parametrize('failure', [TimeoutError('gripper did not settle'), KeyboardInterrupt()])
def test_exit_opening_failure_skips_motion_and_still_cleans_up(tmp_path, monkeypatch, failure):
    from prometheus.workflows import rollout_sync
    from tests.noop import NoopPolicy

    cfg = OmegaConf.load('tests/configs/rollout_sync_noop.yaml')
    cfg.run.output_dir = str(tmp_path)
    cfg.run.runtime_dir = str(tmp_path / 'runtime')
    cfg.workflow.return_to_shutdown_pose = True
    events = []
    cleanup = rollout_sync._stop_data_and_hardware

    def infer(*args):
        raise KeyboardInterrupt

    def opening(*args):
        events.append('open')
        raise failure

    def stop(data, hardware):
        events.append('cleanup')
        cleanup(data, hardware)

    monkeypatch.setattr(NoopPolicy, 'infer', infer)
    monkeypatch.setattr(rollout_sync, '_open_grippers_after_policy', opening)
    monkeypatch.setattr(rollout_sync, '_stop_data_and_hardware', stop)
    with pytest.raises(RuntimeError, match='shutdown gripper opening failed'):
        run_from_config(cfg)
    manifest = json.loads((tmp_path / 'workflow_manifest.json').read_text())
    assert manifest['status'] == 'error'
    assert manifest['shutdown_pose']['status'] == 'skipped'
    assert manifest['stop_reason'] == 'shutdown_gripper_open_failed'
    assert events == ['open', 'cleanup']


def test_rollout_sync_preserves_startup_exception(tmp_path, monkeypatch):
    from prometheus.sessions.hardware import HardwareSession

    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")

    def fail_start(self):
        raise FileNotFoundError("A5 SDK is not built")

    monkeypatch.setattr(HardwareSession, "start", fail_start)
    with pytest.raises(FileNotFoundError, match="A5 SDK is not built"):
        run_from_config(cfg)

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["chunks"] == 0
    assert manifest["actions_sent"] == 0


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
