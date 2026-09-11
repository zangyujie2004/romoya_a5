import numpy as np
import pytest

from prometheus.policy.scheduler import ActionChunk
from prometheus.policy.smoothing import PolicyActionSmoother


def chunk(values):
    return ActionChunk(values, action_space='abs_qpos', hz=30, metadata={'done': True})


def test_feedback_seed_smooths_first_target_preserving_raw_and_grippers():
    original = chunk(np.ones((2, 14)))
    smoother = PolicyActionSmoother(enabled=True, alpha=.6)
    smoother.reset(np.zeros(14))
    result = smoother.apply(original)
    joints = [i for i in range(14) if i not in (6, 13)]
    np.testing.assert_allclose(result.actions[0, joints], .4)
    np.testing.assert_allclose(result.actions[1, joints], .64)
    np.testing.assert_array_equal(result.actions[:, [6, 13]], 1)
    np.testing.assert_array_equal(original.actions, 1)
    assert original.metadata == {'done': True}
    assert result.metadata['done']
    assert result.hz == 30


def test_chunk_boundary_does_not_reset_smoothing():
    targets = np.zeros((8, 14), dtype=np.float32)
    targets[:4] = 1
    targets[4:] = -1
    whole = PolicyActionSmoother(enabled=True, gripper=True)
    split = PolicyActionSmoother(enabled=True, gripper=True)
    whole.reset(np.zeros(14))
    split.reset(np.zeros(14))
    expected = whole.apply(chunk(targets)).actions
    actual = np.concatenate([split.apply(chunk(targets[:4])).actions, split.apply(chunk(targets[4:])).actions])
    np.testing.assert_allclose(actual, expected)
    assert -1 < actual[4, 0] < actual[3, 0]


def test_disabled_is_identity_and_reset_discards_old_state():
    original = chunk(np.ones(14))
    assert PolicyActionSmoother().apply(original) is original
    smoother = PolicyActionSmoother(enabled=True, gripper=True)
    with pytest.raises(RuntimeError, match='seed'):
        smoother.apply(original)
    smoother.reset(np.zeros(14))
    smoother.apply(original)
    smoother.reset(np.ones(14))
    np.testing.assert_array_equal(smoother.apply(original).actions, 1)


@pytest.mark.parametrize('alpha', [-.1, 1, float('nan'), float('inf')])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match='alpha'):
        PolicyActionSmoother(enabled=True, alpha=alpha)


def test_zero_alpha_preserves_targets():
    smoother = PolicyActionSmoother(enabled=True, alpha=0, gripper=True)
    smoother.reset(np.zeros(14))
    original = chunk(np.arange(28).reshape(2, 14))
    np.testing.assert_array_equal(smoother.apply(original).actions, original.actions)


def test_workflow_smooths_policy_only_and_records_raw_predictions(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from prometheus.sessions.hardware import HardwareSession
    from prometheus.sessions.data import DataSession
    from prometheus.workflows import rollout_sync

    cfg = OmegaConf.load('tests/configs/rollout_sync_noop.yaml')
    cfg.run.output_dir = str(tmp_path)
    cfg.run.runtime_dir = str(tmp_path / 'runtime')
    cfg.run.steps = 2
    cfg.robot.action_dim = 14
    cfg.robot.action_space = 'abs_qpos'
    cfg.policy.value = 1.0
    cfg.workflow.action_smoothing = {'enabled': True, 'alpha': .6, 'gripper': False}
    clients, raw = [], []
    opening = np.zeros(14, dtype=np.float32)
    opening[[6, 13]] = -3.14

    def lifecycle(hardware, scheduler, config):
        if not clients:
            clients.append(hardware.robot_client)
        scheduler.execute(chunk(opening))

    def read_state(self):
        return SimpleNamespace(read_state=lambda: SimpleNamespace(qpos=self.robot_client.history[-1].copy()))

    monkeypatch.setattr(HardwareSession, 'robot_driver', property(read_state))
    monkeypatch.setattr(rollout_sync, '_open_grippers_before_policy', lifecycle)
    monkeypatch.setattr(rollout_sync, '_open_grippers_after_policy', lifecycle)
    monkeypatch.setattr(DataSession, 'record_policy_action_chunk', lambda self, value, **kw: raw.append(value.actions.copy()))
    assert rollout_sync.run_from_config(cfg) == 0
    sent = np.asarray(clients[0].history)
    np.testing.assert_array_equal(sent[0], opening)
    np.testing.assert_allclose(sent[1:3, 0], [.4, .64])
    np.testing.assert_array_equal(sent[1:3, [6, 13]], 1)
    np.testing.assert_array_equal(sent[-1], opening)
    assert len(raw) == 2
    np.testing.assert_array_equal(raw, 1)
