from types import SimpleNamespace

import numpy as np
import pytest

from prometheus.policy.scheduler import ActionChunk, ActionScheduler
from prometheus.workflows.rollout_sync import _return_to_shutdown_pose


def test_fm_config_has_fourteen_limits():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(Path('prometheus/configs').resolve()), version_base=None):
        cfg = compose(config_name='rollout_fm_sync')
        limits = OmegaConf.to_container(cfg.scheduler.max_velocity, resolve=True)
    assert limits == ([cfg.user.joint_max_velocity] * 6 + [cfg.user.gripper_max_velocity]) * 2
    assert cfg.workflow.return_to_shutdown_pose


class FakeClient:
    action_space = 'abs_qpos'

    def __init__(self):
        self.now = 0.0
        self.sent = []

    def send_action(self, action, **kwargs):
        self.sent.append((self.now, action.copy()))
        self.now += 0.003  # Slow transport must never cause a catch-up burst.
        return {'accepted': True}


def test_policy_offset_is_applied_once_before_interpolation(monkeypatch):
    from prometheus.nodes.robots import a5
    from prometheus.workflows.rollout_sync import _prepare_policy_chunk

    monkeypatch.setattr(a5, 'GIPPER_OFFSET', 0.1)
    hardware = SimpleNamespace(robot={'driver': {'_target_': 'a5'}}, robot_driver=a5.A5DragDriver())
    raw = np.zeros((1, 14), dtype=np.float32)
    raw[:, [6, 13]] = [-1, -1.5]
    original = ActionChunk(raw, action_space='abs_qpos', hz=5, metadata={'done': True})
    adjusted = _prepare_policy_chunk(hardware, original)
    client = FakeClient()
    scheduler = ActionScheduler(client, max_velocity=[1.0] * 14)
    scheduler._spin_sleep = lambda duration: None
    scheduler.seed_current_action(raw[0])
    scheduler.execute(adjusted)
    np.testing.assert_allclose(client.sent[-1][1][[6, 13]], [-0.9, -1.4])
    np.testing.assert_array_equal(original.actions, raw)
    assert adjusted.metadata == original.metadata

    # Explicit lifecycle targets bypass the policy transform.
    opened = raw.copy()
    opened[:, [6, 13]] = -3.14
    scheduler.execute(ActionChunk(opened, action_space='abs_qpos', hz=5))
    np.testing.assert_allclose(client.sent[-1][1][[6, 13]], [-3.14, -3.14])


def test_interpolation_limits_first_target_and_reversals():
    client = FakeClient()
    scheduler = ActionScheduler(client, max_velocity=[0.1, 0.02], interpolation_hz=50)
    scheduler._spin_sleep = lambda duration: setattr(client, 'now', client.now + duration)
    initial = np.array([0.4, 0.1])
    scheduler.seed_current_action(initial)
    chunk = ActionChunk([[0.5, 0.12], [0.3, 0.08]], action_space='abs_qpos', hz=5)
    count = scheduler.execute(chunk)
    times = np.array([0.0] + [t for t, _ in client.sent])
    positions = np.vstack([initial] + [q for _, q in client.sent])
    velocity = np.abs(np.diff(positions, axis=0)) / np.diff(times)[:, None]
    assert np.all(velocity <= np.array([0.1, 0.02]) + 1e-5)
    np.testing.assert_allclose(positions[-1], chunk.actions[-1])
    assert count == scheduler.actions_processed == len(client.sent)
    assert times[-1] >= 3.0


def test_limits_require_feedback_before_any_command():
    client = FakeClient()
    scheduler = ActionScheduler(client, max_velocity=[0.1, 0.1])
    with pytest.raises(ValueError, match='feedback'):
        scheduler.execute(ActionChunk([[1, 1]], action_space='abs_qpos', hz=5))
    assert not client.sent


def test_dry_run_interpolates_without_sending(capsys):
    client = FakeClient()
    scheduler = ActionScheduler(client, dry_run=True, max_velocity=[0.1, 0.1])
    scheduler._spin_sleep = lambda duration: None
    scheduler.seed_current_action([0, 0])
    assert scheduler.execute(ActionChunk([[0.01, 0.01]], action_space='abs_qpos', hz=5)) >= 10
    assert not client.sent
    assert 'dry_run action=' in capsys.readouterr().out


@pytest.mark.parametrize('status', ['completed', 'interrupted'])
def test_return_stops_owner_before_moving(status):
    events = []
    hardware = SimpleNamespace(
        robot_owner=SimpleNamespace(stop=lambda: events.append('stop_owner')),
        robot_driver=SimpleNamespace(is_connected=True, move_to_shutdown_safe_position=lambda: events.append('return')),
    )
    result = _return_to_shutdown_pose(hardware, {'return_to_shutdown_pose': True}, SimpleNamespace(dry_run=False), status)
    assert result['status'] == 'completed'
    assert events == ['stop_owner', 'return']


@pytest.mark.parametrize('dry_run,status', [(True, 'completed'), (True, 'interrupted'), (False, 'error')])
def test_return_skips_dry_run_and_faults(dry_run, status):
    result = _return_to_shutdown_pose(None, {'return_to_shutdown_pose': True}, SimpleNamespace(dry_run=dry_run), status)
    assert result['status'] == 'skipped'


def test_return_failure_is_reported():
    def fail():
        raise TimeoutError('did not settle')
    hardware = SimpleNamespace(
        robot_owner=SimpleNamespace(stop=lambda: None),
        robot_driver=SimpleNamespace(is_connected=True, move_to_shutdown_safe_position=fail),
    )
    result = _return_to_shutdown_pose(hardware, {'return_to_shutdown_pose': True}, SimpleNamespace(dry_run=False), 'interrupted')
    assert result['status'] == 'error'
    assert 'did not settle' in result['error']


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_opens_only_grippers_and_waits_for_feedback(monkeypatch, phase):
    from prometheus.workflows import rollout_sync
    client = FakeClient()
    scheduler = ActionScheduler(client, max_velocity=[0.5] * 14, interpolation_hz=30)
    scheduler._spin_sleep = lambda duration: setattr(client, 'now', client.now + duration)
    monkeypatch.setattr(rollout_sync.time, 'monotonic', lambda: client.now)
    initial = np.arange(14, dtype=np.float32) * 0.01
    initial[[6, 13]] = 0
    hardware = SimpleNamespace(robot_driver=SimpleNamespace(
        read_state=lambda: SimpleNamespace(qpos=client.sent[-1][1] if client.sent else initial)))
    getattr(rollout_sync, f'_open_grippers_{phase}_policy')(hardware, scheduler, {
        f'open_grippers_{phase}_policy': {'enabled': True, 'positions': [-3.14, -3.14]}})
    arm_indices = [i for i in range(14) if i not in (6, 13)]
    for _, action in client.sent:
        np.testing.assert_allclose(action[arm_indices], initial[arm_indices])
    np.testing.assert_allclose(client.sent[-1][1][[6, 13]], [-3.14, -3.14])
    assert client.now - client.sent[-1][0] >= 0.5


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_gripper_dry_run_never_accesses_hardware(phase):
    from prometheus.workflows import rollout_sync
    getattr(rollout_sync, f'_open_grippers_{phase}_policy')(None, SimpleNamespace(dry_run=True), {
        f'open_grippers_{phase}_policy': {'enabled': True, 'positions': [-3.14, -3.14]}})


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_gripper_timeout_aborts(monkeypatch, phase):
    from prometheus.workflows import rollout_sync
    client = FakeClient()
    scheduler = ActionScheduler(client, max_velocity=[0.5] * 14)
    scheduler._spin_sleep = lambda duration: setattr(client, 'now', client.now + duration)
    monkeypatch.setattr(rollout_sync.time, 'monotonic', lambda: client.now)
    hardware = SimpleNamespace(robot_driver=SimpleNamespace(read_state=lambda: SimpleNamespace(qpos=np.zeros(14))))
    with pytest.raises(TimeoutError, match='did not reach'):
        getattr(rollout_sync, f'_open_grippers_{phase}_policy')(hardware, scheduler, {
            f'open_grippers_{phase}_policy': {'enabled': True, 'positions': [-1, -1], 'timeout_s': 0.1}})


@pytest.mark.parametrize('actual,accepted', [([-3.10025215, -2.99725342], True), ([-3.10, -2.0], False)])
def test_exit_gripper_tolerance_accepts_observed_offset_but_not_blocked_gripper(monkeypatch, actual, accepted):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from prometheus.workflows import rollout_sync

    with initialize_config_dir(config_dir=str(Path('prometheus/configs').resolve()), version_base=None):
        cfg = compose(config_name='rollout_fm_sync')
        workflow = OmegaConf.to_container(cfg.workflow, resolve=True)
    client = FakeClient()
    scheduler = ActionScheduler(client, max_velocity=[1.0] * 14)
    scheduler._spin_sleep = lambda duration: setattr(client, 'now', client.now + duration)
    monkeypatch.setattr(rollout_sync.time, 'monotonic', lambda: client.now)
    feedback = np.zeros(14, dtype=np.float32)
    feedback[[6, 13]] = actual
    hardware = SimpleNamespace(robot_driver=SimpleNamespace(read_state=lambda: SimpleNamespace(qpos=feedback)))
    if accepted:
        rollout_sync._open_grippers_after_policy(hardware, scheduler, workflow)
        assert client.now - client.sent[-1][0] >= workflow['open_grippers_after_policy']['settle_s']
    else:
        with pytest.raises(TimeoutError, match='did not reach'):
            rollout_sync._open_grippers_after_policy(hardware, scheduler, workflow)
