import numpy as np
import pytest

from prometheus.policy.scheduler import ActionChunk, ActionScheduler
from prometheus.policy.trajectory import plan_acceleration_limited


def test_trajectory_bounds_continuity_and_no_overshoot():
    rng = np.random.default_rng(5)
    targets = np.cumsum(rng.uniform(-.1, .1, (30, 14)), axis=0)
    targets[:, 3] = 0  # a stationary joint must remain stationary
    vmax = np.full(14, .4)
    amax = np.full(14, .8)
    amax[[6, 13]] = np.inf
    segments = plan_acceleration_limited(np.zeros(14), targets, vmax, amax, 1 / 30)
    previous, previous_velocity = np.zeros(14), np.zeros(14)
    joints = np.isfinite(amax)
    phase = np.linspace(0, 1, 1001)[:, None]
    for segment in segments:
        a, b, c, d = segment.coefficients
        position = segment.position(phase)
        velocity = (3 * a * phase**2 + 2 * b * phase + c) / segment.duration
        acceleration = (6 * a * phase + 2 * b) / segment.duration**2
        assert np.all(np.abs(velocity) <= vmax + 1e-10)
        assert np.all(np.abs(acceleration[:, joints]) <= amax[joints] + 1e-10)
        assert np.all(position >= np.minimum(previous, segment.target) - 1e-10)
        assert np.all(position <= np.maximum(previous, segment.target) + 1e-10)
        np.testing.assert_allclose(velocity[0, joints], previous_velocity[joints], atol=1e-10)
        np.testing.assert_allclose(position[0], previous, atol=1e-10)
        np.testing.assert_allclose(position[-1], segment.target, atol=1e-10)
        previous, previous_velocity = segment.target, velocity[-1]
    np.testing.assert_allclose(previous_velocity[joints], 0, atol=1e-10)
    # Continuous travel through an interior waypoint; no stop at every action.
    straight = plan_acceleration_limited(np.zeros(1), np.array([[.1], [.2], [.3]]),
                                         np.array([1.]), np.array([2.]), .1)
    assert straight[1].coefficients[2, 0] > 0


class Client:
    action_space = 'abs_qpos'

    def __init__(self, latency=0):
        self.time = 0.
        self.latency = latency
        self.sent = []

    def send_action(self, action, **kwargs):
        self.sent.append((self.time, action.copy()))
        self.time += self.latency
        return {'accepted': True}


def scheduler(client, acceleration):
    obj = ActionScheduler(client, max_velocity=[.4] * 14, max_acceleration=acceleration, interpolation_hz=60)
    obj._spin_sleep = lambda duration: setattr(client, 'time', client.time + duration)
    obj.seed_current_action(np.zeros(14))
    return obj


def test_sent_targets_respect_caps_across_chunks_and_reversals():
    client = Client()
    obj = scheduler(client, [.8] * 6 + [None] + [.8] * 6 + [None])
    for values in [[.04, .08], [-.02, -.06]]:
        obj.execute(ActionChunk(np.repeat(np.array(values)[:, None], 14, axis=1), action_space='abs_qpos', hz=30))
    times = np.array([0.] + [item[0] for item in client.sent])
    positions = np.vstack([np.zeros(14)] + [item[1] for item in client.sent])
    intervals = np.diff(times)
    velocity = np.diff(positions, axis=0) / intervals[:, None]
    velocity_times = (times[1:] + times[:-1]) / 2
    acceleration = np.diff(velocity, axis=0) / np.diff(velocity_times)[:, None]
    joints = [i for i in range(14) if i not in (6, 13)]
    assert np.max(np.abs(velocity)) <= .4001
    assert np.max(np.abs(acceleration[:, joints])) <= .8005
    np.testing.assert_allclose(positions[-1], -.06)


def test_gripper_only_motion_keeps_velocity_only_timing():
    target = np.zeros(14)
    target[[6, 13]] = -.4
    chunk = ActionChunk([target], action_space='abs_qpos', hz=30)
    outputs = []
    for caps in [None, [.8] * 6 + [None] + [.8] * 6 + [None], [None] * 14]:
        client = Client()
        obj = scheduler(client, caps)
        obj.execute(chunk)
        outputs.append(np.array([[stamp, *value] for stamp, value in client.sent]))
    for output in outputs[1:]:
        np.testing.assert_allclose(output, outputs[0], atol=1e-7)


def test_transport_delay_does_not_trigger_catchup_bursts():
    client = Client(latency=.04)
    obj = scheduler(client, [.8] * 14)
    obj.execute(ActionChunk([np.full(14, .04)], action_space='abs_qpos', hz=30))
    assert np.all(np.diff([stamp for stamp, _ in client.sent]) > .04)


@pytest.mark.parametrize('velocity,acceleration', [(None, [1.]), ([1.], [0.]), ([1.], [-1.]),
                                                  ([1.], [float('nan')]), ([1.], [float('inf')]),
                                                  ([1.], [1., 1.])])
def test_invalid_limits_fail_before_sending(velocity, acceleration):
    with pytest.raises(ValueError, match='max_acceleration'):
        ActionScheduler(Client(), max_velocity=velocity, max_acceleration=acceleration)


def test_fm_config_limits_arm_axes_only():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path('prometheus/configs').resolve()), version_base=None):
        cfg = compose(config_name='rollout_fm_sync')
        assert OmegaConf.to_container(cfg.scheduler.max_acceleration, resolve=True) == (
            [cfg.user.joint_max_acceleration] * 6 + [None]) * 2
        cfg = compose(config_name='rollout_fm_sync', overrides=['user.joint_max_acceleration=null'])
        assert OmegaConf.to_container(cfg.scheduler.max_acceleration, resolve=True) == [None] * 14
