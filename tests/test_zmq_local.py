from __future__ import annotations

import importlib.abc
import json
import multiprocessing as mp
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from prometheus.transport.local import Node, endpoint
from prometheus.transport.messages import (
    Header, Image, ImageBridge, JointState, PointCloud2, PointField, String,
    create_cloud, pack, unpack,
)


class _NoRos(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rclpy', 'sensor_msgs', 'sensor_msgs_py', 'std_msgs', 'cv_bridge', 'geometry_msgs'}:
            raise AssertionError(f'Unexpected ROS import: {fullname}')


@pytest.fixture(autouse=True)
def isolated_namespace(monkeypatch):
    monkeypatch.setenv('PROMETHEUS_ZMQ_NAMESPACE', 'test-' + uuid.uuid4().hex)
    guard = _NoRos()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def _producer(stop):
    sys.meta_path.insert(0, _NoRos())
    node = Node('synthetic_sensors', domain_id=22)
    pubs = {name: node.create_publisher(cls, '/' + name, 10)
            for name, cls in [('base_0_color', Image), ('left_wrist_0_tactile_difference', Image),
                              ('left_wrist_0_tactile_flow', PointCloud2)]}
    bridge = ImageBridge()
    fields = [PointField(name=name, offset=i * 4) for i, name in enumerate(('x', 'y', 'z', 'dx', 'dy', 'dz'))]
    try:
        while not stop.is_set():
            header = Header(stamp=node.get_clock().now().to_msg())
            for name, shape in [('base_0_color', (240, 320, 3)),
                                ('left_wrist_0_tactile_difference', (200, 350, 3))]:
                array = np.zeros(shape, np.uint8)
                array[..., 0] = 17
                array[..., 2] = 91
                msg = bridge.cv2_to_imgmsg(array, 'bgr8')
                msg.header = header
                pubs[name].publish(msg)
            pubs['left_wrist_0_tactile_flow'].publish(create_cloud(header, fields, np.ones((700, 6), np.float32)))
            stop.wait(1 / 30)
    finally:
        node.destroy_node()


class _Driver:
    def __init__(self):
        self.actions = []
        self.modes = []
        self.sides = ('left', 'right')
        self.robot_id = 'fake'
        self.eef_topics = {'left': '/left_eef', 'right': '/right_eef'}
        self.eef_pubs = {}

    def setup_transport(self, **kwargs):
        from prometheus.nodes.robots.a5 import A5DragDriver
        A5DragDriver.setup_transport(self, **kwargs)

    def publish_transport_state(self, **kwargs):
        from prometheus.nodes.robots.a5 import A5DragDriver
        A5DragDriver.publish_transport_state(self, **kwargs)

    def connect(self):
        pass

    def read_state(self):
        return SimpleNamespace(joint_names=[f'joint_{i}' for i in range(14)], qpos=np.zeros(14),
                               qvel=None, effort=None, eef_wxyz=np.array([.1, .2, .3, 1., 0., 0., 0.] * 2))

    def send_action(self, space, action):
        self.actions.append(action.tolist())

    def set_teach_mode(self):
        self.modes.append('teach')

    def reset_home(self):
        self.modes.append('reset')

    def status(self):
        return {}

    def close(self):
        pass


def _owner_process(stop, report):
    sys.meta_path.insert(0, _NoRos())
    from prometheus.nodes.robots.owner import RobotOwnerNode
    driver = _Driver()
    owner = RobotOwnerNode(driver=driver, robot_id='fake', ros_domain_id=21, hz=100,
                           topics={'state': '/state', 'control': '/control', 'result': '/result', 'status': '/status'},
                           action_dims={'joint': 14})
    try:
        owner.start()
        stop.wait(15)
    finally:
        owner.close()
        report.put({'actions': driver.actions, 'modes': driver.modes, 'error': owner.status()['error']})


def _client(domain=21):
    from prometheus.nodes.robots.client import RobotControlClient
    return RobotControlClient(action_space='joint', action_dim=14, control_topic='/control',
                              result_topic='/result', ros_domain_id=domain, retry_interval_s=0.03)


def _join(process, stop):
    stop.set()
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(3)
        pytest.fail('ZMQ child did not shut down')
    assert process.exitcode == 0


def test_cross_process_commands_ack_retry_dedup_expiry_and_isolation():
    ctx = mp.get_context('spawn')
    stop, report = ctx.Event(), ctx.Queue()
    process = ctx.Process(target=_owner_process, args=(stop, report))
    process.start()
    try:
        with _client() as client:
            client.wait_for_owner(5)
            assert client.set_teach_mode()['accepted']
            assert client.send_action(np.ones(14))['accepted']
            assert client.reset_home()['accepted']
            # Same request id and payload must be acknowledged but never actuated twice.
            msg = String(json.dumps({'request_id': 'dedup', 'client_stamp_ns': time.time_ns(),
                                     'command': 'set_teach_mode', 'command_source': client.command_source}))
            for _ in range(2):
                client.request_id, client.result = 'dedup', None
                client.pub.publish(msg)
                deadline = time.monotonic() + 2
                while client.result is None and time.monotonic() < deadline:
                    client.spin_once(.02)
                assert client.result['accepted']
            expired = json.loads(msg.data)
            expired.update(request_id='expired', expires_ns=time.time_ns()-1)
            client.request_id, client.result = 'expired', None
            client.pub.publish(String(json.dumps(expired)))
            deadline = time.monotonic() + 2
            while client.result is None and time.monotonic() < deadline:
                client.spin_once(.02)
            assert not client.result['accepted']
            assert 'expired' in client.result['message']
            with _client(domain=22) as wrong:
                with pytest.raises(TimeoutError):
                    wrong.wait_for_owner(.15)
    finally:
        _join(process, stop)
    result = report.get(timeout=2)
    assert result['modes'] == ['teach', 'reset', 'teach']
    assert len(result['actions']) == 1
    assert not result['error']


def test_cross_process_images_flow_numpy_and_video_recording(tmp_path):
    from prometheus.sessions.data import DataSession
    streams = {name: {'topic': '/' + name, 'msg_type': 'prometheus.transport.messages.' + cls,
                      'domain': 'sensor', 'ros_domain_id': 22}
               for name, cls in [('base_0_color', 'Image'), ('left_wrist_0_tactile_difference', 'Image'),
                                ('left_wrist_0_tactile_flow', 'PointCloud2')]}
    data = DataSession(streams, history=32,
                       bridge={'enabled': True, 'name': 'test_ingress', 'wait_all_streams': True, 'qos_depth': 10},
                       numpy={'enabled': True, 'history': 32},
                       recording={'enabled': True, 'output_dir': str(tmp_path), 'queue_size': 128,
                                  'fps': 30., 'default_event_value': 0, 'enable_plots': False})
    ctx = mp.get_context('spawn')
    stop = ctx.Event()
    process = ctx.Process(target=_producer, args=(stop,))
    process.start()
    try:
        data.start()
        data.wait_ready(5)
        data.start_episode('0000')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if all(value >= 10 for value in data.numpy_counts().values()) and len(data.numpy_counts()) == 3:
                break
            time.sleep(.02)
        samples = data.latest_numpy(list(streams))
        image = samples['base_0_color'].data
        assert image.shape == (240, 320, 3)
        assert image[0, 0].tolist() == [91, 0, 17]  # BGR on wire -> RGB model input
        assert samples['left_wrist_0_tactile_difference'].data.shape == (200, 350, 3)
        np.testing.assert_array_equal(samples['left_wrist_0_tactile_flow'].data, np.ones((700, 6)))
        assert all(sample.stamp_ns > 0 for sample in samples.values())
        data.stop_episode()
    finally:
        data.close()
        _join(process, stop)
    videos = list((tmp_path / '0000').rglob('*.mkv')) + list((tmp_path / '0000').rglob('*.mp4'))
    assert len(videos) >= 2
    shapes = []
    for path in videos:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                '-show_entries', 'stream=width,height', '-of', 'json', str(path)],
                               check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)['streams'][0]
        frame = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1',
                                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True)
        assert len(frame.stdout) == stream['height'] * stream['width'] * 3
        shapes.append((stream['height'], stream['width'], 3))
    assert (200, 350, 3) in shapes
    with (tmp_path / '0000/tactile/tactile_pointcloud_dict.pkl').open('rb') as file:
        flow = pickle.load(file)
    assert len(flow['left_wrist_0']['data']) > 0
    np.testing.assert_array_equal(flow['left_wrist_0']['data'][0], np.ones((700, 6)))


def test_subscriber_reconnects_when_publisher_restarts():
    receiver = Node('receiver', domain_id=22)
    received = []
    receiver.create_subscription(String, '/restart', lambda msg: received.append(msg.data), 2)
    try:
        for value in ['before', 'after']:
            producer = Node('producer', domain_id=22)
            try:
                pub = producer.create_publisher(String, '/restart', 2)
                deadline = time.monotonic() + 3
                while value not in received and time.monotonic() < deadline:
                    pub.publish(String(value))
                    receiver.spin_once(.02)
                assert value in received
            finally:
                producer.destroy_node()
    finally:
        receiver.destroy_node()


class _SyntheticSensors:
    def start(self):
        ctx = mp.get_context('spawn')
        self.stop_event = ctx.Event()
        self.process = ctx.Process(target=_producer, args=(self.stop_event,))
        self.process.start()

    def stop(self):
        if hasattr(self, 'process'):
            _join(self.process, self.stop_event)
            del self.process

    def close(self):
        self.stop()


def _resolve_fake_robot(**kwargs):
    streams = {name: {'topic': '/' + name, 'msg_type': 'prometheus.transport.messages.' + cls,
                      'domain': 'sensor', 'ros_domain_id': 22}
               for name, cls in [('base_0_color', 'Image'), ('left_wrist_0_tactile_difference', 'Image'),
                                ('left_wrist_0_tactile_flow', 'PointCloud2')]}
    streams['robot_state'] = {'topic': '/state', 'msg_type': 'prometheus.transport.messages.JointState',
                              'domain': 'robot', 'ros_domain_id': 21}
    for side in ('left', 'right'):
        streams[f'{side}_eef'] = {'topic': f'/{side}_eef', 'msg_type': 'prometheus.transport.messages.PoseStamped',
                                  'domain': 'robot', 'ros_domain_id': 21}
    return {
        'driver': {'_target_': 'tests.test_zmq_local._Driver'},
        'owner': {'_target_': 'prometheus.nodes.robots.owner.RobotOwnerNode', 'robot_id': 'fake',
                  'ros_domain_id': 21, 'hz': 100, 'action_dims': {'joint': 14},
                  'topics': {'state': '/state', 'control': '/control', 'result': '/result', 'status': '/status'}},
        'client': {'_target_': 'prometheus.nodes.robots.client.RobotControlClient', 'action_space': 'joint',
                   'action_dim': 14, 'ros_domain_id': 21, 'control_topic': '/control', 'result_topic': '/result'},
        'sensors': {'synthetic': {'_target_': 'tests.test_zmq_local._SyntheticSensors'}},
        'streams': streams,
    }


from tests.noop import NoopPolicy


class _ObservingPolicy(NoopPolicy):
    anchor = 'base_0_color'
    stream_names = ('robot_state', 'left_eef', 'right_eef', 'base_0_color',
                    'left_wrist_0_tactile_difference', 'left_wrist_0_tactile_flow')
    window_size = 1

    def infer(self, data):
        frame = data.window_numpy(anchor=self.anchor, names=self.stream_names, count=1, slop_ms=100)[-1]
        assert frame.samples['robot_state'].data['position'].shape == (14,)
        assert frame.samples['base_0_color'].data.shape == (240, 320, 3)
        assert frame.samples['left_wrist_0_tactile_flow'].data.shape == (700, 6)
        np.testing.assert_allclose(frame.samples['left_eef'].data, [.1, .2, .3, 1., 0., 0., 0.])
        return super().infer(data)


def test_rollout_workflow_with_real_zmq_and_synthetic_hardware(tmp_path):
    from omegaconf import OmegaConf
    from prometheus.workflows.rollout_sync import run_from_config
    cfg = OmegaConf.load('tests/configs/rollout_sync_noop.yaml')
    cfg.run.output_dir = str(tmp_path)
    cfg.run.runtime_dir = str(tmp_path / 'runtime')
    cfg.run.wait_ready_timeout_s = 5
    cfg.robot = {'_target_': 'tests.test_zmq_local._resolve_fake_robot', 'action_space': 'joint', 'action_dim': 14}
    cfg.hardware.sensors = 'auto'
    cfg.hardware.streams = 'auto'
    cfg.policy._target_ = 'tests.test_zmq_local._ObservingPolicy'
    cfg.data.numpy = {'enabled': True, 'history': 32}
    assert run_from_config(cfg) == 0
    manifest = json.loads((tmp_path / 'workflow_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['actions_sent'] == 3


def test_duplicate_bind_rejected_and_endpoint_reusable():
    first, second = Node('first'), Node('second')
    try:
        first.create_publisher(String, '/only-one', 2)
        with pytest.raises(RuntimeError, match='already owned'):
            second.create_publisher(String, '/only-one', 2)
        first.destroy_node()
        second.create_publisher(String, '/only-one', 2)
    finally:
        first.destroy_node()
        second.destroy_node()


def test_wire_timestamp_image_roundtrip_and_namespace():
    msg = ImageBridge().cv2_to_imgmsg(np.arange(24, dtype=np.uint8).reshape(2, 4, 3), 'rgb8')
    msg.header.frame_id = 'tactile'
    restored = unpack(pack(msg))
    assert restored == msg
    assert endpoint(21, '/state', namespace='one') != endpoint(21, '/state', namespace='two')
    with pytest.raises(TypeError):
        pack(object())


def test_fm_raw_tactile_adapter_needs_no_ros():
    from prometheus.policy.fm_policy import _ensure_flow_matching_importable
    _ensure_flow_matching_importable()
    from infer.preprocess import pointcloud2_to_numpy
    fields = [PointField(name=name, offset=i * 4) for i, name in enumerate(('x', 'y', 'z', 'dx', 'dy', 'dz'))]
    values = np.arange(4200, dtype=np.float32).reshape(700, 6)
    message = unpack(pack(create_cloud(Header(), fields, values)))
    np.testing.assert_array_equal(pointcloud2_to_numpy(message), values)
