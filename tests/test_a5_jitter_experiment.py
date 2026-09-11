import numpy as np

from scripts.a5_jitter_experiment import Monitor
from scripts.a5_sensor_load import load_config, sensor_contract, receiver


def test_diagnostic_camera_selection_removes_tactile_streams(tmp_path):
    original = load_config('prometheus/configs/robot/a5_drag.yaml', 'all')
    camera_only = load_config('prometheus/configs/robot/a5_drag.yaml', 'cameras')
    sensors, streams = sensor_contract(camera_only)
    assert set(sensors) == {'realsense'}
    assert not any('tactile' in name for name in streams)
    assert len(original['tactile']['serial_no']) == 4


def test_receiver_has_complete_bridge_configuration(monkeypatch):
    from prometheus.nodes.data_bridge import RosDataBridge
    calls = []
    monkeypatch.setattr(RosDataBridge, 'start', lambda self: calls.append(self.name))
    monkeypatch.setattr(RosDataBridge, 'close', lambda self: None)
    data = receiver(load_config('prometheus/configs/robot/a5_drag.yaml', 'cameras'))
    try:
        data.start()
        assert calls == ['jitter_sensor_receiver']
    finally:
        data.close()


def test_monitor_separates_excursion_from_single_step_jump(tmp_path):
    monitor = Monitor(None, tmp_path)
    try:
        for i in range(10):
            q = np.zeros(14)
            q[5] = .01*i
            q[12] = 1.5 if i >= 5 else 0
            monitor.rows.append({'qpos':q.tolist(),'monotonic':i*.01})
        result = monitor.summary()
        assert np.isclose(result['range_rad'][5], .09)
        assert result['steps_over_0_05_rad'][5] == 0
        assert result['steps_over_0_05_rad'][12] == 1
        assert result['max_step_rad'][12] == 1.5
    finally:
        monitor.close()
