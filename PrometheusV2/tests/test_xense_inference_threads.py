from types import SimpleNamespace

import numpy as np
import pytest

import prometheus.nodes.sensors.xense as xense
from prometheus.nodes.sensors.xense import _publish_tactile_image, create_xense_sensor, normalize_resize


def test_difference_image_is_published_at_half_resolution(monkeypatch):
    class Bridge:
        image = None
        encoding = None

        def cv2_to_imgmsg(self, image, *, encoding):
            self.image = image.copy()
            self.encoding = encoding
            return SimpleNamespace(header=SimpleNamespace(stamp=None, frame_id=""))

    class Publisher:
        message = None

        def publish(self, message):
            self.message = message

    bridge = Bridge()
    publisher = Publisher()
    stamp = SimpleNamespace(sec=1, nanosec=2)

    def resize(image, resize_config):
        assert xense._target_size(image.shape[1], image.shape[0], resize_config) == (350, 200)
        return image[::2, ::2]

    monkeypatch.setattr(xense, "_resize_image", resize)

    _publish_tactile_image(
        bridge,
        publisher,
        "left_wrist_0",
        "difference",
        np.zeros((400, 700, 3), dtype=np.uint8),
        stamp,
        normalize_resize(0.5),
    )

    assert bridge.image.shape == (200, 350, 3)
    assert bridge.encoding == "rgb8"
    assert publisher.message.header.stamp is stamp
    assert publisher.message.header.frame_id == "left_wrist_0_difference_frame"


def test_thread_limit_preserves_providers_and_restores_constructor(monkeypatch):
    import onnxruntime as ort

    calls = []

    def original_init(self, model, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(ort.InferenceSession, "__init__", original_init)
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_options = [{"device_id": 0}, {}]
    sensor = SimpleNamespace()

    def create(serial):
        assert serial == "test_serial"
        for _ in range(2):
            ort.InferenceSession(b"model", options, providers, provider_options)
        return sensor

    result, count = create_xense_sensor(SimpleNamespace(create=create), "test_serial", inference_threads=1)
    assert result is sensor
    assert count == 2
    assert ort.InferenceSession.__init__ is original_init
    for call in calls:
        assert call["providers"] is providers
        assert call["provider_options"] is provider_options
        assert call["sess_options"] is options
        assert options.intra_op_num_threads == 1
        assert options.inter_op_num_threads == 1
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
        assert options.get_session_config_entry("session.inter_op.allow_spinning") == "0"
        assert options.enable_mem_pattern is False


def test_constructor_restored_on_sdk_failure():
    import onnxruntime as ort

    original = ort.InferenceSession.__init__

    def create(serial):
        raise RuntimeError("camera unavailable")

    with pytest.raises(RuntimeError, match="camera unavailable"):
        create_xense_sensor(SimpleNamespace(create=create), "test_serial", inference_threads=1)
    assert ort.InferenceSession.__init__ is original


def test_sdk_bypassing_onnx_fails_and_releases_sensor():
    released = []
    sensor = SimpleNamespace(release=lambda: released.append(True))
    with pytest.raises(RuntimeError, match="thread limit was not applied"):
        create_xense_sensor(SimpleNamespace(create=lambda _: sensor), "test_serial", inference_threads=1)
    assert released == [True]
