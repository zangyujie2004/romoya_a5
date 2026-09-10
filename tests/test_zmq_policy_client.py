from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from prometheus.workflows.zmq_policy_client import ZmqPolicyData


def test_zmq_policy_client_config_defaults_to_fm() -> None:
    root = Path("prometheus/configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name="zmq_policy_client", overrides=["user.steps=2"])
    assert cfg.policy._target_ == "prometheus.policy.fm_policy.FMPolicy"
    assert cfg.client.max_steps == 2
    assert cfg.client.obs_keys == "auto"


def test_zmq_policy_data_window_numpy_returns_history_frames() -> None:
    info = {
        "obs_history": [
            {"base_0_color": np.zeros((2, 2, 3), dtype=np.uint8)},
            {"base_0_color": np.ones((2, 2, 3), dtype=np.uint8)},
        ],
        "obs_history_timestamps": [
            {"base_0_color": 10},
            {"base_0_color": 20},
        ],
        "obs_history_frame_timestamps": [10, 20],
    }
    data = ZmqPolicyData(obs={"base_0_color": np.ones((2, 2, 3), dtype=np.uint8)}, info=info)

    frames = data.window_numpy(anchor="base_0_color", names=("base_0_color",), count=2, slop_ms=60.0)

    assert [frame.stamp_ns for frame in frames] == [10, 20]
    assert frames[1].samples["base_0_color"].stamp_ns == 20
    np.testing.assert_array_equal(frames[1].samples["base_0_color"].data, np.ones((2, 2, 3), dtype=np.uint8))
