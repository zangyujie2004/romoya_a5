from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_FLOW_MATCHING_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "flow_matching"
if str(_FLOW_MATCHING_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOW_MATCHING_ROOT))

from infer.config import default_action_process, parse_deploy_config
from infer.postprocess import apply_action_process
from infer.preprocess import build_obs_from_frames, parse_preprocess_config
from infer.types import DEFAULT_JOINT_NAMES, PreprocessConfig


class _IdentityNormalizer:
    def normalize_state_np(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)


def test_default_action_process_for_joint():
    cfg = {"data": {"action_type": "joint"}}
    assert default_action_process(cfg) == "abs_qpos"
    deploy = parse_deploy_config(cfg)
    assert deploy.action_process == "abs_qpos"


def test_build_obs_from_frames_joint_state_shape():
    cfg = PreprocessConfig(action_type="joint")
    frames = [_joint_frame(idx) for idx in range(8)]
    obs, state_raw = build_obs_from_frames(
        frames,
        cfg,
        _IdentityNormalizer(),
        window_size=8,
    )
    assert state_raw.shape == (8, 14)
    assert obs["state"].shape == (8, 14)
    assert obs["image"].shape == (1, 1, 3, 3, 224, 224)


def test_apply_action_process_abs_qpos_pass_through():
    traj = np.arange(32 * 14, dtype=np.float32).reshape(32, 14)
    out = apply_action_process(traj, "abs_qpos")
    np.testing.assert_array_equal(out, traj)


def test_parse_preprocess_config_reads_joint_action_type():
    cfg = {"data": {"action_type": "joint", "image_size": 224}}
    parsed = parse_preprocess_config(cfg)
    assert parsed.action_type == "joint"
    assert parsed.state_dim == 14
    assert parsed.joint_names == DEFAULT_JOINT_NAMES


def _joint_frame(idx: int) -> Any:
    positions = np.arange(14, dtype=np.float32) + float(idx)
    return SimpleNamespace(
        samples={
            "robot_state": SimpleNamespace(
                msg=SimpleNamespace(
                    name=list(DEFAULT_JOINT_NAMES),
                    position=positions.tolist(),
                )
            ),
            "base_0_color": _image_msg(np.zeros((4, 4, 3), dtype=np.uint8), "rgb8"),
            "left_wrist_0_color": _image_msg(np.zeros((4, 4, 3), dtype=np.uint8), "rgb8"),
            "right_wrist_0_color": _image_msg(np.zeros((4, 4, 3), dtype=np.uint8), "rgb8"),
        }
    )


def _image_msg(array: np.ndarray, encoding: str) -> Any:
    array = np.ascontiguousarray(array)
    return SimpleNamespace(
        msg=SimpleNamespace(
            height=array.shape[0],
            width=array.shape[1],
            encoding=encoding,
            step=array.strides[0],
            data=array.tobytes(),
        )
    )
