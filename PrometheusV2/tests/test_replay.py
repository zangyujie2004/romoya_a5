from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from prometheus.utils.robot import quat_wxyz_to_rpy, rpy_to_quaternion_wxyz
from prometheus.workflows.replay import (
    build_replay_actions,
    downsample_to_hz,
    filter_valid_indices,
    load_replay_chunk,
    load_robot_state,
    resolve_robot_state_path,
    run_from_config,
)


def _make_robot_state(*, num_frames: int = 125, step_ms: int = 8, start_ms: int = 1_000) -> dict:
    timestamps = [start_ms + index * step_ms for index in range(num_frames)]
    left_joints = []
    right_joints = []
    left_eefs = []
    right_eefs = []
    for index in range(num_frames):
        left_joint = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07 + index * 0.001], dtype=np.float32)
        right_joint = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.08 + index * 0.001], dtype=np.float32)
        left_quat = rpy_to_quaternion_wxyz(0.1 + index * 0.001, -0.2, 0.3)
        right_quat = rpy_to_quaternion_wxyz(-0.1, 0.2 + index * 0.001, -0.3)
        left_eef = np.concatenate([np.array([0.24, 0.01, 0.15], dtype=np.float32), left_quat]).astype(np.float32)
        right_eef = np.concatenate([np.array([0.23, 0.02, 0.14], dtype=np.float32), right_quat]).astype(np.float32)
        left_joints.append(left_joint)
        right_joints.append(right_joint)
        left_eefs.append(left_eef)
        right_eefs.append(right_eef)
    return {
        "left": {
            "joint": left_joints,
            "eef": left_eefs,
            "vel": left_joints,
            "effort": left_joints,
            "timestamps": timestamps,
        },
        "right": {
            "joint": right_joints,
            "eef": right_eefs,
            "vel": right_joints,
            "effort": right_joints,
            "timestamps": timestamps,
        },
    }


def _write_robot_state(tmp_path: Path, robot_state: dict) -> Path:
    episode_dir = tmp_path / "0000"
    robot_dir = episode_dir / "robot"
    robot_dir.mkdir(parents=True)
    path = robot_dir / "robot_state_dict.pkl"
    with path.open("wb") as file:
        pickle.dump(robot_state, file)
    return episode_dir


def test_quat_wxyz_to_rpy_roundtrip():
    roll, pitch, yaw = 0.2, -0.3, 0.5
    quat = rpy_to_quaternion_wxyz(roll, pitch, yaw)
    recovered = quat_wxyz_to_rpy(quat)
    assert recovered.shape == (3,)
    assert recovered[0] == pytest.approx(roll, abs=1e-4)
    assert recovered[1] == pytest.approx(pitch, abs=1e-4)
    assert recovered[2] == pytest.approx(yaw, abs=1e-4)


def test_downsample_to_hz_matches_fixed_fps():
    robot_state = _make_robot_state(num_frames=125, step_ms=8, start_ms=1_000)
    indices = downsample_to_hz(robot_state["left"]["timestamps"], 30.0)
    assert indices.shape == (30,)
    assert int(indices[0]) == 0
    assert np.all(np.diff(indices) >= 0)


def test_build_replay_actions_joint_and_eef(tmp_path):
    robot_state = _make_robot_state(num_frames=10, step_ms=33, start_ms=10_000)
    indices = np.array([0, 5, 9], dtype=np.int64)

    joint_actions = build_replay_actions(robot_state, indices, mode="joint")
    assert joint_actions.shape == (3, 14)
    np.testing.assert_allclose(joint_actions[0, :7], robot_state["left"]["joint"][0])
    np.testing.assert_allclose(joint_actions[0, 7:], robot_state["right"]["joint"][0])

    eef_actions = build_replay_actions(robot_state, indices, mode="eef")
    assert eef_actions.shape == (3, 14)
    assert eef_actions[0, 6] == pytest.approx(robot_state["left"]["joint"][0][6])
    assert eef_actions[0, 13] == pytest.approx(robot_state["right"]["joint"][0][6])
    np.testing.assert_allclose(eef_actions[0, :3], robot_state["left"]["eef"][0][:3])
    np.testing.assert_allclose(
        eef_actions[0, 3:6],
        quat_wxyz_to_rpy(robot_state["left"]["eef"][0][3:7]),
    )


def test_filter_valid_indices_skips_invalid_eef(tmp_path):
    robot_state = _make_robot_state(num_frames=5, step_ms=33, start_ms=10_000)
    robot_state["left"]["eef"][2] = None
    indices = np.arange(5, dtype=np.int64)
    kept, skipped = filter_valid_indices(robot_state, indices, mode="eef")
    assert skipped == 1
    assert 2 not in kept


def test_load_replay_chunk_from_episode_dir(tmp_path):
    episode_dir = _write_robot_state(tmp_path, _make_robot_state())
    chunk, metadata = load_replay_chunk(episode_dir, mode="joint", hz=30.0)
    assert chunk.action_space == "abs_qpos"
    assert chunk.hz == pytest.approx(30.0)
    assert chunk.actions.shape == (30, 14)
    assert metadata["replay_frames"] == 30
    assert metadata["skipped_frames"] == 0


def test_resolve_robot_state_path_accepts_pickle_directly(tmp_path):
    episode_dir = _write_robot_state(tmp_path, _make_robot_state(num_frames=3, step_ms=100, start_ms=0))
    pkl_path = episode_dir / "robot" / "robot_state_dict.pkl"
    assert resolve_robot_state_path(pkl_path) == pkl_path
    assert resolve_robot_state_path(episode_dir) == pkl_path
    loaded = load_robot_state(pkl_path)
    assert len(loaded["left"]["timestamps"]) == 3


def test_replay_noop_workflow(tmp_path):
    episode_dir = _write_robot_state(tmp_path, _make_robot_state(num_frames=31, step_ms=33, start_ms=0))
    cfg = OmegaConf.load("tests/configs/replay_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.workflow.input = str(episode_dir)

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["actions_sent"] == 30
    assert manifest["replay"]["replay_frames"] == 30
    assert manifest["robot_client"]["sent_actions"] == 30


@pytest.mark.skipif(
    not Path("/mnt/oss_data/arx1/pen/pen_0705_50/0000/robot/robot_state_dict.pkl").exists(),
    reason="sample replay data is unavailable",
)
def test_load_pen_sample_replay_chunk():
    chunk, metadata = load_replay_chunk(
        "/mnt/oss_data/arx1/pen/pen_0705_50/0000",
        mode="eef",
        hz=30.0,
    )
    assert chunk.action_space == "abs_eef"
    assert chunk.actions.shape[1] == 14
    assert 1000 <= metadata["replay_frames"] <= 1030
    assert metadata["skipped_frames"] >= 0
