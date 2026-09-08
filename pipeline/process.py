from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline.align import align_timestamps
from task.streams import RGB_REMOVE_HAND_BUNDLE, RGB_REMOVE_HAND_STREAM

_INTERPOLATION = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


def quat_wxyz_to_rot6d(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, np.nan, norm)
    q = q / norm
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    r20 = 2 * (x * z - y * w)
    r01 = 2 * (x * y - z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + x * w)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)


def build_eef9p1(joint7, eef7):
    joint7 = np.asarray(joint7, dtype=np.float64)
    eef7 = np.asarray(eef7, dtype=np.float64)
    xyz = eef7[..., :3]
    rot6d = quat_wxyz_to_rot6d(eef7[..., 3:7])
    gripper = joint7[..., 6:7]
    return np.concatenate([xyz, rot6d, gripper], axis=-1)


def shift_next_along_time(x):
    x = np.asarray(x)
    if len(x) <= 1:
        return x.copy()
    return np.concatenate([x[1:], x[-1:]], axis=0)


def build_arm_state(state_raw):
    return {
        "joint": np.asarray(state_raw["joint"], dtype=np.float64),
        "vel": np.asarray(state_raw["vel"], dtype=np.float64),
        "effort": np.asarray(state_raw["effort"], dtype=np.float64),
        "eef": build_eef9p1(state_raw["joint"], state_raw["eef"]),
    }


def build_arm_action(arm_state):
    return {key: shift_next_along_time(arr) for key, arr in arm_state.items()}


def _camera_cfg(process_cfg: dict[str, Any]) -> dict[str, Any]:
    camera_cfg = process_cfg.get("camera", {})
    interpolation = str(camera_cfg.get("interpolation", "area")).lower()
    return {
        "resize": bool(camera_cfg.get("resize", True)),
        "resize_hw": tuple(camera_cfg.get("resize_hw", [224, 224])),
        "resize_streams": set(camera_cfg.get("streams", ["rgb"])),
        "interpolation": _INTERPOLATION.get(interpolation, cv2.INTER_AREA),
    }


def _tactile_cfg(process_cfg: dict[str, Any]) -> tuple[int, int, int]:
    shape = process_cfg.get("tactile", {}).get("pointcloud_shape", [35, 20, 6])
    return tuple(shape)


def _robot_cfg(process_cfg: dict[str, Any]) -> dict[str, Any]:
    robot_cfg = process_cfg.get("robot", {})
    return {"history_steps": int(robot_cfg.get("history_steps", 4))}


def process_episode_bundles(
    camera_dict,
    tactile_dict,
    robot_dict,
    event_dict,
    synced_timestamps: dict,
    timeline: dict,
    process_cfg: dict[str, Any] | None = None,
) -> dict:
    process_cfg = process_cfg or {}
    camera_opts = _camera_cfg(process_cfg)
    pointcloud_shape = _tactile_cfg(process_cfg)
    robot_opts = _robot_cfg(process_cfg)

    master_timeline = timeline["master_timeline"]
    master_sel_idx = timeline["master_sel_idx"]
    master_bundle = timeline["master_bundle"]

    aligned_camera = {}
    for bundle, payload in camera_dict.items():
        aligned = {}
        if bundle == master_bundle:
            index = master_sel_idx
        else:
            source_ts = synced_timestamps["camera"][bundle]
            _, index = align_timestamps(master_timeline, source_ts)
        for key, arr in payload.items():
            if key == "timestamps":
                continue
            arr = np.asarray(arr)
            if len(arr) == len(payload["timestamps"]):
                arr = arr[index]
                resize_key = key == RGB_REMOVE_HAND_STREAM or key in camera_opts["resize_streams"]
                if camera_opts["resize"] and resize_key:
                    arr = np.stack(
                        [
                            cv2.resize(img, camera_opts["resize_hw"], interpolation=camera_opts["interpolation"])
                            for img in arr
                        ]
                    )
            elif key == RGB_REMOVE_HAND_STREAM:
                raise ValueError(
                    f"camera/{bundle}/{RGB_REMOVE_HAND_STREAM} length {len(arr)} "
                    f"!= timestamps {len(payload['timestamps'])}"
                )
            aligned[key] = arr
        if (
            bundle == RGB_REMOVE_HAND_BUNDLE
            and RGB_REMOVE_HAND_STREAM in aligned
            and "rgb" in aligned
        ):
            rgb = np.asarray(aligned["rgb"])
            remove = np.asarray(aligned[RGB_REMOVE_HAND_STREAM])
            if remove.shape[0] != rgb.shape[0]:
                raise ValueError(
                    f"camera/{bundle}/{RGB_REMOVE_HAND_STREAM} T={remove.shape[0]} "
                    f"!= rgb T={rgb.shape[0]} after align"
                )
            if remove.shape[1:] != rgb.shape[1:]:
                raise ValueError(
                    f"camera/{bundle}/{RGB_REMOVE_HAND_STREAM} shape {remove.shape[1:]} "
                    f"!= rgb shape {rgb.shape[1:]} after resize"
                )
        aligned_camera[bundle] = aligned

    aligned_tactile = {}
    for bundle, payload in tactile_dict.items():
        aligned = {}
        source_ts = synced_timestamps["tactile"][bundle]
        _, index = align_timestamps(master_timeline, source_ts)
        for key, arr in payload.items():
            if key == "timestamps":
                continue
            arr = np.asarray(arr)
            if len(arr) == len(payload["timestamps"]):
                arr = arr[index]
                if key == "pointcloud":
                    arr = arr.reshape(arr.shape[0], *pointcloud_shape)
            aligned[key] = arr
        aligned_tactile[bundle] = aligned

    aligned_event = {}
    for bundle, payload in event_dict.items():
        aligned = {}
        source_ts = synced_timestamps["event"][bundle]
        _, index = align_timestamps(master_timeline, source_ts)
        for key, arr in payload.items():
            if key == "timestamps":
                continue
            arr = np.asarray(arr)
            if len(arr) == len(payload["timestamps"]):
                arr = arr[index]
            aligned[key] = arr
        aligned_event[bundle] = aligned

    history_steps = robot_opts["history_steps"]
    state_30hz = {}
    action_30hz = {}
    state_120hz = {}
    action_120hz = {}
    for bundle, payload in robot_dict.items():
        source_ts = synced_timestamps["robot"][bundle]
        _, idx_30hz = align_timestamps(master_timeline, source_ts)
        offsets = np.arange(history_steps - 1, -1, -1, dtype=np.int64)
        history_idx = np.stack(
            [np.clip(idx_30hz - off, 0, len(source_ts) - 1) for off in offsets],
            axis=1,
        )
        state_raw_30 = {}
        state_raw_120 = {}
        for key, arr in payload.items():
            if key == "timestamps":
                continue
            arr = np.asarray(arr)
            if len(arr) != len(payload["timestamps"]):
                continue
            state_raw_30[key] = arr[idx_30hz]
            state_raw_120[key] = arr[history_idx]
        arm_state_30 = build_arm_state(state_raw_30)
        arm_action_30 = build_arm_action(arm_state_30)
        arm_state_120 = build_arm_state(state_raw_120)
        arm_action_120 = build_arm_action(arm_state_120)
        state_30hz[bundle] = arm_state_30
        action_30hz[bundle] = arm_action_30
        state_120hz[bundle] = arm_state_120
        action_120hz[bundle] = arm_action_120

    return {
        "timestamps": master_timeline,
        "camera": aligned_camera,
        "tactile": aligned_tactile,
        "event": aligned_event,
        "state_30hz": state_30hz,
        "action_30hz": action_30hz,
        "state_120hz": state_120hz,
        "action_120hz": action_120hz,
    }
