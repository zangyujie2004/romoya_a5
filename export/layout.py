from __future__ import annotations

import numpy as np

# Bundle order matches meta.json key order; zarr/hdf5 merge follows the same order.
CAMERA_BUNDLE_ORDER = ("base_0", "left_wrist_0", "right_wrist_0")
CAMERA_STREAM = "rgb"
# Compact bypass for dual impainting (base_0 remove-hand only).
CAMERA_BASE_REMOVE_HAND_KEY = "camera_base_remove_hand"
CAMERA_REMOVE_HAND_STREAM = "rgb_remove_hand"
BASE_REMOVE_HAND_PRESENT = "present"
BASE_REMOVE_HAND_NONE = "none"
META_EPISODE_ENDS_REMOVE_HAND = "episode_ends_remove_hand"

TACTILE_BUNDLE_ORDER = ("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")
TACTILE_STREAM = "pointcloud"

ROBOT_ARM_ORDER = ("left", "right")
# Storage merge: group by field, left then right within each field.
ROBOT_STATE_KEYS = ("joint", "eef", "effort", "vel")

MERGE_30HZ_KEYS = ("camera", "tactile", "state_30hz", "action_30hz")
EVENT_STORAGE_KEY = "event"


def storage_keys_30hz(*, include_event: bool = False) -> tuple[str, ...]:
    if include_event:
        return MERGE_30HZ_KEYS + (EVENT_STORAGE_KEY,)
    return MERGE_30HZ_KEYS
MERGE_120HZ_KEYS = ("state_120hz", "action_120hz")


def _concat_feature_shapes(shapes: list[tuple[int, ...]], axis: int = -1) -> tuple[int, ...]:
    if not shapes:
        return ()
    ref = shapes[0]
    if axis < 0:
        axis = len(ref) + axis
    total = sum(s[axis] for s in shapes)
    return ref[:axis] + (total,) + ref[axis + 1 :]


def merged_camera_shape(camera: dict) -> tuple[int, ...]:
    shapes = []
    for bundle in CAMERA_BUNDLE_ORDER:
        arr = np.asarray(camera[bundle][CAMERA_STREAM])
        shapes.append(tuple(int(x) for x in arr.shape[1:]))
    return _concat_feature_shapes(shapes, axis=-1)


def merged_tactile_shape(tactile: dict) -> tuple[int, ...]:
    shapes = []
    for bundle in TACTILE_BUNDLE_ORDER:
        arr = np.asarray(tactile[bundle][TACTILE_STREAM])
        shapes.append(tuple(int(x) for x in arr.shape[1:]))
    return _concat_feature_shapes(shapes, axis=-1)


def merged_robot_shape(state: dict, *, history: bool = False) -> tuple[int, ...]:
    field_shapes = []
    for key in ROBOT_STATE_KEYS:
        arm_shapes = []
        for arm in ROBOT_ARM_ORDER:
            arr = np.asarray(state[arm][key])
            feat = tuple(int(x) for x in arr.shape[1:])
            arm_shapes.append(feat)
        field_shapes.append(_concat_feature_shapes(arm_shapes, axis=-1))
    merged = _concat_feature_shapes(field_shapes, axis=-1)
    if history and len(merged) == 2:
        return (merged[-1],)
    return merged
