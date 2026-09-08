from __future__ import annotations

import numpy as np

from export.layout import (
    BASE_REMOVE_HAND_NONE,
    BASE_REMOVE_HAND_PRESENT,
    CAMERA_BASE_REMOVE_HAND_KEY,
    CAMERA_BUNDLE_ORDER,
    CAMERA_REMOVE_HAND_STREAM,
    CAMERA_STREAM,
    EVENT_STORAGE_KEY,
    MERGE_120HZ_KEYS,
    MERGE_30HZ_KEYS,
    ROBOT_ARM_ORDER,
    ROBOT_STATE_KEYS,
    TACTILE_BUNDLE_ORDER,
    TACTILE_STREAM,
)


def _concat_on_axis(arrays: list[np.ndarray], axis: int = -1) -> np.ndarray:
    return np.concatenate([np.asarray(a) for a in arrays], axis=axis)


def merge_camera(camera: dict) -> np.ndarray:
    rgbs = []
    ref = None
    for bundle in CAMERA_BUNDLE_ORDER:
        if bundle not in camera or CAMERA_STREAM not in camera[bundle]:
            raise KeyError(f"missing camera/{bundle}/{CAMERA_STREAM}")
        rgb = np.asarray(camera[bundle][CAMERA_STREAM])
        if ref is None:
            ref = rgb
        elif rgb.shape[0] != ref.shape[0]:
            raise ValueError(f"camera length mismatch: {bundle} {rgb.shape[0]} vs {ref.shape[0]}")
        rgbs.append(rgb)
    return _concat_on_axis(rgbs, axis=-1)


def extract_camera_base_remove_hand(camera: dict) -> np.ndarray | None:
    """Return base_0 remove-hand RGB if present; else None.

    Does not invent data. Caller decides whether to write the compact bypass.
    """
    base = camera.get(CAMERA_BUNDLE_ORDER[0]) or {}
    if CAMERA_REMOVE_HAND_STREAM not in base:
        return None
    remove = np.asarray(base[CAMERA_REMOVE_HAND_STREAM])
    if CAMERA_STREAM not in base:
        raise KeyError(f"missing camera/{CAMERA_BUNDLE_ORDER[0]}/{CAMERA_STREAM}")
    rgb = np.asarray(base[CAMERA_STREAM])
    if remove.shape[0] != rgb.shape[0]:
        raise ValueError(
            f"{CAMERA_BASE_REMOVE_HAND_KEY}: T={remove.shape[0]} != rgb T={rgb.shape[0]}"
        )
    if remove.shape[1:] != rgb.shape[1:]:
        raise ValueError(
            f"{CAMERA_BASE_REMOVE_HAND_KEY}: feature {remove.shape[1:]} != rgb {rgb.shape[1:]}"
        )
    return remove


def base_remove_hand_status(camera: dict | None) -> str:
    if not camera:
        return BASE_REMOVE_HAND_NONE
    return (
        BASE_REMOVE_HAND_PRESENT
        if extract_camera_base_remove_hand(camera) is not None
        else BASE_REMOVE_HAND_NONE
    )


def merge_tactile(tactile: dict) -> np.ndarray:
    pcs = []
    ref = None
    for bundle in TACTILE_BUNDLE_ORDER:
        if bundle not in tactile or TACTILE_STREAM not in tactile[bundle]:
            raise KeyError(f"missing tactile/{bundle}/{TACTILE_STREAM}")
        pc = np.asarray(tactile[bundle][TACTILE_STREAM])
        if ref is None:
            ref = pc
        elif pc.shape[0] != ref.shape[0]:
            raise ValueError(f"tactile length mismatch: {bundle} {pc.shape[0]} vs {ref.shape[0]}")
        pcs.append(pc)
    return _concat_on_axis(pcs, axis=-1)


def merge_robot(group: dict, *, history: bool = False) -> np.ndarray:
    field_vectors = []
    for key in ROBOT_STATE_KEYS:
        parts = []
        for arm in ROBOT_ARM_ORDER:
            if arm not in group:
                raise KeyError(f"missing robot arm '{arm}'")
            if key not in group[arm]:
                raise KeyError(f"missing {arm}/{key}")
            parts.append(np.asarray(group[arm][key], dtype=np.float32))
        field_vectors.append(_concat_on_axis(parts, axis=-1))

    merged = _concat_on_axis(field_vectors, axis=-1)
    if history and merged.ndim == 3 and merged.shape[1] == 4:
        merged = merged.reshape(-1, merged.shape[-1])
    return merged


def merge_event(event: dict) -> np.ndarray:
    if "event" not in event or "data" not in event["event"]:
        raise KeyError("missing event/event/data")
    return np.asarray(event["event"]["data"], dtype=np.int64)


def episode_to_storage_arrays(data_dict: dict, *, include_event: bool = False) -> dict[str, np.ndarray]:
    """Merge episode dict into compact storage keys (same layout as meta.storage.data)."""
    camera = data_dict.get("camera")
    tactile = data_dict.get("tactile")
    if not camera or not tactile:
        raise ValueError("camera and tactile are required for storage export")

    arrays = {
        "camera": merge_camera(camera),
        "tactile": merge_tactile(tactile),
        "state_30hz": merge_robot(data_dict.get("state_30hz", {}), history=False),
        "action_30hz": merge_robot(data_dict.get("action_30hz", {}), history=False),
        "state_120hz": merge_robot(data_dict.get("state_120hz", {}), history=True),
        "action_120hz": merge_robot(data_dict.get("action_120hz", {}), history=True),
    }
    event = data_dict.get("event")
    if include_event:
        if not event:
            raise ValueError("include_event=True but event payload is missing")
        arrays[EVENT_STORAGE_KEY] = merge_event(event)
    return arrays


def storage_keys_30hz(*, include_event: bool = False) -> tuple[str, ...]:
    from export.layout import storage_keys_30hz as _layout_keys

    return _layout_keys(include_event=include_event)


def storage_keys_120hz() -> tuple[str, ...]:
    return MERGE_120HZ_KEYS
