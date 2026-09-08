from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from export.layout import (
    BASE_REMOVE_HAND_NONE,
    BASE_REMOVE_HAND_PRESENT,
    CAMERA_BASE_REMOVE_HAND_KEY,
    CAMERA_BUNDLE_ORDER,
    CAMERA_REMOVE_HAND_STREAM,
    CAMERA_STREAM,
    MERGE_120HZ_KEYS,
    MERGE_30HZ_KEYS,
    META_EPISODE_ENDS_REMOVE_HAND,
    ROBOT_ARM_ORDER,
    ROBOT_STATE_KEYS,
    TACTILE_BUNDLE_ORDER,
    TACTILE_STREAM,
    merged_camera_shape,
    merged_robot_shape,
    merged_tactile_shape,
)

EPISODE_KEYS = (
    "camera",
    "tactile",
    "event",
    "state_30hz",
    "action_30hz",
    "state_120hz",
    "action_120hz",
)

META_SCHEMA = "preprocess_meta/v1"


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "()"
    return "(" + ",".join(str(x) for x in shape) + ")"


def _field(shape: tuple[int, ...], dtype: np.dtype | str, **extra: str) -> str:
    parts = [f"shape:{_format_shape(shape)}", f"dtype:{dtype}"]
    parts.extend(f"{key}:{value}" for key, value in extra.items())
    return ",".join(parts)


def _feature_shape(arr: np.ndarray) -> tuple[int, ...]:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return ()
    if arr.ndim == 1:
        return (int(arr.shape[0]),)
    return tuple(int(x) for x in arr.shape[1:])


def _feature_field(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    return _field(_feature_shape(arr), arr.dtype)


def _source_path(group: str, bundle: str, stream: str) -> str:
    return f"{group}/{bundle}/{stream}"


def _robot_source_order(prefix: str) -> str:
    paths = []
    for key in ROBOT_STATE_KEYS:
        for arm in ROBOT_ARM_ORDER:
            paths.append(f"{prefix}/{arm}/{key}")
    return ",".join(paths)


def _robot_storage_layout(group: dict) -> str:
    """Dim ranges per field and arm for splitting merged state/action vectors."""
    offset = 0
    parts = []
    for key in ROBOT_STATE_KEYS:
        left_dim = int(np.asarray(group["left"][key]).shape[-1])
        right_dim = int(np.asarray(group["right"][key]).shape[-1])
        left_end = offset + left_dim
        right_end = left_end + right_dim
        parts.append(f"{key}:left={offset}-{left_end},right={left_end}-{right_end}")
        offset = right_end
    return ";".join(parts)


def _build_episode_section(data_dict: dict) -> dict[str, Any]:
    episode: dict[str, Any] = {}
    for group in EPISODE_KEYS:
        payload = data_dict.get(group)
        if not payload:
            continue
        group_meta: dict[str, Any] = {}
        for bundle, streams in payload.items():
            bundle_meta = {
                key: _feature_field(np.asarray(value))
                for key, value in streams.items()
                if key != "timestamps"
            }
            if bundle_meta:
                group_meta[bundle] = bundle_meta
        if group_meta:
            episode[group] = group_meta
    return episode


def _build_storage_section(
    data_dict: dict,
    *,
    has_camera_base_remove_hand: bool = False,
) -> dict[str, str]:
    """Describe replay_buffer array layout for downstream readers."""
    keys: dict[str, str] = {}

    camera = data_dict.get("camera")
    if camera:
        feat = merged_camera_shape(camera)
        dtype = np.asarray(camera[CAMERA_BUNDLE_ORDER[0]][CAMERA_STREAM]).dtype
        order = ",".join(_source_path("camera", b, CAMERA_STREAM) for b in CAMERA_BUNDLE_ORDER)
        keys["camera"] = _field(
            ("T", *feat),
            dtype,
            order=order,
            merge="concat@-1",
        )
        if has_camera_base_remove_hand:
            base = camera.get(CAMERA_BUNDLE_ORDER[0]) or {}
            if CAMERA_REMOVE_HAND_STREAM in base:
                rh = np.asarray(base[CAMERA_REMOVE_HAND_STREAM])
                rh_feat = tuple(int(x) for x in rh.shape[1:])
                rh_dtype = rh.dtype
            else:
                # Schema-only: same HWC as base rgb when sample ep has none.
                rgb = np.asarray(camera[CAMERA_BUNDLE_ORDER[0]][CAMERA_STREAM])
                rh_feat = tuple(int(x) for x in rgb.shape[1:])
                rh_dtype = rgb.dtype
            keys[CAMERA_BASE_REMOVE_HAND_KEY] = _field(
                ("T_rh", *rh_feat),
                rh_dtype,
                source=f"camera/{CAMERA_BUNDLE_ORDER[0]}/{CAMERA_REMOVE_HAND_STREAM}",
                merge="compact_present_only",
                desc="base_0 remove-hand RGB; only present episodes concatenated",
            )

    tactile = data_dict.get("tactile")
    if tactile:
        feat = merged_tactile_shape(tactile)
        dtype = np.asarray(tactile[TACTILE_BUNDLE_ORDER[0]][TACTILE_STREAM]).dtype
        order = ",".join(_source_path("tactile", b, TACTILE_STREAM) for b in TACTILE_BUNDLE_ORDER)
        keys["tactile"] = _field(
            ("T", *feat),
            dtype,
            order=order,
            merge="concat@-1",
        )

    for rate_key in ("state_30hz", "action_30hz"):
        group = data_dict.get(rate_key)
        if group:
            feat = merged_robot_shape(group, history=False)
            dtype = np.asarray(group["left"]["joint"]).dtype
            keys[rate_key] = _field(
                ("T", *feat),
                dtype,
                order=_robot_source_order(rate_key),
                layout=_robot_storage_layout(group),
                merge="concat@-1,group:field",
            )

    event = data_dict.get("event")
    if event and "event" in event and "data" in event["event"]:
        event_arr = np.asarray(event["event"]["data"])
        keys["event"] = _field(("T",), event_arr.dtype, source="event/event/data")

    for rate_key in MERGE_120HZ_KEYS:
        group = data_dict.get(rate_key)
        if group:
            feat = merged_robot_shape(group, history=True)
            dtype = np.asarray(group["left"]["joint"]).dtype
            keys[rate_key] = _field(
                ("T*4", *feat),
                dtype,
                order=_robot_source_order(rate_key),
                layout=_robot_storage_layout(group),
                merge="concat@-1,group:field,reshape_history",
                history="4",
            )

    keys["meta/episode_ends"] = "shape:(num_episodes,),dtype:int64,desc:cumulative 30hz frame count"
    keys["meta/episode_ends_4x"] = "shape:(num_episodes,),dtype:int64,desc:cumulative 120hz frame count"
    if has_camera_base_remove_hand:
        keys[f"meta/{META_EPISODE_ENDS_REMOVE_HAND}"] = (
            "shape:(num_present,),dtype:int64,"
            "desc:cumulative camera_base_remove_hand frame count (present eps only)"
        )

    return keys


def build_episode_meta(
    data_dict: dict,
    *,
    cfg: dict | None = None,
    num_episodes: int | None = None,
    base_remove_hand_per_episode: list[str] | None = None,
) -> dict[str, Any]:
    """Build self-describing meta: episode layout + replay_buffer storage layout."""
    if data_dict is None:
        raise ValueError("data_dict is required to build meta")

    align_cfg = (cfg or {}).get("align", {})
    fps = align_cfg.get("target_fps", 30)

    timestamps = np.asarray(data_dict["timestamps"])
    t_len = int(timestamps.shape[0])

    export_cfg = (cfg or {}).get("export", {})
    export_fmt = str(export_cfg.get("format", "zarr")).lower()
    storage_names = {
        "hdf5": ("hdf5_replay_buffer", "replay_buffer.hdf5"),
        "zarr": ("zarr_replay_buffer", "replay_buffer.zarr"),
        "lerobot": ("lerobot_v2.1", "lerobot/"),
    }
    storage_format, storage_path = storage_names.get(export_fmt, storage_names["zarr"])

    flags = list(base_remove_hand_per_episode or [])
    if flags:
        invalid = [f for f in flags if f not in (BASE_REMOVE_HAND_PRESENT, BASE_REMOVE_HAND_NONE)]
        if invalid:
            raise ValueError(f"invalid base_remove_hand flags: {invalid[:5]}")
        if num_episodes is not None and len(flags) != int(num_episodes):
            raise ValueError(
                f"base_remove_hand_per_episode length {len(flags)} != num_episodes {num_episodes}"
            )
    has_rh = any(f == BASE_REMOVE_HAND_PRESENT for f in flags)

    # Episode schema: annotate base_0 remove-hand status field (sample may omit stream).
    episode_section = _build_episode_section(data_dict)
    camera_ep = episode_section.setdefault("camera", {}).setdefault(CAMERA_BUNDLE_ORDER[0], {})
    # Prefer run-level annotation over sample-only stream presence.
    if flags:
        # Schema note: per-episode values live under dataset.base_remove_hand.per_episode.
        camera_ep["base_remove_hand"] = (
            f'enum:{BASE_REMOVE_HAND_PRESENT}|{BASE_REMOVE_HAND_NONE},'
            "desc:per-episode flag in dataset.base_remove_hand.per_episode"
        )

    meta: dict[str, Any] = {
        "schema": META_SCHEMA,
        "convention": "episode.* omits time dim; storage.* uses T as 30hz frame count",
        "time": {
            "timestamps": _field(("T",), timestamps.dtype),
            "fps": fps,
            "sample_T": t_len,
        },
        "episode": episode_section,
        "storage": {
            "format": storage_format,
            "path": storage_path,
            "data": _build_storage_section(
                data_dict, has_camera_base_remove_hand=has_rh
            ),
        },
    }

    dataset: dict[str, Any] = {}
    if num_episodes is not None:
        dataset["num_episodes"] = num_episodes
    if flags:
        present_count = sum(1 for f in flags if f == BASE_REMOVE_HAND_PRESENT)
        none_count = sum(1 for f in flags if f == BASE_REMOVE_HAND_NONE)
        dataset["base_remove_hand"] = {
            "present_count": present_count,
            "none_count": none_count,
            "per_episode": flags,
        }
    if dataset:
        meta["dataset"] = dataset

    return meta


def save_meta(meta: dict[str, Any], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return output_path
