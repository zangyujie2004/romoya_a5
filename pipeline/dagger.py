from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.process import build_arm_action
from task.streams import ROBOT_BUNDLES

DAGGER_WORKFLOW = "rollout_dagger"

def dagger_enabled(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("dagger") or {}).get("enabled", False))


def resolve_gap_threshold_ms(cfg: dict[str, Any]) -> float | None:
    if not dagger_enabled(cfg):
        return None
    dagger_cfg = cfg.get("dagger") or {}
    return float(dagger_cfg["recording_threshold"])


def read_episode_workflow(episode_path: Path) -> str | None:
    manifests_dir = Path(episode_path) / "manifests"
    if not manifests_dir.is_dir():
        return None
    for manifest_path in sorted(manifests_dir.glob("*.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for record in payload.values():
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            workflow = metadata.get("workflow")
            if workflow:
                return str(workflow)
    return None


def should_apply_event_filter(episode_path: Path, cfg: dict[str, Any]) -> bool:
    if not dagger_enabled(cfg):
        return False
    workflow = read_episode_workflow(episode_path)
    return workflow == DAGGER_WORKFLOW


def _event_mask(data_dict: dict[str, Any], keep_events: list[int]) -> np.ndarray:
    event_group = data_dict.get("event") or {}
    event_bundle = event_group.get("event")
    if not isinstance(event_bundle, dict) or "data" not in event_bundle:
        raise ValueError("dagger event filter requires aligned event.event.data")

    event_data = np.asarray(event_bundle["data"])
    keep = {int(value) for value in keep_events}
    return np.array([int(value) in keep for value in event_data], dtype=bool)


def _filter_group_dict(group: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for bundle, streams in group.items():
        filtered[bundle] = {}
        for key, value in streams.items():
            arr = np.asarray(value)
            if arr.shape[0] == mask.shape[0]:
                filtered[bundle][key] = arr[mask]
            else:
                filtered[bundle][key] = arr
    return filtered


def _recompute_robot_actions(data_dict: dict[str, Any]) -> None:
    for rate_key in ("30hz", "120hz"):
        state_key = f"state_{rate_key}"
        action_key = f"action_{rate_key}"
        state_group = data_dict.get(state_key)
        if not isinstance(state_group, dict):
            continue
        action_group: dict[str, Any] = {}
        for arm in ROBOT_BUNDLES:
            if arm not in state_group:
                continue
            action_group[arm] = build_arm_action(state_group[arm])
        data_dict[action_key] = action_group


def apply_dagger_postprocess(
    data_dict: dict[str, Any],
    cfg: dict[str, Any],
    *,
    apply_event_filter: bool,
) -> dict[str, Any]:
    if not dagger_enabled(cfg):
        return data_dict

    dagger_cfg = cfg.get("dagger") or {}
    if not apply_event_filter:
        return data_dict

    keep_events = [int(value) for value in dagger_cfg.get("keep_events", [0, 1])]
    mask = _event_mask(data_dict, keep_events)
    if not np.any(mask):
        return {
            "timestamps": np.asarray([], dtype=np.int64),
            "camera": {},
            "tactile": {},
            "event": {},
            "state_30hz": {},
            "action_30hz": {},
            "state_120hz": {},
            "action_120hz": {},
        }

    filtered = {
        "timestamps": np.asarray(data_dict["timestamps"], dtype=np.int64)[mask],
        "camera": _filter_group_dict(data_dict.get("camera", {}), mask),
        "tactile": _filter_group_dict(data_dict.get("tactile", {}), mask),
        "event": _filter_group_dict(data_dict.get("event", {}), mask),
        "state_30hz": _filter_group_dict(data_dict.get("state_30hz", {}), mask),
        "action_30hz": _filter_group_dict(data_dict.get("action_30hz", {}), mask),
        "state_120hz": _filter_group_dict(data_dict.get("state_120hz", {}), mask),
        "action_120hz": _filter_group_dict(data_dict.get("action_120hz", {}), mask),
    }
    _recompute_robot_actions(filtered)
    return filtered


def episode_frame_count(data_dict: dict[str, Any]) -> int:
    timestamps = data_dict.get("timestamps")
    if timestamps is None:
        return 0
    return int(np.asarray(timestamps).shape[0])


def min_frames_required(cfg: dict[str, Any]) -> int:
    if not dagger_enabled(cfg):
        return 1
    return int((cfg.get("dagger") or {}).get("min_frames", 1))
