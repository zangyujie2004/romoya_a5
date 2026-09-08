from __future__ import annotations
from typing import Any
import numpy as np
from task.streams import ROBOT_BUNDLES
from pipeline.gap import downsample_fixed_fps, downsample_fixed_fps_gap_aware, find_recording_gaps

def _find_start_end_by_norm(
    xyz,
    timestamps,
    threshold: float = 0.1,
    step: int = 2,
    start_offset: int = 0,
    end_offset: int = 30,
    min_start_idx: int = 10,
):
    norm = np.linalg.norm(np.asarray(xyz)[:, :3], axis=1)
    norm = np.nan_to_num(norm, nan=0.0)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    t_len = len(norm)
    if len(timestamps) != t_len:
        raise ValueError("xyz and timestamps must have the same length.")

    start_id = 0
    for i in range(t_len - step):
        if abs(norm[i + step] - norm[i]) > threshold:
            start_id = i
            break

    end_id = t_len - 1
    for i in range(t_len - 1, step - 1, -1):
        if abs(norm[i] - norm[i - step]) > threshold:
            end_id = i
            break

    start_idx = int(np.max([start_id - start_offset, min_start_idx]))
    end_idx = int(np.min([end_id + end_offset, t_len - 1]))
    return start_id, end_id, start_idx, end_idx, int(timestamps[start_idx]), int(timestamps[end_idx])


def find_task_window_dual(
    robot_dict,
    pos_threshold: float = 0.1,
    step: int = 2,
    start_offset: int = 0,
    end_offset: int = 30,
    min_start_idx: int = 10,
):
    arms = {}
    for arm in ROBOT_BUNDLES:
        start_id, end_id, start_idx, end_idx, start_time, end_time = _find_start_end_by_norm(
            robot_dict[arm]["eef"],
            robot_dict[arm]["timestamps"],
            threshold=pos_threshold,
            step=step,
            start_offset=start_offset,
            end_offset=end_offset,
            min_start_idx=min_start_idx,
        )
        arms[arm] = {
            "start_id": start_id,
            "end_id": end_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": start_time,
            "end_time": end_time,
        }
    task_start = min(a["start_time"] for a in arms.values())
    task_end = max(a["end_time"] for a in arms.values())
    return task_start, task_end


def crop_timeline_by_window(timeline, start_time, end_time):
    timeline = np.asarray(timeline, dtype=np.int64).reshape(-1)
    start_idx = int(np.searchsorted(timeline, start_time, side="left"))
    end_idx = int(np.searchsorted(timeline, end_time, side="right"))
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(timeline))

    if start_idx >= end_idx:
        raise ValueError(
            f"empty task window on master timeline: start={start_time}, end={end_time}, "
            f"timeline=[{timeline[0]}, {timeline[-1]}]"
        )
    return timeline[start_idx:end_idx], start_idx, end_idx


def align_timestamps(reference_ts, source_ts):
    reference_ts = np.asarray(reference_ts, dtype=np.float64).reshape(-1)
    source_ts = np.asarray(source_ts, dtype=np.float64).reshape(-1)

    if len(reference_ts) == 0:
        raise ValueError("align_timestamps: reference_ts is empty")
    if len(source_ts) == 0:
        raise ValueError("align_timestamps: source_ts is empty")

    idx = np.searchsorted(source_ts, reference_ts, side="left")
    idx_r = np.clip(idx, 0, len(source_ts) - 1)
    idx_l = np.clip(idx - 1, 0, len(source_ts) - 1)
    d_r = np.abs(source_ts[idx_r] - reference_ts)
    d_l = np.abs(source_ts[idx_l] - reference_ts)
    source_indices = np.where(d_l <= d_r, idx_l, idx_r).astype(np.int64)
    aligned_source_ts = source_ts[source_indices].astype(np.int64)
    return aligned_source_ts, source_indices


def _task_window_cfg(align_cfg: dict[str, Any]) -> dict[str, Any]:
    task_window = align_cfg.get("task_window", {})
    return {
        "pos_threshold": float(task_window.get("pos_threshold", 0.0008)),
        "step": int(task_window.get("step", 2)),
        "start_offset": int(task_window.get("start_offset", 30)),
        "end_offset": int(task_window.get("end_offset", 60)),
        "min_start_idx": int(task_window.get("min_start_idx", 10)),
    }


def build_master_timeline(
    robot_dict,
    synced_timestamps: dict,
    master_clock: str,
    align_cfg: dict[str, Any] | None = None,
    *,
    gap_threshold_ms: float | None = None,
) -> dict:
    align_cfg = align_cfg or {}
    task_window = _task_window_cfg(align_cfg)
    target_fps = float(align_cfg.get("target_fps", 30))

    task_start_time, task_end_time = find_task_window_dual(robot_dict, **task_window)

    master_modality, master_bundle = master_clock.split("/", 1)
    master_ts = synced_timestamps[master_modality][master_bundle]
    if gap_threshold_ms is not None:
        master_timeline_full, master_sel_idx_full = downsample_fixed_fps_gap_aware(
            master_ts,
            target_fps=target_fps,
            gap_threshold_ms=float(gap_threshold_ms),
        )
    else:
        master_timeline_full, master_sel_idx_full = downsample_fixed_fps(master_ts, target_fps=target_fps)
    master_timeline, master_start_id, master_end_id = crop_timeline_by_window(
        master_timeline_full,
        task_start_time,
        task_end_time,
    )
    master_sel_idx = master_sel_idx_full[master_start_id:master_end_id]
    if align_cfg.get("verbose"):
        print(f"task window: {master_start_id} - {master_end_id} = {master_end_id - master_start_id}")
    timeline = {
        "task_start_time": task_start_time,
        "task_end_time": task_end_time,
        "master_clock": master_clock,
        "master_bundle": master_bundle,
        "master_timeline_full": master_timeline_full,
        "master_sel_idx_full": master_sel_idx_full,
        "master_timeline": master_timeline,
        "master_start_id": master_start_id,
        "master_end_id": master_end_id,
        "master_sel_idx": master_sel_idx,
    }
    if gap_threshold_ms is not None:
        timeline["recording_gaps"] = find_recording_gaps(master_ts, gap_threshold_ms=float(gap_threshold_ms))
    return timeline
