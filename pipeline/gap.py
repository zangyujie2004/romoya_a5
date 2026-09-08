from __future__ import annotations

import numpy as np


def downsample_fixed_fps(timestamp, target_fps: float):
    timestamp = np.asarray(timestamp)
    start_time = timestamp[0]
    end_time = timestamp[-1]
    step = 1000 / target_fps

    num_out = int(np.floor((end_time - start_time) / step)) + 1
    target_times = start_time + np.arange(num_out) * step

    n = len(timestamp)
    idx = np.searchsorted(timestamp, target_times, side="left")
    idx_r = np.clip(idx, 0, n - 1)
    idx_l = np.clip(idx - 1, 0, n - 1)
    d_r = np.abs(timestamp[idx_r] - target_times)
    d_l = np.abs(timestamp[idx_l] - target_times)
    sel_idx = np.where(d_l <= d_r, idx_l, idx_r)
    return timestamp[sel_idx], sel_idx


def find_recording_gaps(
    timestamps_ms: np.ndarray,
    *,
    gap_threshold_ms: float,
) -> list[tuple[int, int]]:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64).reshape(-1)
    if timestamps_ms.size < 2:
        return []
    gaps: list[tuple[int, int]] = []
    for index in range(int(timestamps_ms.size) - 1):
        start = int(timestamps_ms[index])
        end = int(timestamps_ms[index + 1])
        if end - start > float(gap_threshold_ms):
            gaps.append((start, end))
    return gaps


def split_segment_ranges(length: int, timestamps_ms: np.ndarray, *, gap_threshold_ms: float) -> list[tuple[int, int]]:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64).reshape(-1)
    if length <= 0:
        return []
    if timestamps_ms.size < 2:
        return [(0, length)]

    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(int(timestamps_ms.size) - 1):
        if int(timestamps_ms[index + 1]) - int(timestamps_ms[index]) > float(gap_threshold_ms):
            segments.append((start, index + 1))
            start = index + 1
    segments.append((start, length))
    return [(seg_start, seg_end) for seg_start, seg_end in segments if seg_end > seg_start]


def downsample_fixed_fps_gap_aware(
    timestamps_ms: np.ndarray,
    *,
    target_fps: float,
    gap_threshold_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    if timestamps_ms.size == 0:
        raise ValueError("downsample_fixed_fps_gap_aware: timestamps_ms is empty")

    timeline_parts: list[np.ndarray] = []
    sel_idx_parts: list[np.ndarray] = []
    for seg_start, seg_end in split_segment_ranges(len(timestamps_ms), timestamps_ms, gap_threshold_ms=gap_threshold_ms):
        segment = timestamps_ms[seg_start:seg_end]
        if segment.size == 0:
            continue
        timeline, rel_idx = downsample_fixed_fps(segment, target_fps=target_fps)
        timeline_parts.append(np.asarray(timeline, dtype=np.int64))
        sel_idx_parts.append(np.asarray(rel_idx, dtype=np.int64) + int(seg_start))

    if not timeline_parts:
        raise ValueError("downsample_fixed_fps_gap_aware: no valid segments after gap split")

    return np.concatenate(timeline_parts), np.concatenate(sel_idx_parts)
