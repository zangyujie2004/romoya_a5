from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from task.streams import ROBOT_BUNDLES, ROBOT_STREAMS

ARM_LABELS = {"left": "left arm", "right": "right arm"}
STATE_COLOR = "#1f77b4"
ACTION_COLOR = "#ff7f0e"
GAP_LINE_COLOR = "#888888"
DEFAULT_GAP_THRESHOLD_MS = 300.0


def _flatten_history(arr: np.ndarray) -> np.ndarray:
    """(T, D) or (T, history, D) -> (T, D), using latest history step."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        return arr[:, -1, :]
    return arr


def _extract_tcp_from_eef(eef_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """eef (T, 10) or (T, 4, 10) -> xyz (T, 3), gripper (T,)"""
    eef_arr = _flatten_history(eef_arr)
    return eef_arr[:, :3], eef_arr[:, -1]


def _field_dim_labels(field: str, dim: int) -> list[str]:
    if field == "eef" and dim == 10:
        return ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"]
    return [f"{field}{i}" for i in range(dim)]


def _nominal_step_ms(timestamps_ms: np.ndarray, *, gap_threshold_ms: float) -> float:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64).reshape(-1)
    if timestamps_ms.size < 2:
        return 1000.0 / 30.0
    dts = np.diff(timestamps_ms)
    normal = dts[dts <= float(gap_threshold_ms)]
    if normal.size == 0:
        return 1000.0 / 30.0
    return float(np.median(normal))


def _connected_time_s(
    timestamps_ms: np.ndarray,
    *,
    gap_threshold_ms: float = DEFAULT_GAP_THRESHOLD_MS,
) -> np.ndarray:
    """Map raw timestamps to a continuous visual timeline with recording gaps removed."""
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    if timestamps_ms.size == 0:
        return np.array([], dtype=np.float64)

    nominal = _nominal_step_ms(timestamps_ms, gap_threshold_ms=gap_threshold_ms)
    offset = 0.0
    t_s = np.zeros(timestamps_ms.size, dtype=np.float64)
    for index, ts in enumerate(timestamps_ms):
        t_s[index] = (ts - timestamps_ms[0] - offset) / 1000.0
        if index + 1 < timestamps_ms.size:
            dt = timestamps_ms[index + 1] - timestamps_ms[index]
            if dt > float(gap_threshold_ms):
                offset += dt - nominal
    return t_s


def _gap_boundary_times_s(
    timestamps_ms: np.ndarray,
    *,
    gap_threshold_ms: float = DEFAULT_GAP_THRESHOLD_MS,
) -> np.ndarray:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64).reshape(-1)
    if timestamps_ms.size < 2:
        return np.array([], dtype=np.float64)

    dts = np.diff(timestamps_ms)
    jump_indices = np.where(dts > float(gap_threshold_ms))[0]
    if jump_indices.size == 0:
        return np.array([], dtype=np.float64)

    t_conn = _connected_time_s(timestamps_ms, gap_threshold_ms=gap_threshold_ms)
    return t_conn[jump_indices + 1]


def _split_ranges_at_gaps(length: int, timestamps_ms: np.ndarray, *, gap_threshold_ms: float) -> list[tuple[int, int]]:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64).reshape(-1)
    if length <= 0:
        return []
    if timestamps_ms.size < 2:
        return [(0, length)]

    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(int(timestamps_ms.size) - 1):
        if int(timestamps_ms[index + 1]) - int(timestamps_ms[index]) > float(gap_threshold_ms):
            if index + 1 > start:
                ranges.append((start, index + 1))
            start = index + 1
    if start < length:
        ranges.append((start, length))
    return ranges


def _draw_gap_markers(ax, gap_times_s: np.ndarray, *, draw_label: bool) -> None:
    for idx, gap_t in enumerate(gap_times_s):
        ax.axvline(
            gap_t,
            color=GAP_LINE_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            label="recording gap" if draw_label and idx == 0 else None,
        )


def _plot_state_action_field(
    *,
    field: str,
    state_dict: dict[str, Any],
    action_dict: dict[str, Any],
    t: np.ndarray,
    timestamps_ms: np.ndarray,
    save_path: Path,
    rate_key: str,
    gap_threshold_ms: float = DEFAULT_GAP_THRESHOLD_MS,
) -> None:
    ref_arm = next(arm for arm in ROBOT_BUNDLES if arm in state_dict and field in state_dict[arm])
    dim = _flatten_history(state_dict[ref_arm][field]).shape[-1]
    dim_labels = _field_dim_labels(field, dim)
    n_rows = len(dim_labels)
    gap_times_s = _gap_boundary_times_s(timestamps_ms, gap_threshold_ms=gap_threshold_ms)

    fig_h = max(2.2 * n_rows, 6)
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, fig_h), sharex=True)
    if n_rows == 1:
        axs = np.asarray([axs])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.06, hspace=0.32, wspace=0.18)

    for col, arm in enumerate(ROBOT_BUNDLES):
        if arm not in state_dict or arm not in action_dict or field not in state_dict[arm]:
            continue
        st = _flatten_history(state_dict[arm][field])
        ac = _flatten_history(action_dict[arm][field])
        for row in range(n_rows):
            ax = axs[row, col]
            ax.plot(t, st[:, row], label="state", linewidth=1.5, color=STATE_COLOR)
            ax.plot(
                t,
                ac[:, row],
                label="action",
                linewidth=1.5,
                linestyle="--",
                alpha=0.85,
                color=ACTION_COLOR,
            )
            _draw_gap_markers(ax, gap_times_s, draw_label=(row == 0 and col == 1))
            ax.set_ylabel(dim_labels[row])
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(ARM_LABELS[arm])
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="upper right")

    for ax in axs[-1]:
        ax.set_xlabel("time (s, gaps removed)")

    fig.suptitle(f"{field} state & action ({rate_key})", fontsize=13)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {save_path}")


def vis_state_action(
    data_dict: dict[str, Any],
    output_dir: str | Path,
    *,
    rate_key: str = "30hz",
    gap_threshold_ms: float = DEFAULT_GAP_THRESHOLD_MS,
) -> None:
    """Plot state vs action on a gap-compressed timeline; mark recording gaps with vertical lines."""
    state_key = f"state_{rate_key}"
    action_key = f"action_{rate_key}"
    state_dict = data_dict[state_key]
    action_dict = data_dict[action_key]
    timestamps = np.asarray(data_dict["timestamps"], dtype=np.int64)
    t = _connected_time_s(timestamps, gap_threshold_ms=gap_threshold_ms)

    output_dir = Path(output_dir) / "state_action"
    for field in ROBOT_STREAMS:
        _plot_state_action_field(
            field=field,
            state_dict=state_dict,
            action_dict=action_dict,
            t=t,
            timestamps_ms=timestamps,
            save_path=output_dir / f"{field}_{rate_key}.png",
            rate_key=rate_key,
            gap_threshold_ms=gap_threshold_ms,
        )
