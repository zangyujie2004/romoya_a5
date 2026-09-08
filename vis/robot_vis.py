from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from task.streams import ROBOT_BUNDLES

RAW_COLOR = "#1f77b4"
VQ_COLOR = "#ff7f0e"


def vis_gripper_vq(
    robot_dict: dict,
    gripper_before: dict[str, np.ndarray],
    *,
    threshold: float,
    save_path: str | Path,
) -> None:
    """Plot raw vs discretized gripper (joint[:, 6]); raw eef has no gripper dim."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = min(robot_dict[arm]["timestamps"][0] for arm in ROBOT_BUNDLES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, arm in zip(axes, ROBOT_BUNDLES):
        timestamps = np.asarray(robot_dict[arm]["timestamps"], dtype=np.int64)
        t = (timestamps - t0) / 1000.0
        raw = np.asarray(gripper_before[arm], dtype=np.float64)
        vq = np.asarray(robot_dict[arm]["joint"][:, 6], dtype=np.float64)
        ax.plot(t, raw, label="before", linewidth=1.2, color=RAW_COLOR, alpha=0.85)
        ax.plot(t, vq, label="after", linewidth=1.2, color=VQ_COLOR, alpha=0.85)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"threshold={threshold:.3f}")
        ax.set_title(f"{arm} gripper")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("gripper width (m)")
    fig.suptitle("gripper_vq: before vs after (joint[:, 6])", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {save_path}")


def vis_robot_eef(robot_dict, task_window=None, save_path=None):
    t0 = min(robot_dict[arm]["timestamps"][0] for arm in ROBOT_BUNDLES)
    left_xyz = robot_dict["left"]["eef"][:, :3]
    right_xyz = robot_dict["right"]["eef"][:, :3]
    t_left = (robot_dict["left"]["timestamps"] - t0) / 1000.0
    t_right = (robot_dict["right"]["timestamps"] - t0) / 1000.0
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for i, axis in enumerate("xyz"):
        ax_l.plot(t_left, left_xyz[:, i], label=f"left {axis}")
        ax_r.plot(t_right, right_xyz[:, i], label=f"right {axis}")
    if task_window is not None:
        t_start = (task_window[0] - t0) / 1000.0
        t_end = (task_window[1] - t0) / 1000.0
        for ax in (ax_l, ax_r):
            ax.axvline(t_start, color="green", linestyle="--", linewidth=1.5, label="task start")
            ax.axvline(t_end, color="red", linestyle="--", linewidth=1.5, label="task end")
        for arm, ax in (("left", ax_l), ("right", ax_r)):
            t_arm_start = (task_window[0] - t0) / 1000.0
            t_arm_end = (task_window[1] - t0) / 1000.0
            ax.axvline(t_arm_start, color="green", linestyle=":", linewidth=1, alpha=0.6)
            ax.axvline(t_arm_end, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax_l.set_title("left eef")
    ax_r.set_title("right eef")
    for ax in (ax_l, ax_r):
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
    ax_l.set_ylabel("position (m)")
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
