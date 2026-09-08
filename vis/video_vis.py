from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from task.streams import CAMERA_BUNDLES, ROBOT_BUNDLES, TACTILE_BUNDLES
from vis.tactile_common import (
    add_tactile_quiver,
    mask_tactile_depth,
    resolve_tactile_depth_cmap,
    style_tactile_ax,
    tactile_depth_range,
    tactile_grid_coords,
    tactile_quiver_scale,
    tactile_vis_channels,
)
from vis.state_action_vis import _extract_tcp_from_eef


def _fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _write_mp4(save_path: Path, fps: int, frame_size: tuple[int, int], frame_iter) -> None:
    """Encode h264 locally (compatible + OSS-safe), then copy to destination."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    width, height = frame_size
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-movflags",
        "+faststart",
        tmp_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frame_iter:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read().decode()
        ret = proc.wait()
    if ret != 0:
        os.unlink(tmp_path)
        raise RuntimeError(f"ffmpeg failed (exit {ret}): {stderr}")
    try:
        shutil.copy2(tmp_path, save_path)
    finally:
        os.unlink(tmp_path)


def vis_episode_video(
    data_dict: dict[str, Any],
    save_path: str | Path,
    fps: int = 30,
    camera_bundles: list[str] | None = None,
    tactile_bundles: list[str] | None = None,
    **kwargs,
) -> None:
    """30Hz overview video: cameras + tactile (scientific colormap) + TCP state/action."""
    aligned_camera = data_dict["camera"]
    aligned_tactile = data_dict["tactile"]
    state_30hz = data_dict["state_30hz"]
    action_30hz = data_dict["action_30hz"]
    timestamps = np.asarray(data_dict["timestamps"], dtype=np.int64)

    t = (timestamps - timestamps[0]) / 1000.0
    if len(t) == 0:
        print("vis_episode_video: empty timeline, skip")
        return

    if camera_bundles is None:
        camera_bundles = [b for b in CAMERA_BUNDLES if b in aligned_camera and "rgb" in aligned_camera[b]]
    if tactile_bundles is None:
        tactile_bundles = [
            b for b in TACTILE_BUNDLES if b in aligned_tactile and "pointcloud" in aligned_tactile[b]
        ]
    if not camera_bundles and not tactile_bundles:
        print("vis_episode_video: no camera/tactile data, skip")
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cam_rgbs = [np.asarray(aligned_camera[b]["rgb"]) for b in camera_bundles]
    tactile_vis = [tactile_vis_channels(aligned_tactile[b]["pointcloud"]) for b in tactile_bundles]

    style = "scientific"
    depth_mask_eps = kwargs.get("depth_mask_eps", 1e-4)
    depth_cmap = resolve_tactile_depth_cmap(style, kwargs)
    depth_alpha = kwargs.get("depth_alpha", 0.92)
    tac_quiver_scale = None
    tac_extent = None
    tac_x_mesh, tac_y_mesh = None, None
    tac_box_aspect = 1.0
    z_min, z_max = 0.0, 1.0

    if tactile_vis:
        grid_h, grid_w = tactile_vis[0][0].shape[-2:]
        tac_x_mesh, tac_y_mesh, tac_extent, x_span, y_span = tactile_grid_coords(grid_h, grid_w)
        tac_box_aspect = y_span / x_span
        z_min, z_max = tactile_depth_range(tactile_vis)
        tac_quiver_scale = kwargs.get("quiver_scale")
        if tac_quiver_scale is None:
            tac_quiver_scale = tactile_quiver_scale(
                tactile_vis,
                x_span,
                y_span,
                target_frac=kwargs.get("quiver_target_frac", 0.35),
            )

    robot_series = {}
    for bundle in ROBOT_BUNDLES:
        if bundle not in state_30hz or bundle not in action_30hz:
            continue
        st_xyz, st_g = _extract_tcp_from_eef(state_30hz[bundle]["eef"])
        ac_xyz, ac_g = _extract_tcp_from_eef(action_30hz[bundle]["eef"])
        robot_series[bundle] = {"st_xyz": st_xyz, "st_g": st_g, "ac_xyz": ac_xyz, "ac_g": ac_g}

    fig = plt.figure(figsize=(28, 11), facecolor="white")
    gs_root = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.9, 0.82], wspace=0.018)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs_root[0],
        height_ratios=[0.78, 1.22],
        hspace=0.008,
    )

    cam_imgs = []
    if camera_bundles:
        gs_cam = gridspec.GridSpecFromSubplotSpec(
            1,
            len(camera_bundles),
            subplot_spec=gs_left[0],
            wspace=0.004,
        )
        for j, bundle in enumerate(camera_bundles):
            ax = fig.add_subplot(gs_cam[j])
            img = cam_rgbs[j][0]
            im = ax.imshow(img, aspect="auto")
            ax.set_title(bundle, fontsize=7, pad=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            cam_imgs.append(im)
    else:
        ax = fig.add_subplot(gs_left[0])
        ax.axis("off")
        ax.set_title("camera (none)", fontsize=9)

    tac_hms, tac_qvs = [], []
    if tactile_bundles:
        gs_tac = gridspec.GridSpecFromSubplotSpec(
            1,
            len(tactile_bundles),
            subplot_spec=gs_left[1],
            wspace=0.006,
        )
        for j, (bundle, (dx, dy, depth)) in enumerate(zip(tactile_bundles, tactile_vis)):
            ax = fig.add_subplot(gs_tac[j])
            ax.set_xlim(tac_extent[0], tac_extent[1])
            ax.set_ylim(tac_extent[2], tac_extent[3])
            ax.set_aspect("equal")
            ax.set_box_aspect(tac_box_aspect)
            style_tactile_ax(ax, tac_x_mesh, tac_y_mesh, style, kwargs, small=True)
            ax.set_title(bundle, fontsize=6, pad=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#666666")
            hm = ax.imshow(
                mask_tactile_depth(depth[0], z_min, z_max, depth_mask_eps),
                cmap=depth_cmap,
                vmin=z_min,
                vmax=z_max,
                extent=tac_extent,
                origin="upper",
                alpha=depth_alpha,
                zorder=0,
            )
            qv = add_tactile_quiver(
                ax,
                tac_x_mesh,
                tac_y_mesh,
                dx[0],
                dy[0],
                tac_quiver_scale,
                style,
                kwargs,
                small=True,
            )
            tac_hms.append(hm)
            tac_qvs.append((qv, dx, dy, depth))

    gs_robot = gridspec.GridSpecFromSubplotSpec(
        4,
        2,
        subplot_spec=gs_root[1],
        wspace=0.10,
        hspace=0.14,
    )
    axis_names = ["x", "y", "z", "gripper"]
    arm_titles = {"left": "left arm", "right": "right arm"}
    robot_lines = []
    robot_vlines = []
    for col, bundle in enumerate(ROBOT_BUNDLES):
        if bundle not in robot_series:
            continue
        series = robot_series[bundle]
        ydatas = [
            series["st_xyz"][:, 0],
            series["st_xyz"][:, 1],
            series["st_xyz"][:, 2],
            series["st_g"],
        ]
        ydata_a = [
            series["ac_xyz"][:, 0],
            series["ac_xyz"][:, 1],
            series["ac_xyz"][:, 2],
            series["ac_g"],
        ]
        for row in range(4):
            ax = fig.add_subplot(gs_robot[row, col])
            y_all = np.concatenate([ydatas[row], ydata_a[row]])
            margin = 0.05 * (np.nanmax(y_all) - np.nanmin(y_all) + 1e-8)
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(np.nanmin(y_all) - margin, np.nanmax(y_all) + margin)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            if row == 0:
                ax.set_title(arm_titles[bundle], fontsize=8, pad=1)
            if row == 3:
                ax.set_xlabel("time (s)", fontsize=7)
            ax.set_ylabel(
                f"{axis_names[row]} (m)" if row < 3 else "gripper (m)",
                fontsize=7,
            )
            ax.tick_params(labelsize=6)
            ln_s, = ax.plot([], [], linewidth=1.0, label="state", color="#1f77b4")
            ln_a, = ax.plot([], [], linewidth=1.0, linestyle="--", alpha=0.85, label="action", color="#ff7f0e")
            vline = ax.axvline(t[0], color="gray", linestyle=":", linewidth=1, alpha=0.7)
            robot_lines.append((ln_s, ln_a, ydatas[row], ydata_a[row]))
            robot_vlines.append(vline)
            if row == 0 and col == 1:
                ax.legend(fontsize=6, loc="upper right")

    fig.subplots_adjust(left=0.006, right=0.994, top=0.965, bottom=0.028)
    fig.suptitle(f"t={t[0]:.3f}s", fontsize=10, y=0.985)
    dpi = kwargs.get("dpi", 72)
    fig.set_dpi(dpi)
    fig.canvas.draw()
    frame_w, frame_h = fig.canvas.get_width_height()

    def _render_frames():
        for i in range(len(t)):
            for j, im in enumerate(cam_imgs):
                im.set_data(cam_rgbs[j][i])
            for hm, (qv, dx, dy, depth) in zip(tac_hms, tac_qvs):
                hm.set_data(mask_tactile_depth(depth[i], z_min, z_max, depth_mask_eps))
                qv.set_UVC(dx[i], dy[i])
            for (ln_s, ln_a, y_s, y_a), vline in zip(robot_lines, robot_vlines):
                ln_s.set_data(t[: i + 1], y_s[: i + 1])
                ln_a.set_data(t[: i + 1], y_a[: i + 1])
                vline.set_xdata([t[i], t[i]])
            fig.suptitle(f"t={t[i]:.3f}s  {i}/{len(t) - 1}", fontsize=10, y=0.985)
            yield _fig_to_bgr(fig)

    _write_mp4(save_path, fps, (frame_w, frame_h), _render_frames())
    plt.close(fig)
    print(f"vis_episode_video: saved {len(t)} frames @ {fps}fps -> {save_path}")
