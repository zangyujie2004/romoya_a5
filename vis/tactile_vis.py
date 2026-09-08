from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from task.streams import TACTILE_BUNDLES
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


def vis_tactile(
    data_dict: dict[str, Any],
    save_dir: str | Path,
    step: int = 5,
    bundles: list[str] | None = None,
    **kwargs,
) -> None:
    """Visualize tactile displacement (dx, dy, dz): quiver + depth heatmap."""
    aligned_tactile = data_dict["tactile"]
    timestamps = np.asarray(data_dict["timestamps"], dtype=np.int64)

    if bundles is None:
        bundles = list(TACTILE_BUNDLES)
    bundles = [b for b in bundles if b in aligned_tactile and "pointcloud" in aligned_tactile[b]]
    if not bundles:
        print("vis_tactile: no pointcloud streams found, skip")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tactile_vis = [tactile_vis_channels(aligned_tactile[b]["pointcloud"]) for b in bundles]
    timestamps_sec = timestamps / 1000.0
    timestamps_rel = timestamps_sec - timestamps_sec[0]

    grid_h, grid_w = tactile_vis[0][0].shape[-2:]
    x_mesh, y_mesh, extent, x_span, y_span = tactile_grid_coords(grid_h, grid_w)
    style = kwargs.get("style", "warm")
    quiver_scale = kwargs.get("quiver_scale")
    if quiver_scale is None:
        quiver_scale = tactile_quiver_scale(
            tactile_vis,
            x_span,
            y_span,
            target_frac=kwargs.get("quiver_target_frac", 0.35),
        )
    depth_cmap = resolve_tactile_depth_cmap(style, kwargs)
    depth_alpha = kwargs.get("depth_alpha", 1.0)
    depth_mask_eps = kwargs.get("depth_mask_eps", 1e-4)
    dpi = kwargs.get("dpi", 150)
    z_min, z_max = tactile_depth_range(tactile_vis)

    ncols = min(len(bundles), 2)
    nrows = int(np.ceil(len(bundles) / ncols))
    fig_width = kwargs.get("fig_width", 5.2 * ncols)
    ax_width = fig_width / ncols * 0.92
    ax_height = ax_width * y_span / x_span
    fig_height = ax_height * nrows + 0.85

    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), facecolor="white")
    axs = np.atleast_1d(axs).flatten()
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.06, wspace=0.14, hspace=0.22)

    hms, qvs = [], []
    for ax, bundle, (dx, dy, depth) in zip(axs, bundles, tactile_vis):
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        style_tactile_ax(ax, x_mesh, y_mesh, style, kwargs)
        ax.set_title(bundle, fontsize=10, pad=4)
        hm = ax.imshow(
            mask_tactile_depth(depth[0], z_min, z_max, depth_mask_eps),
            cmap=depth_cmap,
            vmin=z_min,
            vmax=z_max,
            extent=extent,
            origin="upper",
            alpha=depth_alpha,
            zorder=0,
        )
        qv = add_tactile_quiver(ax, x_mesh, y_mesh, dx[0], dy[0], quiver_scale, style, kwargs)
        hms.append(hm)
        qvs.append((qv, dx, dy, depth))
    for ax in axs[len(bundles) :]:
        ax.axis("off")

    count = 0
    for i in range(0, len(timestamps), step):
        for hm, (qv, dx, dy, depth) in zip(hms, qvs):
            hm.set_data(mask_tactile_depth(depth[i], z_min, z_max, depth_mask_eps))
            qv.set_UVC(dx[i], dy[i])
        fig.suptitle(f"t={timestamps_rel[i]:.3f}s (frame {i})", fontsize=11, y=0.97)
        t_str = f"{timestamps_rel[i]:.3f}".replace(".", "_")
        fig.savefig(save_dir / f"{t_str}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        count += 1
    plt.close(fig)
    print(f"vis_tactile: saved {count} frames (step={step}) -> {save_dir}")
