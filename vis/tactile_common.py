"""Shared tactile pointcloud visualization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

TACTILE_VIS_BG = "#FFD966"
TACTILE_SCIENTIFIC_BG = "#F5F5F5"


def tactile_vis_channels(pointcloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use displacement channels for visualization: dx, dy, dz.

    Prometheus stores 6 channels per taxel: [x, y, z, dx, dy, dz].
    Legacy 3-channel arrays are treated as [dx, dy, dz].
    """
    pc = np.asarray(pointcloud, dtype=np.float32)
    if pc.shape[-1] >= 6:
        return pc[..., 3], pc[..., 4], pc[..., 5]
    return pc[..., 0], pc[..., 1], pc[..., 2]


def tactile_grid_coords(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray, list[float], float, float]:
    x = np.linspace(-8.5, 8.5, grid_w)
    y = np.linspace(30, 0, grid_h)
    x_mesh, y_mesh = np.meshgrid(x, y)
    extent = [x.min(), x.max(), y.min(), y.max()]
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    return x_mesh, y_mesh, extent, x_span, y_span


def tactile_quiver_scale(
    tactile_vis: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    x_span: float,
    y_span: float,
    target_frac: float = 0.35,
) -> float:
    max_mag = 0.0
    for dx, dy, _ in tactile_vis:
        mag = np.hypot(np.asarray(dx, dtype=np.float64), np.asarray(dy, dtype=np.float64))
        if mag.size:
            max_mag = max(max_mag, float(np.nanmax(mag)))
    grid_span = min(x_span, y_span)
    if max_mag < 1e-8:
        return 1.0
    return max_mag / (grid_span * target_frac)


def tactile_depth_cmap_warm(bg: str | None = None) -> LinearSegmentedColormap:
    bg = bg or TACTILE_VIS_BG
    return LinearSegmentedColormap.from_list(
        "tactile_warm",
        [bg, "#FFCC4D", "#FF9F1C", "#E07000", "#8B4513"],
    )


def tactile_depth_cmap_scientific() -> str:
    return "viridis"


def resolve_tactile_depth_cmap(style: str, kwargs: dict[str, Any]) -> LinearSegmentedColormap | str:
    cmap = kwargs.get("depth_cmap")
    if cmap is not None:
        return cmap
    if style == "scientific":
        return tactile_depth_cmap_scientific()
    return tactile_depth_cmap_warm(kwargs.get("tactile_bg", TACTILE_VIS_BG))


def resolve_tactile_bg(style: str, kwargs: dict[str, Any]) -> str:
    if kwargs.get("tactile_bg") is not None:
        return kwargs["tactile_bg"]
    if style == "scientific":
        return TACTILE_SCIENTIFIC_BG
    return TACTILE_VIS_BG


def mask_tactile_depth(dz: np.ndarray, z_min: float, z_max: float, eps: float) -> np.ma.MaskedArray:
    return np.ma.masked_where(dz >= z_max - eps, dz)


def style_tactile_ax(ax, x_mesh, y_mesh, style: str, kwargs: dict[str, Any], *, small: bool = False) -> None:
    bg = resolve_tactile_bg(style, kwargs)
    ax.set_facecolor(bg)
    scatter_s = kwargs.get("tactile_scatter_s", 14 if not small else 7)
    scatter_alpha = kwargs.get("tactile_scatter_alpha", 0.9 if style != "scientific" else 0.75)
    marker_color = kwargs.get("tactile_marker_color", "black" if style != "scientific" else "#333333")
    ax.scatter(
        x_mesh,
        y_mesh,
        color=marker_color,
        s=scatter_s,
        alpha=scatter_alpha,
        zorder=1,
        linewidths=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    grid_alpha = kwargs.get("tactile_grid_alpha", 0.25 if not small else 0.2)
    grid_color = "#333333" if style != "scientific" else "#BBBBBB"
    ax.grid(True, color=grid_color, alpha=grid_alpha, linewidth=0.5)


def add_tactile_quiver(
    ax,
    x_mesh,
    y_mesh,
    u,
    v,
    quiver_scale: float,
    style: str,
    kwargs: dict[str, Any],
    *,
    small: bool = False,
):
    width = kwargs.get("quiver_width", 0.006 if not small else 0.0045)
    headwidth = kwargs.get("headwidth", 4.5 if not small else 3.5)
    headlength = kwargs.get("headlength", 4.0 if not small else 3.2)
    alpha = kwargs.get("quiver_alpha", 0.95)
    color = kwargs.get("quiver_color", "black" if style != "scientific" else "#D62728")
    return ax.quiver(
        x_mesh,
        y_mesh,
        u,
        v,
        scale_units="xy",
        scale=quiver_scale,
        width=width,
        alpha=alpha,
        headwidth=headwidth,
        headlength=headlength,
        color=color,
        zorder=2,
    )


def tactile_depth_range(tactile_vis: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    all_z = np.concatenate([depth.reshape(-1) for _, _, depth in tactile_vis])
    z_min, z_max = float(np.nanmin(all_z)), float(np.nanmax(all_z))
    if z_max - z_min < 1e-8:
        z_max = z_min + 1e-8
    return z_min, z_max
