from __future__ import annotations

from typing import Any

import numpy as np

from prometheus.data.raw_episode import pointcloud_to_array

TACTILE_GRID_HEIGHT = 35
TACTILE_GRID_WIDTH = 20
TACTILE_CHANNELS = 6
DZ_CHANNEL = 5
DEFAULT_PERCENTILE_LOW = 1.0
DEFAULT_PERCENTILE_HIGH = 99.0
DEFAULT_NOISE_FLOOR_M = 0.0005
DEFAULT_MAX_DEFORMATION_M = 0.01
DEFAULT_MEDIAN_KERNEL = 3


def _jet_bgr_lut() -> np.ndarray:
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.column_stack(
        [
            (blue * 255.0).astype(np.uint8),
            (green * 255.0).astype(np.uint8),
            (red * 255.0).astype(np.uint8),
        ]
    )


_JET_BGR_LUT = _jet_bgr_lut()


def pointcloud_msg_to_dz_grid(msg: Any) -> np.ndarray:
    flat = pointcloud_to_array(msg)
    return flat.reshape(TACTILE_GRID_HEIGHT, TACTILE_GRID_WIDTH, TACTILE_CHANNELS)[:, :, DZ_CHANNEL]


def normalize_grid_to_u8(
    grid: np.ndarray,
    *,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
) -> np.ndarray:
    grid_f = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(grid_f)
    if not np.any(finite):
        return np.zeros(grid_f.shape, dtype=np.uint8)

    valid = grid_f[finite]
    lo = float(np.percentile(valid, percentile_low))
    hi = float(np.percentile(valid, percentile_high))
    if hi <= lo + 1e-8:
        return np.zeros(grid_f.shape, dtype=np.uint8)

    clipped = np.clip(grid_f, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    normalized[~finite] = 0.0
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def normalize_compressive_dz_to_u8(
    grid: np.ndarray,
    *,
    noise_floor_m: float = DEFAULT_NOISE_FLOOR_M,
    max_deformation_m: float = DEFAULT_MAX_DEFORMATION_M,
    median_kernel: int = DEFAULT_MEDIAN_KERNEL,
) -> np.ndarray:
    noise_floor_m = float(noise_floor_m)
    max_deformation_m = float(max_deformation_m)
    median_kernel = int(median_kernel)
    if not np.isfinite(noise_floor_m) or noise_floor_m < 0.0:
        raise ValueError("tactile noise_floor_m must be finite and non-negative")
    if not np.isfinite(max_deformation_m) or max_deformation_m <= noise_floor_m:
        raise ValueError("tactile max_deformation_m must exceed noise_floor_m")
    if median_kernel <= 0 or median_kernel % 2 == 0:
        raise ValueError("tactile median_kernel must be a positive odd integer")

    grid_f = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(grid_f)
    # Xense compression is reported as negative dz. Positive dz is treated as
    # non-contact noise for this deformation preview.
    deformation = np.maximum(-np.where(finite, grid_f, 0.0), 0.0)
    if median_kernel > 1:
        radius = median_kernel // 2
        padded = np.pad(deformation, radius, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (median_kernel, median_kernel),
        )
        deformation = np.median(windows, axis=(-2, -1))

    filtered = np.maximum(deformation - noise_floor_m, 0.0)
    normalized = filtered / (max_deformation_m - noise_floor_m)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def dz_grid_to_bgr(
    grid: np.ndarray,
    *,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
    noise_floor_m: float | None = None,
    max_deformation_m: float | None = None,
    median_kernel: int = DEFAULT_MEDIAN_KERNEL,
) -> np.ndarray:
    if max_deformation_m is None:
        grid_u8 = normalize_grid_to_u8(
            grid,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
    else:
        grid_u8 = normalize_compressive_dz_to_u8(
            grid,
            noise_floor_m=(DEFAULT_NOISE_FLOOR_M if noise_floor_m is None else noise_floor_m),
            max_deformation_m=max_deformation_m,
            median_kernel=median_kernel,
        )
    return _JET_BGR_LUT[grid_u8]


def pointcloud_msg_to_dz_bgr(
    msg: Any,
    *,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
    noise_floor_m: float | None = None,
    max_deformation_m: float | None = None,
    median_kernel: int = DEFAULT_MEDIAN_KERNEL,
) -> np.ndarray:
    return dz_grid_to_bgr(
        pointcloud_msg_to_dz_grid(msg),
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        noise_floor_m=noise_floor_m,
        max_deformation_m=max_deformation_m,
        median_kernel=median_kernel,
    )


def is_tactile_flow_stream(name: str) -> bool:
    return str(name).endswith("_tactile_flow")
