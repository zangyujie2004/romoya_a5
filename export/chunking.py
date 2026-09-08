from __future__ import annotations

import numpy as np

DEFAULT_CHUNK_TARGET_MB = 4
DEFAULT_MAX_CHUNK_T = 4096
DEFAULT_CAMERA_CHUNK_CAP = 128
DEFAULT_TACTILE_CHUNK_CAP = 512


def storage_dtype(name: str) -> np.dtype:
    if name in ("camera", "camera_base_remove_hand"):
        return np.uint8
    if name == "event":
        return np.int64
    return np.float32


def adaptive_chunk_t(
    name: str,
    feature_shape: tuple[int, ...],
    dtype: str | np.dtype,
    *,
    target_bytes: int = DEFAULT_CHUNK_TARGET_MB * 1024 * 1024,
    max_chunk_t: int = DEFAULT_MAX_CHUNK_T,
    camera_chunk_cap: int = DEFAULT_CAMERA_CHUNK_CAP,
    tactile_chunk_cap: int = DEFAULT_TACTILE_CHUNK_CAP,
) -> int:
    itemsize = int(np.dtype(dtype).itemsize)
    feat_elems = int(np.prod(feature_shape)) if feature_shape else 1
    bytes_per_frame = max(itemsize * feat_elems, 1)
    chunk_t = target_bytes // bytes_per_frame

    caps = {
        "camera": camera_chunk_cap,
        "camera_base_remove_hand": camera_chunk_cap,
        "tactile": tactile_chunk_cap,
    }
    cap = caps.get(name, max_chunk_t)
    return max(1, min(int(chunk_t), cap, max_chunk_t))


def dataset_chunks(
    name: str,
    feature_shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    target_bytes: int,
    total_t: int | None = None,
    max_chunk_t: int = DEFAULT_MAX_CHUNK_T,
    camera_chunk_cap: int = DEFAULT_CAMERA_CHUNK_CAP,
    tactile_chunk_cap: int = DEFAULT_TACTILE_CHUNK_CAP,
) -> tuple[int, ...]:
    chunk_t = adaptive_chunk_t(
        name,
        feature_shape,
        dtype,
        target_bytes=target_bytes,
        max_chunk_t=max_chunk_t,
        camera_chunk_cap=camera_chunk_cap,
        tactile_chunk_cap=tactile_chunk_cap,
    )
    if total_t is not None:
        chunk_t = min(chunk_t, total_t)
    return (max(1, chunk_t),) + tuple(feature_shape)
