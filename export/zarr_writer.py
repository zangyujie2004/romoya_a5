from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from export.base import EpisodeExporter
from export.chunking import DEFAULT_CHUNK_TARGET_MB, adaptive_chunk_t, storage_dtype
from export.layout import (
    BASE_REMOVE_HAND_NONE,
    BASE_REMOVE_HAND_PRESENT,
    CAMERA_BASE_REMOVE_HAND_KEY,
    META_EPISODE_ENDS_REMOVE_HAND,
)
from export.merge import (
    episode_to_storage_arrays,
    extract_camera_base_remove_hand,
    storage_keys_120hz,
    storage_keys_30hz,
)


def _zarr_dtype(name: str, arr: np.ndarray) -> str:
    return str(np.dtype(storage_dtype(name)))


def _create_resizable_array(group, name, feature_shape, *, dtype, chunk_t, max_chunk_t, compressors=None):
    feature_shape = tuple(feature_shape)
    chunks = (min(chunk_t, max_chunk_t),) + feature_shape
    kwargs = dict(name=name, shape=(0,) + feature_shape, chunks=chunks, dtype=dtype)
    if compressors is not None:
        kwargs["compressors"] = compressors
    return group.create_array(**kwargs)


class ZarrWriter(EpisodeExporter):
    def __init__(
        self,
        output_path: Path | str,
        *,
        include_event: bool = False,
        compress: bool = False,
        buffer_episodes: bool = True,
        chunk_target_mb: float = DEFAULT_CHUNK_TARGET_MB,
        max_chunk_t: int = 4096,
        camera_chunk_cap: int = 128,
        tactile_chunk_cap: int = 512,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.output_path = output_path
        self.include_event = include_event
        self.keys_30hz = storage_keys_30hz(include_event=include_event)
        self.keys_120hz = storage_keys_120hz()
        self.all_keys = self.keys_30hz + self.keys_120hz
        self.compressors = None if not compress else "auto"
        self.buffer_episodes = buffer_episodes
        self.chunk_target_bytes = int(chunk_target_mb * 1024 * 1024)
        self.max_chunk_t = int(max_chunk_t)
        self.camera_chunk_cap = int(camera_chunk_cap)
        self.tactile_chunk_cap = int(tactile_chunk_cap)

        self.root = zarr.open_group(str(output_path), mode="w")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

        self.arrays: dict[str, zarr.Array] = {}
        self.schemas: dict[str, tuple[tuple[int, ...], str]] = {}
        self.buffers: dict[str, list[np.ndarray]] = {key: [] for key in self.all_keys}

        self.episode_ends: list[int] = []
        self.episode_ends_4x: list[int] = []
        self.total_30hz = 0
        self.total_4x = 0

        # Compact dual remove-hand bypass (only present episodes).
        self._remove_hand_buffers: list[np.ndarray] = []
        self.episode_ends_remove_hand: list[int] = []
        self.total_remove_hand = 0
        self.base_remove_hand_flags: list[str] = []

    def _validate_episode(self, arrays: dict[str, np.ndarray]) -> tuple[int, int]:
        t_30 = int(arrays["camera"].shape[0])
        if t_30 == 0:
            raise ValueError("episode has empty timeline")

        for name in self.keys_30hz:
            arr = np.asarray(arrays[name])
            if arr.shape[0] != t_30:
                raise ValueError(f"{name} length mismatch: {arr.shape[0]} vs {t_30}")

        t_4x = t_30 * 4
        for name in self.keys_120hz:
            arr = np.asarray(arrays[name])
            if arr.shape[0] != t_4x:
                raise ValueError(f"{name} 120hz length mismatch: {arr.shape[0]} vs {t_4x}")

        return t_30, t_4x

    def _get_or_create_array(self, name: str, arr: np.ndarray) -> zarr.Array:
        arr = np.asarray(arr)
        feature_shape = tuple(arr.shape[1:])
        dtype = _zarr_dtype(name, arr)

        if name in self.arrays:
            if feature_shape != self.schemas[name][0]:
                raise ValueError(
                    f"{name} shape mismatch: expected (*, {self.schemas[name][0]}), got {arr.shape}"
                )
            return self.arrays[name]

        chunk_t = adaptive_chunk_t(
            name,
            feature_shape,
            dtype,
            target_bytes=self.chunk_target_bytes,
            max_chunk_t=self.max_chunk_t,
            camera_chunk_cap=self.camera_chunk_cap,
            tactile_chunk_cap=self.tactile_chunk_cap,
        )
        zarr_arr = _create_resizable_array(
            self.data,
            name,
            feature_shape,
            dtype=dtype,
            chunk_t=chunk_t,
            max_chunk_t=self.max_chunk_t,
            compressors=self.compressors,
        )
        self.arrays[name] = zarr_arr
        self.schemas[name] = (feature_shape, dtype)
        return zarr_arr

    def _append_remove_hand(self, remove: np.ndarray) -> None:
        remove = np.asarray(remove, dtype=storage_dtype(CAMERA_BASE_REMOVE_HAND_KEY))
        if self.buffer_episodes:
            self._remove_hand_buffers.append(remove)
        else:
            self._get_or_create_array(CAMERA_BASE_REMOVE_HAND_KEY, remove).append(
                remove, axis=0
            )
        self.total_remove_hand += int(remove.shape[0])
        self.episode_ends_remove_hand.append(self.total_remove_hand)

    def append_episode(
        self,
        data_dict: dict,
        *,
        episode_idx: int,
        episode_path: Path | None = None,
    ) -> None:
        del episode_idx, episode_path  # sequential append order defines episode index
        arrays = episode_to_storage_arrays(data_dict, include_event=self.include_event)
        cast = {}
        for name, arr in arrays.items():
            cast[name] = np.asarray(arr, dtype=storage_dtype(name))
        arrays = cast

        t_30, t_4x = self._validate_episode(arrays)

        if self.buffer_episodes:
            for name, arr in arrays.items():
                self.buffers[name].append(arr)
        else:
            for name, arr in arrays.items():
                self._get_or_create_array(name, arr).append(arr, axis=0)

        self.total_30hz += t_30
        self.episode_ends.append(self.total_30hz)
        self.total_4x += t_4x
        self.episode_ends_4x.append(self.total_4x)

        remove = extract_camera_base_remove_hand(data_dict.get("camera") or {})
        if remove is None:
            self.base_remove_hand_flags.append(BASE_REMOVE_HAND_NONE)
        else:
            if int(remove.shape[0]) != t_30:
                raise ValueError(
                    f"{CAMERA_BASE_REMOVE_HAND_KEY} T={remove.shape[0]} != camera T={t_30}"
                )
            self._append_remove_hand(remove)
            self.base_remove_hand_flags.append(BASE_REMOVE_HAND_PRESENT)

    def _flush_buffers(self) -> None:
        for name, chunks in self.buffers.items():
            if not chunks:
                continue
            data = np.concatenate(chunks, axis=0)
            feature_shape = tuple(data.shape[1:])
            dtype = _zarr_dtype(name, data)
            chunk_t = adaptive_chunk_t(
                name,
                feature_shape,
                dtype,
                target_bytes=self.chunk_target_bytes,
                max_chunk_t=self.max_chunk_t,
                camera_chunk_cap=self.camera_chunk_cap,
                tactile_chunk_cap=self.tactile_chunk_cap,
            )
            chunks_spec = (min(chunk_t, self.max_chunk_t),) + feature_shape
            kwargs = dict(name=name, data=data, chunks=chunks_spec)
            if self.compressors is not None:
                kwargs["compressors"] = self.compressors
            self.data.create_array(**kwargs)
            self.buffers[name].clear()

        if self._remove_hand_buffers:
            data = np.concatenate(self._remove_hand_buffers, axis=0)
            feature_shape = tuple(data.shape[1:])
            dtype = _zarr_dtype(CAMERA_BASE_REMOVE_HAND_KEY, data)
            chunk_t = adaptive_chunk_t(
                CAMERA_BASE_REMOVE_HAND_KEY,
                feature_shape,
                dtype,
                target_bytes=self.chunk_target_bytes,
                max_chunk_t=self.max_chunk_t,
                camera_chunk_cap=self.camera_chunk_cap,
                tactile_chunk_cap=self.tactile_chunk_cap,
            )
            chunks_spec = (min(chunk_t, self.max_chunk_t),) + feature_shape
            kwargs = dict(name=CAMERA_BASE_REMOVE_HAND_KEY, data=data, chunks=chunks_spec)
            if self.compressors is not None:
                kwargs["compressors"] = self.compressors
            self.data.create_array(**kwargs)
            self._remove_hand_buffers.clear()

    def finalize(self) -> None:
        if self.buffer_episodes:
            self._flush_buffers()

        for key in ("episode_ends", "episode_ends_4x", META_EPISODE_ENDS_REMOVE_HAND):
            if key in self.meta:
                del self.meta[key]

        self.meta.create_array(
            "episode_ends",
            data=np.asarray(self.episode_ends, dtype=np.int64),
        )
        self.meta.create_array(
            "episode_ends_4x",
            data=np.asarray(self.episode_ends_4x, dtype=np.int64),
        )
        # Only create compact ends when at least one present episode was written.
        if self.episode_ends_remove_hand:
            self.meta.create_array(
                META_EPISODE_ENDS_REMOVE_HAND,
                data=np.asarray(self.episode_ends_remove_hand, dtype=np.int64),
            )

    @property
    def has_camera_base_remove_hand(self) -> bool:
        return bool(self.episode_ends_remove_hand)
