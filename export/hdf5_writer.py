from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from export.base import EpisodeExporter
from export.chunking import DEFAULT_CHUNK_TARGET_MB, DEFAULT_MAX_CHUNK_T, dataset_chunks, storage_dtype
from export.merge import episode_to_storage_arrays, storage_keys_120hz, storage_keys_30hz


def _cast_storage_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cast = {}
    for name, arr in arrays.items():
        cast[name] = np.asarray(arr, dtype=storage_dtype(name))
    return cast


class Hdf5Writer(EpisodeExporter):
    def __init__(
        self,
        output_path: Path | str,
        *,
        include_event: bool = False,
        compress: bool = False,
        buffer_episodes: bool = True,
        chunk_target_mb: float = DEFAULT_CHUNK_TARGET_MB,
        max_chunk_t: int = DEFAULT_MAX_CHUNK_T,
        camera_chunk_cap: int = 128,
        tactile_chunk_cap: int = 512,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.include_event = include_event
        self.keys_30hz = storage_keys_30hz(include_event=include_event)
        self.keys_120hz = storage_keys_120hz()
        self.all_keys = self.keys_30hz + self.keys_120hz
        self.compress = compress
        self.buffer_episodes = buffer_episodes
        self.chunk_target_bytes = int(chunk_target_mb * 1024 * 1024)
        self.max_chunk_t = int(max_chunk_t)
        self.camera_chunk_cap = int(camera_chunk_cap)
        self.tactile_chunk_cap = int(tactile_chunk_cap)

        self.buffers: dict[str, list[np.ndarray]] = {key: [] for key in self.all_keys}
        self.file: h5py.File | None = None
        self.datasets: dict[str, h5py.Dataset] = {}
        self.schemas: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

        self.episode_ends: list[int] = []
        self.episode_ends_4x: list[int] = []
        self.total_30hz = 0
        self.total_4x = 0

    def _validate_episode(self, arrays: dict[str, np.ndarray]) -> tuple[int, int]:
        t_30 = int(arrays["camera"].shape[0])
        if t_30 == 0:
            raise ValueError("episode has empty timeline")

        for name in self.keys_30hz:
            if np.asarray(arrays[name]).shape[0] != t_30:
                raise ValueError(f"{name} length mismatch with camera at 30hz")

        t_4x = t_30 * 4
        for name in self.keys_120hz:
            if np.asarray(arrays[name]).shape[0] != t_4x:
                raise ValueError(f"{name} length mismatch with 120hz timeline")

        return t_30, t_4x

    def _open_file(self) -> h5py.File:
        if self.file is None:
            self.file = h5py.File(self.output_path, "w")
            self.file.create_group("data")
            self.file.create_group("meta")
        return self.file

    def _compression(self) -> str | None:
        return "gzip" if self.compress else None

    def _append_to_dataset(self, name: str, arr: np.ndarray) -> None:
        file = self._open_file()
        arr = np.asarray(arr, dtype=storage_dtype(name))
        feature_shape = tuple(arr.shape[1:])
        dtype = storage_dtype(name)

        if name not in self.datasets:
            chunks = dataset_chunks(
                name,
                feature_shape,
                dtype,
                target_bytes=self.chunk_target_bytes,
                total_t=arr.shape[0],
                max_chunk_t=self.max_chunk_t,
                camera_chunk_cap=self.camera_chunk_cap,
                tactile_chunk_cap=self.tactile_chunk_cap,
            )
            self.datasets[name] = file["data"].create_dataset(
                name,
                shape=(0, *feature_shape),
                maxshape=(None, *feature_shape),
                chunks=chunks,
                dtype=dtype,
                compression=self._compression(),
            )
            self.schemas[name] = (feature_shape, dtype)
        elif feature_shape != self.schemas[name][0]:
            raise ValueError(f"{name} shape mismatch: expected (*, {self.schemas[name][0]}), got {arr.shape}")

        dset = self.datasets[name]
        old_t = dset.shape[0]
        new_t = old_t + arr.shape[0]
        dset.resize((new_t, *feature_shape))
        dset[old_t:new_t] = arr

    def append_episode(
        self,
        data_dict: dict,
        *,
        episode_idx: int,
        episode_path: Path | None = None,
    ) -> None:
        arrays = _cast_storage_arrays(
            episode_to_storage_arrays(data_dict, include_event=self.include_event)
        )
        t_30, t_4x = self._validate_episode(arrays)

        if self.buffer_episodes:
            for name, arr in arrays.items():
                self.buffers[name].append(arr)
        else:
            for name, arr in arrays.items():
                self._append_to_dataset(name, arr)

        self.total_30hz += t_30
        self.episode_ends.append(self.total_30hz)
        self.total_4x += t_4x
        self.episode_ends_4x.append(self.total_4x)

    def _flush_buffers(self) -> None:
        file = self._open_file()
        for name, chunks in self.buffers.items():
            if not chunks:
                continue
            data = np.concatenate(chunks, axis=0)
            feature_shape = tuple(data.shape[1:])
            dtype = storage_dtype(name)
            file["data"].create_dataset(
                name,
                data=data,
                chunks=dataset_chunks(
                    name,
                    feature_shape,
                    dtype,
                    target_bytes=self.chunk_target_bytes,
                    total_t=data.shape[0],
                    max_chunk_t=self.max_chunk_t,
                    camera_chunk_cap=self.camera_chunk_cap,
                    tactile_chunk_cap=self.tactile_chunk_cap,
                ),
                dtype=dtype,
                compression=self._compression(),
            )
            self.buffers[name].clear()

    def finalize(self) -> None:
        if self.buffer_episodes:
            self._flush_buffers()

        file = self._open_file()
        for key, values in (
            ("episode_ends", self.episode_ends),
            ("episode_ends_4x", self.episode_ends_4x),
        ):
            if key in file["meta"]:
                del file["meta"][key]
            file["meta"].create_dataset(key, data=np.asarray(values, dtype=np.int64))

        if self.file is not None:
            self.file.close()
            self.file = None
