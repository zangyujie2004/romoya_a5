from __future__ import annotations

from pathlib import Path

from pipeline.dagger import dagger_enabled
from export.base import EpisodeExporter
from export.hdf5_writer import Hdf5Writer
from export.lerobot_writer import LeRobotWriter
from export.zarr_writer import ZarrWriter


def replay_buffer_path(output_path: Path, export_cfg: dict) -> Path:
    fmt = str(export_cfg.get("format", "none")).lower()
    if fmt == "hdf5":
        return output_path / "replay_buffer.hdf5"
    if fmt == "lerobot":
        return output_path / "lerobot"
    return output_path / "replay_buffer.zarr"


def create_exporter(export_cfg: dict, output_path: Path, *, cfg: dict | None = None) -> EpisodeExporter | None:
    fmt = str(export_cfg.get("format", "none")).lower()
    if fmt in ("none", ""):
        return None
    include_event = dagger_enabled(cfg or {})
    if fmt == "zarr":
        return ZarrWriter(
            replay_buffer_path(output_path, export_cfg),
            include_event=include_event,
            **(export_cfg.get("zarr") or {}),
        )
    if fmt == "hdf5":
        return Hdf5Writer(
            replay_buffer_path(output_path, export_cfg),
            include_event=include_event,
            **(export_cfg.get("hdf5") or {}),
        )
    if fmt == "lerobot":
        return LeRobotWriter(replay_buffer_path(output_path, export_cfg), **(export_cfg.get("lerobot") or {}))
    raise ValueError(f"unsupported export format: {fmt}")
