from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from export.base import EpisodeExporter
from export.layout import CAMERA_BUNDLE_ORDER, CAMERA_STREAM
from export.lerobot_features import (
    CODEBASE_VERSION,
    LEGACY_DATA_PATH,
    LEGACY_VIDEO_PATH,
    build_lerobot_features,
    camera_feature_key,
    episode_chunk,
)
from export.merge import merge_robot, merge_tactile

try:
    from lerobot.datasets.compute_stats import compute_episode_stats, get_feature_stats
    from lerobot.datasets.utils import serialize_dict
    from lerobot.datasets.video_utils import encode_video_frames
except ImportError as exc:
    raise ImportError("lerobot is required for LeRobot export. Install with: pip install lerobot") from exc


def _video_stats_from_frames(frames: np.ndarray) -> dict:
    n = min(100, len(frames))
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    sampled = np.asarray(frames[idx], dtype=np.float32).transpose(0, 3, 1, 2)
    stats = get_feature_stats(sampled, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class LeRobotWriter(EpisodeExporter):
    """Export episodes to LeRobot dataset format v2.1."""

    def __init__(
        self,
        output_path: Path | str,
        *,
        repo_id: str = "local/dataset",
        fps: int = 30,
        robot_type: str = "arx",
        task: str = "default",
        use_videos: bool = True,
        chunks_size: int = 1000,
        vcodec: str = "h264",
    ):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.repo_id = repo_id
        self.fps = int(fps)
        self.robot_type = robot_type
        self.default_task = task
        self.use_videos = use_videos
        self.chunks_size = int(chunks_size)
        self.vcodec = vcodec

        self.features: dict | None = None
        self.task_to_index: dict[str, int] = {}
        self.episode_records: list[dict] = []
        self.stats_records: list[dict] = []

        self.total_frames = 0
        self.total_episodes = 0
        self._meta_dir = self.output_path / "meta"

    def _task_index(self, task: str) -> int:
        if task not in self.task_to_index:
            self.task_to_index[task] = len(self.task_to_index)
        return self.task_to_index[task]

    def _data_path(self, episode_index: int) -> Path:
        chunk = episode_chunk(episode_index, self.chunks_size)
        rel = LEGACY_DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
        return self.output_path / rel

    def _video_path(self, episode_index: int, video_key: str) -> Path:
        chunk = episode_chunk(episode_index, self.chunks_size)
        rel = LEGACY_VIDEO_PATH.format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        return self.output_path / rel

    def _encode_camera_video(self, frames: np.ndarray, video_path: Path) -> None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp)
            for i, frame in enumerate(frames):
                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                    img_dir / f"frame-{i:06d}.png"
                )
            encode_video_frames(
                img_dir,
                video_path,
                fps=self.fps,
                vcodec=self.vcodec,
                overwrite=True,
            )

    def _write_episode_parquet(
        self,
        episode_index: int,
        *,
        timestamps: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        tactile: np.ndarray,
        task: str,
    ) -> None:
        length = int(state.shape[0])
        task_index = self._task_index(task)
        global_index = np.arange(self.total_frames, self.total_frames + length, dtype=np.int64)

        df = pd.DataFrame(
            {
                "timestamp": np.arange(length, dtype=np.float32) / self.fps,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": global_index,
                "task_index": np.full(length, task_index, dtype=np.int64),
                "observation.state": list(state.astype(np.float32)),
                "action": list(action.astype(np.float32)),
                "observation.tactile": list(tactile.astype(np.float32)),
            }
        )

        out_path = self._data_path(episode_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    def append_episode(
        self,
        data_dict: dict,
        *,
        episode_idx: int,
        episode_path: Path | None = None,
    ) -> None:
        if self.features is None:
            self.features = build_lerobot_features(data_dict, use_videos=self.use_videos)

        task = self.default_task
        timestamps = np.asarray(data_dict["timestamps"])
        state = merge_robot(data_dict["state_30hz"], history=False)
        action = merge_robot(data_dict["action_30hz"], history=False)
        length = int(state.shape[0])
        tactile = merge_tactile(data_dict["tactile"]).reshape(length, -1)

        self._write_episode_parquet(
            episode_idx,
            timestamps=timestamps,
            state=state,
            action=action,
            tactile=tactile,
            task=task,
        )

        if self.use_videos:
            for bundle in CAMERA_BUNDLE_ORDER:
                frames = np.asarray(data_dict["camera"][bundle][CAMERA_STREAM], dtype=np.uint8)
                self._encode_camera_video(frames, self._video_path(episode_idx, camera_feature_key(bundle)))

        stats_input = {
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "observation.tactile": tactile.astype(np.float32),
        }
        non_video_features = {
            k: v for k, v in self.features.items() if v.get("dtype") not in ("video", "image")
        }
        ep_stats = compute_episode_stats(stats_input, non_video_features)
        if self.use_videos:
            for bundle in CAMERA_BUNDLE_ORDER:
                key = camera_feature_key(bundle)
                frames = np.asarray(data_dict["camera"][bundle][CAMERA_STREAM], dtype=np.uint8)
                ep_stats[key] = _video_stats_from_frames(frames)

        self.episode_records.append(
            {
                "episode_index": episode_idx,
                "tasks": [task],
                "length": length,
            }
        )
        self.stats_records.append(
            {
                "episode_index": episode_idx,
                "stats": serialize_dict(ep_stats),
            }
        )

        self.total_frames += length
        self.total_episodes += 1

    def finalize(self) -> None:
        if self.features is None:
            return

        video_keys = [camera_feature_key(b) for b in CAMERA_BUNDLE_ORDER] if self.use_videos else []
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": len(self.task_to_index),
            "total_videos": self.total_episodes * len(video_keys),
            "total_chunks": max(1, episode_chunk(max(0, self.total_episodes - 1), self.chunks_size) + 1),
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "splits": {"train": f"0:{self.total_episodes}"},
            "data_path": LEGACY_DATA_PATH,
            "video_path": LEGACY_VIDEO_PATH if self.use_videos else None,
            "features": self.features,
        }

        self._meta_dir.mkdir(parents=True, exist_ok=True)
        with open(self._meta_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        tasks_records = [
            {"task_index": idx, "task": task}
            for task, idx in sorted(self.task_to_index.items(), key=lambda x: x[1])
        ]
        _write_jsonl(self._meta_dir / "tasks.jsonl", tasks_records)
        _write_jsonl(self._meta_dir / "episodes.jsonl", self.episode_records)
        _write_jsonl(self._meta_dir / "episodes_stats.jsonl", self.stats_records)
