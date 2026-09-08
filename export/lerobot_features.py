from __future__ import annotations

import numpy as np

from export.layout import CAMERA_BUNDLE_ORDER, CAMERA_STREAM
from export.merge import merge_robot, merge_tactile

CODEBASE_VERSION = "v2.1"
LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def camera_feature_key(bundle: str) -> str:
    return f"observation.images.{bundle}"


def build_lerobot_features(data_dict: dict, *, use_videos: bool = True) -> dict:
    camera = data_dict["camera"]
    tactile = data_dict["tactile"]
    state = data_dict["state_30hz"]

    ref_rgb = np.asarray(camera[CAMERA_BUNDLE_ORDER[0]]["rgb"])
    h, w = int(ref_rgb.shape[1]), int(ref_rgb.shape[2])

    tactile_merged = merge_tactile(tactile)
    tactile_shape = [int(np.prod(tactile_merged.shape[1:]))]

    state_dim = int(merge_robot(state, history=False).shape[-1])
    action_dim = int(merge_robot(data_dict["action_30hz"], history=False).shape[-1])

    features = {
        **DEFAULT_FEATURES,
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "observation.tactile": {"dtype": "float32", "shape": tactile_shape, "names": None},
    }

    if use_videos:
        for bundle in CAMERA_BUNDLE_ORDER:
            features[camera_feature_key(bundle)] = {
                "dtype": "video",
                "shape": [3, h, w],
                "names": ["channels", "height", "width"],
            }

    return features


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size
