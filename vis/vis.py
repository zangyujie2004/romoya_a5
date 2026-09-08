from __future__ import annotations

from pathlib import Path
from typing import Any

from vis.camera_vis import vis_camera
from vis.state_action_vis import vis_state_action
from vis.tactile_vis import vis_tactile
from vis.video_vis import vis_episode_video


def run_vis(
    data_dict: dict[str, Any],
    episode_path: str | Path,
    vis_cfg: dict[str, Any],
    *,
    gap_threshold_ms: float | None = None,
) -> None:
    from vis.state_action_vis import DEFAULT_GAP_THRESHOLD_MS

    gap_threshold_ms = (
        float(gap_threshold_ms)
        if gap_threshold_ms is not None
        else DEFAULT_GAP_THRESHOLD_MS
    )
    vis_dir = Path(episode_path) / "vis"
    if vis_cfg["camera"]["enabled"]:
        vis_camera(data_dict, vis_dir / "camera", step=vis_cfg["camera"]["step"])
    if vis_cfg["tactile"]["enabled"]:
        vis_tactile(data_dict, vis_dir / "tactile", step=vis_cfg["tactile"]["step"])
    if vis_cfg["state_action"]["enabled"]:
        vis_state_action(data_dict, vis_dir, rate_key="30hz", gap_threshold_ms=gap_threshold_ms)
        vis_state_action(data_dict, vis_dir, rate_key="120hz", gap_threshold_ms=gap_threshold_ms)
    if vis_cfg["video"]["enabled"]:
        vis_episode_video(data_dict, vis_dir / "episode_30hz.mp4", fps=vis_cfg["video"]["fps"])
