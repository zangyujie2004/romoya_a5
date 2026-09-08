import os
import pickle
from pathlib import Path

import numpy as np

from pipeline.tools.decode_lossless_video import read_video_array
from vis.robot_vis import vis_gripper_vq
from task.streams import (
    CAMERA_BUNDLES,
    CAMERA_STREAMS,
    RGB_REMOVE_HAND_BUNDLE,
    RGB_REMOVE_HAND_STREAM,
    RGB_REMOVE_HAND_SUFFIX,
    ROBOT_BUNDLES,
    ROBOT_STREAMS,
    TACTILE_BUNDLES,
    TACTILE_STREAMS,
    camera_stream_suffix,
    resolve_impainting_mode,
    selected_streams,
)

VIDEO_BACKEND = os.environ.get("PROMETHEUS_VIDEO_BACKEND", "auto")
VIDEO_GPU_ID = int(os.environ.get("PROMETHEUS_VIDEO_GPU_ID", "1"))

def _load_stream(data_path, bundle, suffix, kind, extra=None):
    if kind == "pkl":
        if bundle in extra and "data" in extra[bundle]:
            return np.array(extra[bundle]["data"])
        value = extra[suffix]
        try:
            return np.array(value)
        except (ValueError, TypeError):
            rows = [np.full(7, np.nan) if x is None else np.asarray(x) for x in value]
            return np.array(rows)
    video_path = os.path.join(data_path, bundle + suffix)
    backend = VIDEO_BACKEND
    if backend == "gpu" and kind != "rgb":
        backend = "sw"
    return read_video_array(
        video_path,
        kind=kind,
        backend=backend,
        gpu_id=VIDEO_GPU_ID,
        # Pipeline / zarr / training contract: camera color is RGB.
        color_order="rgb",
    )

def _load_modality_bundle_streams(
    data_path: str,
    bundle: str,
    stream_names: list[str],
    registry: dict[str, dict[str, str]],
    extra=None,
    bundle_cfg: dict | None = None,
) -> dict[str, np.ndarray]:
    streams = {}
    for name in stream_names:
        spec = registry[name]
        if registry is CAMERA_STREAMS:
            suffix = camera_stream_suffix(name, spec["suffix"], bundle_cfg)
        else:
            suffix = spec["suffix"]
        streams[name] = _load_stream(data_path, bundle, suffix, spec["kind"], extra=extra)
    return streams


def _apply_base_remove_hand(
    image_data_path: Path,
    bundle: str,
    streams: dict[str, np.ndarray],
    bundle_cfg: dict | None,
) -> None:
    """Mutate camera streams for base_0 remove-hand according to impainting mode.

    - off: no-op
    - replace: overwrite rgb with remove-hand (missing file → fail)
    - dual: load remove-hand into rgb_remove_hand when present
    """
    if bundle != RGB_REMOVE_HAND_BUNDLE:
        return
    mode = resolve_impainting_mode(bundle_cfg)
    if mode == "off":
        return

    remove_path = image_data_path / f"{bundle}{RGB_REMOVE_HAND_SUFFIX}"
    if mode == "replace":
        if not remove_path.is_file():
            raise FileNotFoundError(
                f"impainting=replace requires {remove_path}"
            )
        remove_rgb = _load_stream(
            str(image_data_path), bundle, RGB_REMOVE_HAND_SUFFIX, "rgb"
        )
        rgb = streams.get("rgb")
        if rgb is not None and len(remove_rgb) != len(rgb):
            raise ValueError(
                f"remove-hand length mismatch for {bundle}: "
                f"remove={len(remove_rgb)} vs rgb={len(rgb)} path={remove_path}"
            )
        streams["rgb"] = remove_rgb
        print(f"[impainting] replace: using {remove_path.name} as {bundle}/rgb")
        return

    # dual
    if not remove_path.is_file():
        return
    remove_rgb = _load_stream(
        str(image_data_path), bundle, RGB_REMOVE_HAND_SUFFIX, "rgb"
    )
    rgb = streams.get("rgb")
    if rgb is None:
        raise ValueError(f"impainting=dual requires {bundle}/rgb before remove-hand")
    if len(remove_rgb) != len(rgb):
        raise ValueError(
            f"remove-hand length mismatch for {bundle}: "
            f"remove={len(remove_rgb)} vs rgb={len(rgb)} path={remove_path}"
        )
    streams[RGB_REMOVE_HAND_STREAM] = remove_rgb
    print(f"[impainting] dual: loaded {remove_path.name} -> {bundle}/{RGB_REMOVE_HAND_STREAM}")

def _estimate_frequency_hz_from_ms(timestamps: np.ndarray) -> float | None:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size < 2:
        return None
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    # This project uses millisecond timestamps in visualization and sync.
    return float(1000.0 / np.median(dt))

def _discretize_gripper(gripper: np.ndarray, *, threshold: float) -> np.ndarray:
    """Per-episode: max->open, min->close; assign by threshold."""
    gripper = np.asarray(gripper, dtype=np.float64)
    open_val = float(np.max(gripper))
    close_val = float(np.min(gripper))
    return np.where(gripper >= threshold, open_val, close_val)


def _apply_gripper_vq(
    robot_data_dict: dict[str, dict[str, np.ndarray]],
    gripper_vq_cfg: dict,
    *,
    episode_path: Path | None = None,
) -> None:
    threshold = float(gripper_vq_cfg.get("threshold", 0.079))
    draw_vis = bool(gripper_vq_cfg.get("vis", False))
    gripper_before: dict[str, np.ndarray] = {}

    for arm in ROBOT_BUNDLES:
        joint = np.asarray(robot_data_dict[arm]["joint"], dtype=np.float64)
        raw = joint[:, 6].copy()
        gripper_before[arm] = raw
        joint[:, 6] = _discretize_gripper(raw, threshold=threshold)
        robot_data_dict[arm]["joint"] = joint

    if draw_vis and episode_path is not None:
        vis_gripper_vq(
            robot_data_dict,
            gripper_before,
            threshold=threshold,
            save_path=episode_path / "vis" / "gripper_vq.png",
        )


def _print_bundle_frequencies(modality: str, data_dict: dict[str, dict[str, np.ndarray]]) -> None:
    for bundle, payload in data_dict.items():
        timestamps = payload.get("timestamps", np.array([]))
        freq = _estimate_frequency_hz_from_ms(timestamps)
        if freq is None:
            print(f"[frequency] {modality}/{bundle}: unavailable (len={len(timestamps)})")
        else:
            print(f"[frequency] {modality}/{bundle}: {freq:.2f} Hz (len={len(timestamps)})")

def load_data(episode_path, stream_cfg):
    episode_path = Path(episode_path)
    image_data_path = episode_path / "camera"
    tactile_data_path = episode_path / "tactile"
    robot_data_path = episode_path / "robot"
    event_data_path = episode_path / "event"
    # camera
    camera_cfg = stream_cfg.get("camera") or {}
    camera_selected = selected_streams(camera_cfg, CAMERA_STREAMS, CAMERA_BUNDLES)
    camera_data_dict = {}
    if camera_selected:
        with open(image_data_path / "timestamp.pkl", "rb") as f:
            cam_ts = pickle.load(f)
        for bundle, stream_names in camera_selected.items():
            bundle_cfg = camera_cfg.get(bundle)
            streams = _load_modality_bundle_streams(
                str(image_data_path),
                bundle,
                stream_names,
                CAMERA_STREAMS,
                bundle_cfg=bundle_cfg,
            )
            _apply_base_remove_hand(image_data_path, bundle, streams, bundle_cfg)
            camera_data_dict[bundle] = {
                "timestamps": np.asarray(cam_ts[bundle], dtype=np.int64),
                **streams,
            }
    # tactile
    tactile_selected = selected_streams(stream_cfg.get("tactile"), TACTILE_STREAMS, TACTILE_BUNDLES)
    tactile_data_dict = {}
    if tactile_selected:
        with open(tactile_data_path / "timestamp.pkl", "rb") as f:
            tac_ts = pickle.load(f)
        with open(tactile_data_path / "tactile_pointcloud_dict.pkl", "rb") as f:
            tac_pc = pickle.load(f)
        for bundle, stream_names in tactile_selected.items():
            tactile_data_dict[bundle] = {
                "timestamps": np.asarray(tac_ts[bundle], dtype=np.int64),
                **_load_modality_bundle_streams(
                    str(tactile_data_path),
                    bundle,
                    stream_names,
                    TACTILE_STREAMS,
                    extra=tac_pc,
                ),
            }
    # robot
    robot_data_dict = {}
    with open(robot_data_path / "robot_state_dict.pkl", "rb") as f:
        robot_data = pickle.load(f)
    for bundle in ROBOT_BUNDLES:
        robot_data_dict[bundle] = {
            "timestamps": np.array(robot_data[bundle]["timestamps"]),
            **_load_modality_bundle_streams(
                str(robot_data_path),
                bundle,
                list(ROBOT_STREAMS),
                {name: {"suffix": name, "kind": "pkl"} for name in ROBOT_STREAMS},
                extra=robot_data[bundle],
            ),
        }
    gripper_vq_cfg = stream_cfg.get("gripper_vq") or {}
    if gripper_vq_cfg.get("enabled"):
        _apply_gripper_vq(robot_data_dict, gripper_vq_cfg, episode_path=episode_path)
    # event
    event_data_dict = {}
    with open(event_data_path / "event_dict.pkl", "rb") as f:
        event_raw = pickle.load(f)
    event_data_dict["event"] = {
        "timestamps": np.asarray(event_raw["timestamps"], dtype=np.int64),
        "data": np.asarray(event_raw["data"]),
    }

    # frequency
    if stream_cfg["verbose_freq"]:
        _print_bundle_frequencies("camera", camera_data_dict)
        _print_bundle_frequencies("tactile", tactile_data_dict)
        _print_bundle_frequencies("robot", robot_data_dict)
        _print_bundle_frequencies("event", event_data_dict)
    
    return camera_data_dict, tactile_data_dict, robot_data_dict, event_data_dict
