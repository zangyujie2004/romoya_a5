from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm
from termcolor import colored

from task.tasks import Task
from pipeline.align import build_master_timeline
from pipeline.dagger import (
    apply_dagger_postprocess,
    dagger_enabled,
    episode_frame_count,
    min_frames_required,
    resolve_gap_threshold_ms,
    should_apply_event_filter,
)
from pipeline.loader import load_data
from pipeline.process import process_episode_bundles
from pipeline.sync import sync_episode_modalities
from vis import run_vis, vis_robot_eef
from vis.state_action_vis import DEFAULT_GAP_THRESHOLD_MS
from export.meta import build_episode_meta, save_meta
from export.factory import create_exporter, replay_buffer_path
from export.zarr_writer import ZarrWriter
from normalize.build import build_and_save_normalizer
from check.zarr_check import validate_camera_base_remove_hand

def load_config(config_path):
    path = Path(config_path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return output_path

def process_one_episode(episode_path, cfg):
    episode_path = Path(episode_path)
    try:
        # load: streams
        stream_cfg = cfg["streams"]
        camera_dict, tactile_dict, robot_dict, event_dict = load_data(episode_path, stream_cfg)
        # sync: timestamps
        master_clock = cfg["sync"]["master_clock"]
        synced_timestamps = sync_episode_modalities(camera_dict, tactile_dict, robot_dict, event_dict, master_clock)
        # align: task window
        align_cfg = cfg["align"]
        gap_threshold_ms = resolve_gap_threshold_ms(cfg)
        timeline = build_master_timeline(
            robot_dict,
            synced_timestamps,
            master_clock,
            align_cfg,
            gap_threshold_ms=gap_threshold_ms,
        )
        if align_cfg["task_vis"]:
            save_path = episode_path / "vis" / "robot_eef.png"
            vis_robot_eef(robot_dict, task_window=[timeline["task_start_time"], timeline["task_end_time"]], save_path=save_path)
        # bundle process
        process_cfg = cfg["process"]
        data_dict = process_episode_bundles(camera_dict, tactile_dict, robot_dict, event_dict, synced_timestamps, timeline, process_cfg)
        if dagger_enabled(cfg):
            data_dict = apply_dagger_postprocess(
                data_dict,
                cfg,
                apply_event_filter=should_apply_event_filter(episode_path, cfg),
            )
            if episode_frame_count(data_dict) < min_frames_required(cfg):
                print(f"skip episode {episode_path}: frame count {episode_frame_count(data_dict)} < min_frames {min_frames_required(cfg)}")
                return None
        # vis
        gap_threshold_ms = resolve_gap_threshold_ms(cfg) or DEFAULT_GAP_THRESHOLD_MS
        run_vis(data_dict, episode_path, cfg["vis"], gap_threshold_ms=gap_threshold_ms)
        return data_dict
    except Exception as e:
        print(f"Error processing episode {episode_path}: {e}")
        return None

def _process_episode_task(item):
    episode_path, cfg = item
    return process_one_episode(episode_path, cfg)

def run_episodes(episode_list, cfg, *, parallel=False, num_workers=1):
    if parallel:
        work_items = [(ep, cfg) for ep in episode_list]
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for i, data in enumerate(ex.map(_process_episode_task, work_items)):
                yield i, episode_list[i], data
    else:
        for i, episode_path in enumerate(episode_list):
            yield i, episode_path, process_one_episode(episode_path, cfg)

if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_config(config_path)
    # task io
    task = Task(cfg["io"])
    save_config(cfg, task.config_path)
    if task.verbose:
        task.log_summary()

    export_cfg = cfg["export"]
    export_path = replay_buffer_path(task.output_path, export_cfg)
    exporter = create_exporter(export_cfg, task.output_path, cfg=cfg)

    ok = 0
    epi_sample = None
    total = len(task.episode_list)

    print(colored("episode process", "green"))
    acc_cfg = cfg["accelerate"]
    for i, episode_path, data in tqdm(
        run_episodes(
            task.episode_list,
            cfg,
            parallel=acc_cfg["enabled"],
            num_workers=acc_cfg["num_workers"],
        ),
        total=total,
        desc="episodes",
        unit="ep",
    ):
        if data is None:
            continue
        ok += 1
        if epi_sample is None:
            epi_sample = data
        if exporter is not None:
            exporter.append_episode(data, episode_idx=i, episode_path=episode_path)

    print(f"done: {ok}/{total} episodes")

    if epi_sample is None:
        print("skip meta/export: no successful episodes")
    else:
        rh_flags = None
        if isinstance(exporter, ZarrWriter):
            rh_flags = list(exporter.base_remove_hand_flags)
        meta = build_episode_meta(
            epi_sample,
            cfg=cfg,
            num_episodes=ok,
            base_remove_hand_per_episode=rh_flags,
        )
        save_meta(meta, task.meta_path)
        print(f"saved: {task.meta_path}")
        if rh_flags is not None:
            present = sum(1 for f in rh_flags if f == "present")
            none_c = sum(1 for f in rh_flags if f == "none")
            print(f"base_remove_hand: present={present} none={none_c}")
        if exporter is not None:
            print(colored("finalize", "green"))
            exporter.finalize()
            print(f"saved: {export_path}")
            if isinstance(exporter, ZarrWriter) and exporter.has_camera_base_remove_hand:
                validate_camera_base_remove_hand(task.output_path, meta=meta)
        normalize_cfg = cfg.get("normalize") or {}
        if normalize_cfg.get("enabled", False) and exporter is not None:
            print(colored("normalizer", "green"))
            normalizer_path = build_and_save_normalizer(task.output_path, normalize_cfg)
            print(f"saved: {normalizer_path}")

