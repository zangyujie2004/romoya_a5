import numpy as np

def build_timestamp_registry(camera_data_dict, tactile_data_dict, robot_data_dict, event_data_dict):
    return {
        modality: {bundle: np.asarray(data["timestamps"], dtype=np.int64) for bundle, data in bundles.items()}
        for modality, bundles in (
            ("camera", camera_data_dict),
            ("tactile", tactile_data_dict),
            ("robot", robot_data_dict),
            ("event", event_data_dict),
        )
    }

def _get_ts(ts_registry: dict, clock: str) -> np.ndarray:
    modality, bundle = clock.split("/", 1)
    return ts_registry[modality][bundle]

def sync_and_truncate_timestamps(ts_registry, master_clock):
    all_ts = [ts for bundles in ts_registry.values() for ts in bundles.values()]
    max_start = max(ts[0] for ts in all_ts)
    master_ts = _get_ts(ts_registry, master_clock)
    master_modality, master_bundle = master_clock.split("/", 1)

    begin_idx = int(np.searchsorted(master_ts, max_start, side="left"))
    if begin_idx >= len(master_ts):
        raise ValueError(f"max_start {max_start} is after {master_clock} timeline")
    begin = int(master_ts[begin_idx])
    start_indices = {}
    for modality, bundles in ts_registry.items():
        start_indices[modality] = {}
        for bundle, ts in bundles.items():
            if modality == master_modality and bundle == master_bundle:
                start_indices[modality][bundle] = begin_idx
            else:
                start_indices[modality][bundle] = int(np.searchsorted(ts, begin, side="left"))

    min_end = min(int(ts[-1]) for ts in all_ts)
    end_idx = int(np.searchsorted(master_ts, min_end, side="right")) - 1
    if end_idx < 0:
        raise ValueError(f"min_end {min_end} is before {master_clock} timeline")
    end = int(master_ts[end_idx])
    end_indices = {}
    synced_timestamps = {}
    for modality, bundles in ts_registry.items():
        end_indices[modality] = {}
        synced_timestamps[modality] = {}
        for bundle, ts in bundles.items():
            s = start_indices[modality][bundle]
            e = int(np.searchsorted(ts, end, side="right"))
            end_indices[modality][bundle] = e
            synced_timestamps[modality][bundle] = ts[s:e]

    return synced_timestamps, start_indices, end_indices

def truncate_modality_dict(data_dict, start_indices, end_indices, modality):
    for bundle, payload in data_dict.items():
        s, e = start_indices[modality][bundle], end_indices[modality][bundle]
        for key, arr in payload.items():
            payload[key] = arr[s:e]

def sync_episode_modalities(camera_dict, tactile_dict, robot_dict, event_dict, master_clock: str) -> dict:
    timestamp_registry = build_timestamp_registry(camera_dict, tactile_dict, robot_dict, event_dict)
    synced_timestamps, start_indices, end_indices = sync_and_truncate_timestamps(
        timestamp_registry, master_clock
    )
    truncate_modality_dict(camera_dict, start_indices, end_indices, "camera")
    truncate_modality_dict(tactile_dict, start_indices, end_indices, "tactile")
    truncate_modality_dict(robot_dict, start_indices, end_indices, "robot")
    truncate_modality_dict(event_dict, start_indices, end_indices, "event")
    return synced_timestamps
