from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from normalize.normalizer import (
    ROBOT_STORAGE_KEYS,
    TACTILE_STORAGE_KEY,
    LinearNormalizer,
    get_robot_merged_normalizer,
    get_tactile_merged_normalizer,
)

DEFAULT_OUTPUT_NAME = "normalizer.pth"


def load_meta(run_dir: Path) -> dict:
    with open(run_dir / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def open_storage_data(run_dir: Path, meta: dict | None = None):
    meta = meta or load_meta(run_dir)
    storage = meta["storage"]
    path = run_dir / storage["path"]
    fmt = str(storage.get("format", "")).lower()

    if fmt.endswith("zarr") or path.name.endswith(".zarr"):
        root = zarr.open_group(str(path), mode="r")
        return root["data"]

    if fmt.endswith("hdf5") or path.suffix in {".h5", ".hdf5"}:
        import h5py

        file = h5py.File(path, "r")
        return file["data"]

    raise ValueError(f"unsupported storage format for normalizer: {fmt or path}")


def _normalize_cfg(cfg: dict | None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    output_range = cfg.get("output_range", [-1.0, 1.0])
    tactile_cfg = dict(cfg.get("tactile") or {})
    return {
        "output_min": float(output_range[0]),
        "output_max": float(output_range[1]),
        "range_eps": float(cfg.get("range_eps", 1e-7)),
        "tactile_channels_per_point": int(tactile_cfg.get("channels_per_point", 6)),
        "tactile_chunk_size": int(tactile_cfg.get("chunk_size", 512)),
        "tactile_num_bundles": int(tactile_cfg.get("num_bundles", 4)),
        "robot_keys": tuple(cfg.get("robot_keys", ROBOT_STORAGE_KEYS)),
        "tactile_key": str(cfg.get("tactile_key", TACTILE_STORAGE_KEY)),
    }


def build_normalizer_from_run_dir(run_dir: Path | str, cfg: dict | None = None) -> LinearNormalizer:
    """Fit normalizer on exported replay buffer. Camera is intentionally excluded."""
    run_dir = Path(run_dir)
    opts = _normalize_cfg(cfg)
    meta = load_meta(run_dir)
    data = open_storage_data(run_dir, meta)

    normalizer = LinearNormalizer()
    for key in opts["robot_keys"]:
        if key not in data:
            raise KeyError(f"missing storage key '{key}' in replay buffer")
        arr = np.asarray(data[key][:], dtype=np.float32)
        normalizer[key] = get_robot_merged_normalizer(
            arr,
            output_max=opts["output_max"],
            output_min=opts["output_min"],
            range_eps=opts["range_eps"],
        )

    tactile_key = opts["tactile_key"]
    if tactile_key not in data:
        raise KeyError(f"missing storage key '{tactile_key}' in replay buffer")
    tactile_arr = data[tactile_key]
    merged_channels = int(tactile_arr.shape[-1])
    channels_per_point = opts["tactile_channels_per_point"]
    if merged_channels % channels_per_point != 0:
        raise ValueError(
            f"tactile last dim {merged_channels} is not divisible by channels_per_point={channels_per_point}"
        )
    num_bundles = merged_channels // channels_per_point
    normalizer[tactile_key] = get_tactile_merged_normalizer(
        tactile_arr,
        num_bundles=num_bundles,
        channels_per_point=channels_per_point,
        chunk_size=opts["tactile_chunk_size"],
        output_max=opts["output_max"],
        output_min=opts["output_min"],
        range_eps=opts["range_eps"],
    )
    return normalizer


def save_normalizer(normalizer: LinearNormalizer, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalizer, path)
    return path


def load_normalizer(path: Path | str) -> LinearNormalizer:
    return torch.load(path, weights_only=False)


def build_and_save_normalizer(
    run_dir: Path | str,
    cfg: dict | None = None,
    *,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    run_dir = Path(run_dir)
    cfg = dict(cfg or {})
    output_path = run_dir / str(cfg.get("output", output_name))
    normalizer = build_normalizer_from_run_dir(run_dir, cfg)
    save_normalizer(normalizer, output_path)
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build normalizer.pth from exported replay buffer.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="processed run directory containing meta.json and replay buffer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .pth path (default: <run-dir>/normalizer.pth)",
    )
    args = parser.parse_args()

    output_path = args.output or (args.run_dir / DEFAULT_OUTPUT_NAME)
    normalizer = build_normalizer_from_run_dir(args.run_dir)
    save_normalizer(normalizer, output_path)
    keys = list(normalizer.params_dict.keys())
    print(f"saved normalizer: {output_path}")
    print(f"  keys: {keys}")


if __name__ == "__main__":
    main()
