from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from task.streams import CAMERA_BUNDLES


def vis_camera(
    data_dict: dict[str, Any],
    save_dir: str | Path,
    step: int = 5,
    bundles: list[str] | None = None,
) -> None:
    """Save camera mosaic frames from processed ``data_dict`` (resize already applied)."""
    aligned_camera = data_dict["camera"]
    timestamps = np.asarray(data_dict["timestamps"], dtype=np.int64)

    if bundles is None:
        bundles = list(CAMERA_BUNDLES)
    bundles = [b for b in bundles if b in aligned_camera and "rgb" in aligned_camera[b]]
    if not bundles:
        print("vis_camera: no rgb streams found, skip")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamps_sec = timestamps / 1000.0
    timestamps_rel = timestamps_sec - timestamps_sec[0]
    rgb_arrays = [aligned_camera[b]["rgb"] for b in bundles]
    count = 0
    for i in range(0, len(timestamps), step):
        panels = []
        for bundle, imgs in zip(bundles, rgb_arrays):
            # Camera arrays in data_dict are RGB; OpenCV imwrite expects BGR.
            img = np.asarray(imgs[i])
            if img.ndim == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                bundle,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            panels.append(bgr)
        mosaic = np.concatenate(panels, axis=1)
        t_str = f"{timestamps_rel[i]:.3f}".replace(".", "_")
        cv2.imwrite(str(save_dir / f"{t_str}.png"), mosaic)
        count += 1
    print(f"vis_camera: saved {count} frames (step={step}) -> {save_dir}")
