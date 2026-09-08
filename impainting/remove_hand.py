#!/usr/bin/env python3
"""Final quality-first universal glove removal with conservative identity tracking."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from _grounded_sam import GroundedSAM, encode_compatible_mp4, filter_boxes
from _propainter_utils import (
    find_propainter_result,
    raise_nofile_limit,
    run_checked,
    write_status,
)
from _mask_utils import open_writer, red_protection, retain_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--propainter-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="base_0_rgb_remove_hand.mp4")
    parser.add_argument("--dino-dir", default="grounding-dino-base")
    parser.add_argument("--sam-dir", default="sam-vit-huge")
    parser.add_argument(
        "--prompt",
        default="rubber glove . gloved human hand . human hand . human forearm .",
    )
    parser.add_argument("--box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--roi-bottom-ratio", type=float, default=0.38)
    parser.add_argument("--dino-interval", type=int, default=1)
    parser.add_argument("--box-expand-ratio", type=float, default=0.20)
    parser.add_argument("--min-box-margin", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--mask-dilation", type=int, default=5)
    parser.add_argument("--temporal-flow-width", type=int, default=320)
    parser.add_argument(
        "--mask-mode", choices=("strict_color", "color_shadow"),
        default="strict_color",
    )
    parser.add_argument("--object-protect-mask", default="")
    parser.add_argument("--neighbor-length", type=int, default=8)
    parser.add_argument("--ref-stride", type=int, default=5)
    parser.add_argument("--subvideo-length", type=int, default=20)
    parser.add_argument("--raft-iter", type=int, default=10)
    parser.add_argument("--propainter-mask-dilation", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--keep-mask-video", action="store_true")
    return parser.parse_args()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mask_to_boxes(
    mask: np.ndarray,
    *,
    expand_ratio: float,
    min_margin: int,
    min_area: int,
) -> np.ndarray:
    """Return expanded component boxes suitable for current-frame SAM prompts."""
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: list[list[float]] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        margin_x = max(min_margin, int(round(w * expand_ratio)))
        margin_y = max(min_margin, int(round(h * expand_ratio)))
        boxes.append(
            [
                float(max(0, x - margin_x)),
                float(max(0, y - margin_y)),
                float(min(width - 1, x + w + margin_x)),
                float(min(height - 1, y + h + margin_y)),
            ]
        )
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def merge_boxes(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Prefer DINO boxes; retain non-overlapping tracked boxes for missed gloves."""
    if primary.size == 0:
        return fallback
    if fallback.size == 0:
        return primary
    merged = [box for box in primary]
    for candidate in fallback:
        x1 = np.maximum(primary[:, 0], candidate[0])
        y1 = np.maximum(primary[:, 1], candidate[1])
        x2 = np.minimum(primary[:, 2], candidate[2])
        y2 = np.minimum(primary[:, 3], candidate[3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        candidate_area = max(1.0, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
        primary_area = np.maximum(
            1.0, (primary[:, 2] - primary[:, 0]) * (primary[:, 3] - primary[:, 1])
        )
        union = candidate_area + primary_area - intersection
        iou = intersection / np.maximum(union, 1.0)
        if float(iou.max(initial=0.0)) < 0.25:
            merged.append(candidate)
    return np.asarray(merged, dtype=np.float32)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments((mask > 0).astype(np.uint8))
    if moments["m00"] <= 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def recovery_reasons(
    previous: np.ndarray | None,
    current: np.ndarray,
    frame: np.ndarray,
    semantic: np.ndarray,
    *,
    min_area: int,
) -> list[str]:
    """Return confidence failures that require same-frame DINO + SAM recovery."""
    if previous is None:
        return []
    previous_area = int(np.count_nonzero(previous))
    current_area = int(np.count_nonzero(current))
    if previous_area < min_area:
        return []
    reasons: list[str] = []
    if current_area < max(min_area, previous_area * 0.35):
        reasons.append("area_drop")
    if current_area > previous_area * 2.8:
        reasons.append("area_jump")
    previous_center = mask_centroid(previous)
    current_center = mask_centroid(current)
    if previous_center is not None and current_center is not None:
        displacement = np.hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        if displacement > 80:
            reasons.append("centroid_jump")
    if current_area >= min_area:
        support = cv2.dilate(previous, np.ones((31, 31), np.uint8), iterations=1) > 0
        overlap = int(np.count_nonzero((current > 0) & support))
        if overlap / max(1, current_area) < 0.15:
            reasons.append("low_temporal_overlap")
    # Colour is a confidence signal only; the final mask still comes from SAM.
    green_candidate = strong_green_mask(frame) & (semantic > 0)
    green_area = int(np.count_nonzero(green_candidate))
    if green_area >= max(200, min_area * 2):
        covered = cv2.dilate(current, np.ones((5, 5), np.uint8), iterations=1) > 0
        coverage = int(np.count_nonzero(green_candidate & covered)) / green_area
        if coverage < 0.45:
            reasons.append("low_green_coverage")

    return reasons


def strong_green_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(frame)
    b16, g16, r16 = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)
    return (
        (h >= 30) & (h <= 105) & (s >= 42) & (v >= 30)
        & (g16 >= r16 + 5) & (g16 >= b16 - 1)
    )


def glove_identity_seed(
    frame: np.ndarray,
    previous: np.ndarray | None,
    *,
    min_area: int,
) -> np.ndarray:
    """Keep green evidence that belongs to a border-entering glove track.

    Task objects are deliberately not classified by name. A pixel may seed a
    glove only at a side/bottom entry band or near the preceding confirmed
    glove mask. This makes uncertain contact pixels source-preserving.
    """
    green = strong_green_mask(frame)
    height, width = green.shape
    entry = np.zeros_like(green)
    side = max(1, int(width * 0.28))
    bottom = max(1, int(height * 0.18))
    entry[:, :side] = True
    entry[:, width - side:] = True
    entry[height - bottom:, :] = True
    gate = entry
    if previous is not None and np.any(previous):
        continuity = cv2.dilate(
            (previous > 0).astype(np.uint8),
            np.ones((25, 25), np.uint8),
            iterations=1,
        ) > 0
        gate = entry | continuity
    seed = (green & gate).astype(np.uint8) * 255
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    return retain_components(seed, max(16, min_area // 4))


def filter_identity_boxes(
    boxes: np.ndarray,
    identity_seed: np.ndarray,
    *,
    min_pixels: int = 16,
) -> tuple[np.ndarray, int]:
    """Reject semantic boxes that do not overlap confirmed glove identity."""
    if boxes.size == 0:
        return boxes, 0
    height, width = identity_seed.shape
    accepted: list[np.ndarray] = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 > x1 and y2 > y1:
            overlap = int(np.count_nonzero(identity_seed[y1:y2, x1:x2]))
            if overlap >= min_pixels:
                accepted.append(box)
    result = np.asarray(accepted, dtype=np.float32)
    if not accepted:
        result = np.zeros((0, 4), dtype=np.float32)
    return result, int(len(boxes) - len(accepted))


def strong_entry_component_area(frame: np.ndarray) -> int:
    green = strong_green_mask(frame)
    height, width = green.shape
    entry = np.zeros_like(green)
    side = int(width * 0.28)
    bottom = int(height * 0.18)
    entry[:, :side] = True
    entry[:, width - side:] = True
    entry[height - bottom:, :] = True
    binary = ((green & entry).astype(np.uint8) * 255)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def detect_boxes(
    segmenter: GroundedSAM,
    frame: np.ndarray,
    args: argparse.Namespace,
    timing: dict[str, float],
) -> np.ndarray:
    cuda_sync()
    started = time.perf_counter()
    boxes = segmenter.boxes(frame, args.prompt, args.box_threshold, args.text_threshold)
    boxes = filter_boxes(boxes, frame.shape, args.roi_bottom_ratio)
    cuda_sync()
    timing["dino_s"] += time.perf_counter() - started
    timing["dino_calls"] += 1
    return boxes


def segment_boxes(
    segmenter: GroundedSAM,
    frame: np.ndarray,
    boxes: np.ndarray,
    identity_seed: np.ndarray,
    timing: dict[str, float],
) -> np.ndarray:
    """Run SAM and choose candidates that explain the tracked glove identity."""
    if boxes.size == 0:
        timing["sam_no_box_frames"] += 1
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    cuda_sync()
    started = time.perf_counter()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = segmenter.sam_processor(
        image, input_boxes=boxes[None, :, :].tolist(), return_tensors="pt"
    ).to(segmenter.device)
    with torch.no_grad():
        outputs = segmenter.sam_model(**inputs)
    masks = segmenter.sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0].detach().cpu().numpy()
    scores = outputs.iou_scores[0].detach().cpu().numpy()
    identity = identity_seed > 0
    identity_support = cv2.dilate(
        identity_seed, np.ones((31, 31), np.uint8), iterations=1
    ) > 0
    selected = np.zeros(frame.shape[:2], dtype=bool)
    for box_index in range(masks.shape[0]):
        x1, y1, x2, y2 = boxes[box_index].astype(int)
        box_identity = np.zeros_like(identity)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        box_identity[y1:y2, x1:x2] = identity[y1:y2, x1:x2]
        identity_count = int(np.count_nonzero(box_identity))
        if identity_count < 16:
            continue
        candidate_scores = []
        for candidate_index in range(masks.shape[1]):
            candidate = masks[box_index, candidate_index] > 0
            candidate_area = max(1, int(np.count_nonzero(candidate)))
            identity_recall = (
                np.count_nonzero(candidate & box_identity) / identity_count
            )
            spill_fraction = (
                np.count_nonzero(candidate & ~identity_support) / candidate_area
            )
            candidate_scores.append(
                float(scores[box_index, candidate_index])
                + 0.75 * identity_recall
                - 0.30 * spill_fraction
            )
        best = int(np.argmax(candidate_scores))
        selected |= masks[box_index, best] > 0
    cuda_sync()
    timing["sam_s"] += time.perf_counter() - started
    timing["sam_calls"] += 1
    return selected.astype(np.uint8) * 255


def quality_mask(
    frame: np.ndarray,
    semantic: np.ndarray,
    identity_seed: np.ndarray,
    object_protect: np.ndarray | None,
    *,
    min_area: int,
    dilation: int,
) -> np.ndarray:
    """Recover only SAM components seeded by the confirmed glove track."""
    semantic_bool = semantic > 0
    if object_protect is not None:
        semantic_bool &= object_protect == 0
    core = ((identity_seed > 0) & semantic_bool).astype(np.uint8) * 255
    core = cv2.morphologyEx(
        core, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    core = retain_components(core, max(40, min_area // 2))
    if not np.any(core):
        return np.zeros_like(semantic)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        semantic_bool.astype(np.uint8), 8
    )
    glove = np.zeros_like(semantic)
    seed_floor = max(24, min_area // 4)
    frame_area = semantic.size
    for label in range(1, count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        seeded = int(np.count_nonzero(component & (core > 0)))
        if seeded >= seed_floor and component_area <= frame_area * 0.42:
            glove[component] = 255
    local_limit = cv2.dilate(
        core, np.ones((25, 25), np.uint8), iterations=1
    ) > 0
    glove[~local_limit] = 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    core_near = cv2.dilate(core, np.ones((5, 5), np.uint8), iterations=1) > 0
    rigid = semantic_bool & ~core_near & (
        ((saturation < 45) & (edge > 75))
        | ((value < 22) & (edge > 45))
    )
    protect = red_protection(frame) | rigid
    if object_protect is not None:
        protect |= object_protect > 0
    glove = cv2.morphologyEx(
        glove, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    if dilation > 1:
        kernel_size = dilation if dilation % 2 == 1 else dilation + 1
        glove = cv2.dilate(
            glove, np.ones((kernel_size, kernel_size), np.uint8), iterations=1
        )
    glove[protect] = 0
    return retain_components(glove, max(40, min_area // 2))


def guarded_blend(
    original: np.ndarray, generated: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Blend only in the one-pixel support of mask; outside is exact source."""
    alpha = cv2.GaussianBlur(mask, (3, 3), 0.65).astype(np.float32) / 255.0
    # Generated pixels never modify an unmasked source pixel.
    support = mask > 0
    alpha[~support] = 0.0
    alpha3 = alpha[:, :, None]
    result = np.clip(
        generated.astype(np.float32) * alpha3
        + original.astype(np.float32) * (1.0 - alpha3),
        0,
        255,
    ).astype(np.uint8)
    result[~support] = original[~support]
    return result


def build_masks(
    args: argparse.Namespace,
    mask_dir: Path,
    raw_mask: Path,
    fps: float,
    size: tuple[int, int],
    timing: dict[str, float],
) -> tuple[int, list[int], dict[str, int]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_started = time.perf_counter()
    segmenter = GroundedSAM(args.models_root, args.dino_dir, args.sam_dir, device)
    timing["model_load_s"] += time.perf_counter() - model_started
    video = cv2.VideoCapture(args.input)
    object_video = (
        cv2.VideoCapture(args.object_protect_mask)
        if args.object_protect_mask else None
    )
    if not video.isOpened():
        raise RuntimeError(f"failed to open input: {args.input}")
    if object_video is not None and not object_video.isOpened():
        raise RuntimeError("failed to open object-protection mask")
    previous: np.ndarray | None = None
    frames = 0
    areas: list[int] = []
    diagnostics = defaultdict(int)
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            object_mask = None
            if object_video is not None:
                object_ok, encoded_object = object_video.read()
                if not object_ok:
                    raise RuntimeError(
                        f"object-protection mask ended at frame {frames}"
                    )
                gray_object = cv2.cvtColor(encoded_object, cv2.COLOR_BGR2GRAY)
                object_mask = np.where(
                    gray_object >= 128, 255, 0
                ).astype(np.uint8)
            identity_seed = glove_identity_seed(
                frame, previous, min_area=args.min_area
            )
            if not np.any(identity_seed):
                diagnostics["identity_seed_empty_frames"] += 1
            raw_current_boxes = detect_boxes(segmenter, frame, args, timing)
            if not raw_current_boxes.size:
                diagnostics["dino_empty_frames"] += 1
            current_boxes, rejected = filter_identity_boxes(
                raw_current_boxes, identity_seed
            )
            if rejected:
                diagnostics["identity_rejected_boxes"] += rejected
                diagnostics["identity_rejected_frames"] += 1
            tracked_boxes = (
                mask_to_boxes(
                    previous,
                    expand_ratio=args.box_expand_ratio,
                    min_margin=args.min_box_margin,
                    min_area=args.min_area,
                )
                if previous is not None
                else np.zeros((0, 4), dtype=np.float32)
            )
            if current_boxes.size:
                boxes = merge_boxes(current_boxes, tracked_boxes)
            else:
                boxes = tracked_boxes
                if tracked_boxes.size:
                    diagnostics["fallback_box_frames"] += 1
            semantic = segment_boxes(
                segmenter, frame, boxes, identity_seed, timing
            )
            refine_started = time.perf_counter()
            refined = quality_mask(
                frame, semantic, identity_seed, object_mask,
                min_area=args.min_area, dilation=args.mask_dilation,
            )
            timing["mask_refine_s"] += time.perf_counter() - refine_started
            if not np.any(refined):
                diagnostics["model_mask_empty_frames"] += 1
            if not cv2.imwrite(str(mask_dir / f"{frames:06d}.png"), refined):
                raise RuntimeError(f"failed to write model mask frame {frames}")
            areas.append(int(np.count_nonzero(refined)))
            previous = refined
            frames += 1
            if args.progress_every > 0 and frames % args.progress_every == 0:
                print(
                    f"[green-v9] model-mask progress={frames} "
                    f"dino={int(timing['dino_calls'])} "
                    f"sam={int(timing['sam_calls'])}",
                    flush=True,
                )
    finally:
        video.release()
        if object_video is not None:
            object_video.release()
        del segmenter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if frames == 0:
        raise RuntimeError("decoded zero frames")
    return frames, areas, dict(diagnostics)


def warp_neighbor_mask(
    neighbor_mask: np.ndarray,
    current_gray: np.ndarray,
    neighbor_gray: np.ndarray,
    *,
    flow_width: int,
    engine,
    timing: dict[str, float],
) -> np.ndarray:
    height, width = current_gray.shape
    small_width = min(flow_width, width)
    small_height = max(1, round(height * small_width / width))
    current_small = cv2.resize(current_gray, (small_width, small_height))
    neighbor_small = cv2.resize(neighbor_gray, (small_width, small_height))
    started = time.perf_counter()
    flow = engine.calc(current_small, neighbor_small, None)
    grid_x, grid_y = np.meshgrid(
        np.arange(small_width, dtype=np.float32),
        np.arange(small_height, dtype=np.float32),
    )
    neighbor_small_mask = cv2.resize(
        neighbor_mask, (small_width, small_height),
        interpolation=cv2.INTER_NEAREST,
    )
    warped_small = cv2.remap(
        neighbor_small_mask,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    timing["optical_flow_s"] += time.perf_counter() - started
    return cv2.resize(
        warped_small, (width, height), interpolation=cv2.INTER_NEAREST
    )


def stabilize_masks(
    video_path: str,
    model_mask_dir: Path,
    final_mask_dir: Path,
    frames: int,
    args: argparse.Namespace,
    timing: dict[str, float],
) -> tuple[list[int], dict[str, int]]:
    """Bidirectional model-anchored temporal fill using a three-frame window."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open temporal input: {video_path}")

    def read_item(index: int) -> tuple[np.ndarray, np.ndarray]:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"temporal decode failed at frame {index}")
        mask = cv2.imread(
            str(model_mask_dir / f"{index:06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise RuntimeError(f"missing model mask frame {index}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
        return gray, binary

    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    engine.setUseSpatialPropagation(True)
    diagnostics = defaultdict(int)
    areas: list[int] = []
    previous = None
    try:
        current = read_item(0)
        following = read_item(1) if frames > 1 else None
        for index in range(frames):
            current_gray, current_mask = current
            stable = current_mask.copy()
            if previous is not None and following is not None:
                previous_gray, previous_mask = previous
                following_gray, following_mask = following
                warped_previous = warp_neighbor_mask(
                    previous_mask, current_gray, previous_gray,
                    flow_width=args.temporal_flow_width,
                    engine=engine, timing=timing,
                )
                warped_following = warp_neighbor_mask(
                    following_mask, current_gray, following_gray,
                    flow_width=args.temporal_flow_width,
                    engine=engine, timing=timing,
                )
                agreement = (warped_previous > 0) & (warped_following > 0)
                current_area = int(np.count_nonzero(current_mask))
                if current_area == 0:
                    if np.count_nonzero(agreement) >= args.min_area:
                        stable = agreement.astype(np.uint8) * 255
                        diagnostics["bridged_empty_frames"] += 1
                else:
                    support = cv2.dilate(
                        current_mask, np.ones((21, 21), np.uint8),
                        iterations=1,
                    ) > 0
                    added = agreement & support & (current_mask == 0)
                    if np.any(added):
                        stable[added] = 255
                        diagnostics["temporally_filled_frames"] += 1
                        diagnostics["temporally_added_pixels"] += int(
                            np.count_nonzero(added)
                        )
                    stable = cv2.morphologyEx(
                        stable, cv2.MORPH_CLOSE,
                        np.ones((5, 5), np.uint8), iterations=1,
                    )
            if not cv2.imwrite(
                str(final_mask_dir / f"{index:06d}.png"), stable
            ):
                raise RuntimeError(f"failed to write stable mask frame {index}")
            areas.append(int(np.count_nonzero(stable)))
            previous = current
            current = following
            next_index = index + 2
            following = (
                read_item(next_index) if next_index < frames else None
            )
    finally:
        capture.release()
    return areas, dict(diagnostics)


def write_mask_video(
    mask_dir: Path,
    raw_path: Path,
    frames: int,
    fps: float,
    size: tuple[int, int],
) -> None:
    writer = open_writer(raw_path, fps, size)
    try:
        for index in range(frames):
            mask = cv2.imread(
                str(mask_dir / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                raise RuntimeError(f"missing stable mask frame {index}")
            writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def guarded_composite_video(
    source_path: str,
    generated_path: Path,
    mask_dir: Path,
    raw_output: Path,
    frames: int,
    fps: float,
    size: tuple[int, int],
) -> None:
    source = cv2.VideoCapture(source_path)
    generated = cv2.VideoCapture(str(generated_path))
    writer = open_writer(raw_output, fps, size)
    try:
        for index in range(frames):
            source_ok, original = source.read()
            generated_ok, filled = generated.read()
            if not source_ok or not generated_ok:
                raise RuntimeError(f"composite decode failed at frame {index}")
            mask = cv2.imread(str(mask_dir / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"missing mask frame {index}")
            writer.write(guarded_blend(original, filled, mask))
    finally:
        source.release()
        generated.release()
        writer.release()


def main() -> None:
    raise_nofile_limit()
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_status(out, "starting", from_scratch=True, version="v8")
    timing: dict[str, float] = defaultdict(float)
    timing["optical_flow_s"] = 0.0
    wall_started = time.perf_counter()

    cap = cv2.VideoCapture(args.input)
    # There is deliberately no global background plate in v7.
    timing["background_build_s"] = 0.0

    if not cap.isOpened():
        raise RuntimeError(f"failed to open input: {args.input}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    temporary = Path(tempfile.mkdtemp(prefix="green_v8_", dir=os.environ.get("TMPDIR")))
    model_mask_dir = temporary / "model_masks"
    mask_dir = temporary / "masks"
    propainter_out = temporary / "propainter_output"
    raw_mask = temporary / "mask.raw.avi"
    raw_result = temporary / "result.raw.avi"
    model_mask_dir.mkdir()
    mask_dir.mkdir()
    final_output = out / args.output_name
    try:
        mask_started = time.perf_counter()
        frames, _, diagnostics = build_masks(
            args, model_mask_dir, raw_mask, fps, (width, height), timing
        )
        temporal_started = time.perf_counter()
        areas, temporal_diagnostics = stabilize_masks(
            args.input, model_mask_dir, mask_dir, frames, args, timing
        )
        timing["temporal_stabilize_s"] = time.perf_counter() - temporal_started
        diagnostics.update(temporal_diagnostics)
        timing["mask_total_s"] = time.perf_counter() - mask_started
        if args.keep_mask_video:
            write_mask_video(mask_dir, raw_mask, frames, fps, (width, height))
            encode_compatible_mp4(raw_mask, out / "glove_mask_v8.mp4", fps)

        write_status(out, "propainter", processed_frames=frames)
        propainter_root = Path(args.propainter_root).resolve()
        propainter_started = time.perf_counter()
        run_checked(
            [
                os.sys.executable,
                str(propainter_root / "inference_propainter.py"),
                "--video", args.input,
                "--mask", str(mask_dir),
                "--output", str(propainter_out),
                "--width", str(width),
                "--height", str(height),
                "--mask_dilation", str(args.propainter_mask_dilation),
                "--neighbor_length", str(args.neighbor_length),
                "--ref_stride", str(args.ref_stride),
                "--subvideo_length", str(args.subvideo_length),
                "--raft_iter", str(args.raft_iter),
                "--save_fps", str(round(fps)),
                "--fp16",
            ],
            cwd=propainter_root,
        )
        timing["propainter_s"] = time.perf_counter() - propainter_started
        generated = find_propainter_result(propainter_out, Path(args.input).stem)

        composite_started = time.perf_counter()
        guarded_composite_video(
            args.input, generated, mask_dir, raw_result,
            frames, fps, (width, height),
        )
        encode_compatible_mp4(raw_result, final_output, fps)
        timing["composite_encode_s"] = time.perf_counter() - composite_started
        timing["frames"] = frames
        timing["wall_s"] = time.perf_counter() - wall_started

        metadata = {
            "scheme": "every_frame_dino_sam_universal_identity_temporal_propainter_v9",
            "input": args.input,
            "output": str(final_output),
            "frames": frames,
            "total_frames": total,
            "fps": fps,
            "mask_pixels": {
                "min": min(areas), "max": max(areas),
                "mean": float(np.mean(areas)),
                "zero_frames": int(sum(area == 0 for area in areas)),
            },
            "model_calls": {
                "dino": int(timing["dino_calls"]),
                "sam": int(timing["sam_calls"]),
                "sam_no_box_frames": int(timing["sam_no_box_frames"]),
            },
            "diagnostics": diagnostics,
            "arguments": vars(args),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out / "timing.json").write_text(json.dumps(dict(timing), indent=2), encoding="utf-8")
        write_status(
            out, "complete", processed_frames=frames,
            output=str(final_output), wall_s=timing["wall_s"],
        )
        print(
            f"[green-v9] complete output={final_output} "
            f"wall={timing['wall_s']:.1f}s avg={timing['wall_s']*1000/frames:.1f}ms/frame",
            flush=True,
        )
    except Exception as exc:
        write_status(out, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
