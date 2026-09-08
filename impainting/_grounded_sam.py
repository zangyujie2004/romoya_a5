#!/usr/bin/env python3
"""Final DINO/SAM and MP4 helpers for Remove_Hand."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

class GroundedSAM:

    def __init__(self, models_root: str, dino_dir: str, sam_dir: str, device: str) -> None:
        root = Path(models_root)
        self.device = device
        dino_path = root / dino_dir
        sam_path = root / sam_dir
        print(f'[glove-inpaint] GroundingDINO={dino_path}')
        print(f'[glove-inpaint] SAM={sam_path}')
        self.dino_processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True).to(device)
        self.sam_processor = SamProcessor.from_pretrained(sam_path, local_files_only=True)
        self.sam_model = SamModel.from_pretrained(sam_path, local_files_only=True).to(device)

    def boxes(self, frame_bgr: np.ndarray, prompt: str, box_threshold: float, text_threshold: float) -> np.ndarray:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.dino_processor(images=image, text=prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        try:
            results = self.dino_processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]])
        except TypeError as exc:
            if 'box_threshold' not in str(exc):
                raise
            results = self.dino_processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]])
        boxes = results[0].get('boxes')
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return boxes.detach().cpu().numpy().astype(np.float32)

    def mask_from_boxes(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        if boxes_xyxy.size == 0:
            return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        input_boxes = boxes_xyxy[None, :, :].tolist()
        inputs = self.sam_processor(image, input_boxes=input_boxes, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0]
        masks = masks.detach().cpu().numpy()
        mask = np.any(masks > 0, axis=(0, 1)).astype(np.uint8) * 255
        return mask

def filter_boxes(boxes: np.ndarray, frame_shape: tuple[int, int, int], roi_bottom_ratio: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    (h, _) = frame_shape[:2]
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    return boxes[cy >= h * roi_bottom_ratio]

def encode_compatible_mp4(src_path: Path, dst_path: Path, fps: float) -> None:
    """Write a QuickTime/VSCode friendly MP4: H.264, yuv420p, faststart."""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError('FFmpeg is required for final MP4 encoding. Install system ffmpeg or add imageio-ffmpeg to the active Python environment.') from exc
    tmp_dir = Path(tempfile.mkdtemp(prefix='glove_inpaint_mp4_'))
    tmp_dst = tmp_dir / dst_path.name
    cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-fflags', '+genpts', '-i', str(src_path), '-r', f'{fps:.6f}', '-an', '-c:v', 'libx264', '-tag:v', 'avc1', '-preset', 'veryfast', '-crf', '18', '-bf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp_dst)]
    subprocess.run(cmd, check=True)
    if not tmp_dst.is_file() or tmp_dst.stat().st_size == 0:
        raise RuntimeError(f'FFmpeg produced an empty output: {tmp_dst}')
    probe = cv2.VideoCapture(str(tmp_dst))
    (ok, _) = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError(f'encoded MP4 failed decode validation: {tmp_dst}')
    dst_tmp = dst_path.with_suffix(dst_path.suffix + '.tmp')
    dst_tmp.write_bytes(tmp_dst.read_bytes())
    dst_tmp.replace(dst_path)
    tmp_dst.unlink(missing_ok=True)
    tmp_dir.rmdir()
