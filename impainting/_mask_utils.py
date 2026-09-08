#!/usr/bin/env python3
"""Final mask utility helpers for Remove_Hand."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

def red_protection(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsv)
    red = (((h <= 12) | (h >= 170)) & (s >= 90) & (v >= 55)).astype(np.uint8)
    return cv2.dilate(red, np.ones((5, 5), np.uint8), iterations=1) > 0

def retain_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_area and width >= 6 and (height >= 6):
            kept[labels == label] = 255
    return kept

def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'failed to open writer: {path}')
    return writer
