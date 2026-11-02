"""Pure image-processing helpers used by the inference pipeline."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def mild_clahe_bgr(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )
    return cv2.addWeighted(image_bgr, 0.65, enhanced, 0.35, 0)


def safe_crop(image_bgr: np.ndarray, box: np.ndarray, padding_ratio: float = 0.06) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    pad_x = max(1.0, x2 - x1) * padding_ratio
    pad_y = max(1.0, y2 - y1) * padding_ratio
    x1_i, y1_i = max(0, round(x1 - pad_x)), max(0, round(y1 - pad_y))
    x2_i, y2_i = min(width, round(x2 + pad_x)), min(height, round(y2 + pad_y))
    return image_bgr[y1_i:y2_i, x1_i:x2_i].copy()


def pad_to_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def spatial_order(boxes: np.ndarray) -> list[int]:
    if len(boxes) == 0:
        return []
    heights = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    row_unit = max(1.0, float(np.median(heights)) * 0.75)
    return sorted(
        range(len(boxes)),
        key=lambda index: (
            round(((boxes[index, 1] + boxes[index, 3]) / 2.0) / row_unit),
            (boxes[index, 0] + boxes[index, 2]) / 2.0,
        ),
    )
