"""Re-ID-only crop geometry; detector/tracker boxes are never modified."""

import cv2

from .config import (
    REID_CROP_BOTTOM_MARGIN,
    REID_CROP_MODE,
    REID_CROP_SIDE_MARGIN,
    REID_CROP_TOP_MARGIN,
    resolve_reid_crop_mode,
)


def _clamp_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def extract_person_crop(frame, x1, y1, x2, y2):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, width, height)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 1 or box_height <= 1:
        return None
    side = int(box_width * REID_CROP_SIDE_MARGIN)
    top = int(box_height * REID_CROP_TOP_MARGIN)
    bottom = int(box_height * REID_CROP_BOTTOM_MARGIN)
    crop_box = _clamp_bbox(
        x1 + side, y1 + top, x2 - side, y2 - bottom, width, height
    )
    crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    return crop if crop is not None and crop.size else None


def extract_person_crop_without_margin(frame, x1, y1, x2, y2):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, width, height)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop is not None and crop.size else None


def get_reid_crop(frame, x1, y1, x2, y2, crop_mode=None):
    selected = REID_CROP_MODE if crop_mode is None else resolve_reid_crop_mode(crop_mode)
    if selected == "improved":
        return extract_person_crop_without_margin(frame, x1, y1, x2, y2)
    return extract_person_crop(frame, x1, y1, x2, y2)


def extract_person_embedding(frame, x1, y1, x2, y2, extractor, crop_mode=None):
    crop = get_reid_crop(frame, x1, y1, x2, y2, crop_mode)
    if crop is None or extractor is None:
        return None
    return extractor.extract(crop)
