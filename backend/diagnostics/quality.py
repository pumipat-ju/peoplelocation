"""Pure Re-ID crop quality diagnostics."""

import cv2

def build_reid_quality_metadata(frame, box, confidence, overlap=False):
    """Describe a Re-ID crop without retaining the image itself."""
    x1, y1, x2, y2 = [int(value) for value in box]
    frame_h, frame_w = frame.shape[:2]
    crop_w = max(0, x2 - x1)
    crop_h = max(0, y2 - y1)
    touches_border = (
        x1 <= 0 or y1 <= 0 or x2 >= frame_w - 1 or y2 >= frame_h - 1
    )
    border_clip_ratio = 0.25 if touches_border else 0.0
    crop = frame[
        max(0, y1):min(frame_h, y2),
        max(0, x1):min(frame_w, x2)
    ]
    blur_variance = 0.0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "crop_size": (crop_w, crop_h),
        "detector_confidence": confidence,
        "overlap": bool(overlap),
        "border_clip_ratio": border_clip_ratio,
        "blur_variance": blur_variance,
    }
