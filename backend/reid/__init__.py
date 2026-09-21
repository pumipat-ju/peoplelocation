"""Re-ID building blocks extracted from the production pipeline.

The package is dependency-light at import time.  Model construction remains
explicit so importing ``backend.reid`` does not load a checkpoint or start a
worker.
"""

from .config import (
    REID_CROP_MODE,
    REID_DEVICE_CONFIG,
    REID_INFERENCE_INTERVAL,
    REID_LOCAL_EMA_ALPHA,
    REID_MIN_CROP_SIZE,
    REID_MODEL_NAME,
    REID_MODEL_PATH,
    REID_MODEL_PATH_CONFIG,
    REID_REQUIRE_CONFIRMED_TRACK,
    REID_RUNTIME_STATUS,
    REID_INPUT_H,
    REID_INPUT_W,
    USE_OSNET,
    resolve_reid_checkpoint_path,
    resolve_reid_crop_mode,
)
from .crop import (
    extract_person_crop,
    extract_person_crop_without_margin,
    get_reid_crop,
)
from .similarity import cosine_similarity, l2_normalize
from .extractor import (
    OSNetFeatureExtractor,
    load_validated_osnet_checkpoint,
    resolve_reid_device,
)

__all__ = [
    "USE_OSNET",
    "REID_CROP_MODE",
    "REID_DEVICE_CONFIG",
    "REID_INFERENCE_INTERVAL",
    "REID_LOCAL_EMA_ALPHA",
    "REID_MIN_CROP_SIZE",
    "REID_MODEL_NAME",
    "REID_MODEL_PATH",
    "REID_MODEL_PATH_CONFIG",
    "REID_REQUIRE_CONFIRMED_TRACK",
    "REID_RUNTIME_STATUS",
    "REID_INPUT_H",
    "REID_INPUT_W",
    "resolve_reid_checkpoint_path",
    "resolve_reid_crop_mode",
    "extract_person_crop",
    "extract_person_crop_without_margin",
    "get_reid_crop",
    "cosine_similarity",
    "l2_normalize",
    "OSNetFeatureExtractor",
    "load_validated_osnet_checkpoint",
    "resolve_reid_device",
]
