"""Re-ID configuration values and environment resolution.

Values intentionally mirror the production defaults in ``backend/main.py``.
This module has no dependency on the application module.
"""

import logging
import os

try:
    from ..reid_config import (
        OSNET_ARCHITECTURE,
        OSNET_DEFAULT_CHECKPOINT_NAME,
        OSNET_EMBEDDING_DIMENSION,
        OSNET_INPUT_HEIGHT,
        OSNET_INPUT_WIDTH,
        osnet_preprocessing_metadata,
    )
except ImportError:  # pragma: no cover - direct module compatibility
    from reid_config import (
        OSNET_ARCHITECTURE,
        OSNET_DEFAULT_CHECKPOINT_NAME,
        OSNET_EMBEDDING_DIMENSION,
        OSNET_INPUT_HEIGHT,
        OSNET_INPUT_WIDTH,
        osnet_preprocessing_metadata,
    )

logger = logging.getLogger(__name__)

USE_OSNET = os.getenv("REID_ENABLED", "true").strip().lower() not in {
    "0", "false", "no", "off",
}
REID_MODEL_NAME = OSNET_ARCHITECTURE
REID_MODEL_PATH_CONFIG = os.getenv(
    "REID_CHECKPOINT_PATH",
    os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
        "weights",
        OSNET_DEFAULT_CHECKPOINT_NAME,
    ),
).strip()
REID_DEVICE_CONFIG = os.getenv("REID_DEVICE", "auto").strip().lower()
REID_THRESHOLD_SAFETY_MODE = os.getenv(
    "REID_THRESHOLD_SAFETY_MODE", "conservative",
).strip().lower()


def resolve_reid_crop_mode(configured_value):
    crop_mode = str(
        configured_value if configured_value is not None else "original"
    ).strip().lower()
    if crop_mode not in {"original", "improved"}:
        logger.warning(
            "Unknown REID_CROP_MODE=%s; using original", configured_value
        )
        return "original"
    return crop_mode


REID_CROP_MODE = resolve_reid_crop_mode(
    os.getenv("REID_CROP_MODE", "improved")
)


def resolve_reid_checkpoint_path(configured_path):
    path = os.path.expanduser(os.path.expandvars(configured_path))
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    return os.path.abspath(path)


REID_MODEL_PATH = resolve_reid_checkpoint_path(REID_MODEL_PATH_CONFIG)
REID_INPUT_H = OSNET_INPUT_HEIGHT
REID_INPUT_W = OSNET_INPUT_WIDTH
REID_INFERENCE_INTERVAL = 5
REID_CACHE_MAX_AGE_FRAMES = 15
REID_LOCAL_EMA_ALPHA = 0.90
REID_REQUIRE_CONFIRMED_TRACK = True
REID_MIN_CROP_SIZE = 24
REID_CROP_SIDE_MARGIN = 0.18
REID_CROP_TOP_MARGIN = 0.10
REID_CROP_BOTTOM_MARGIN = 0.18

REID_RUNTIME_STATUS = {
    "enabled": bool(USE_OSNET),
    "model_architecture": REID_MODEL_NAME,
    "checkpoint_path": REID_MODEL_PATH,
    "checkpoint_name": os.path.basename(REID_MODEL_PATH),
    "checkpoint_loaded": False,
    "device": None,
    "fallback_active": False,
    "embedding_dimension": None,
    "expected_embedding_dimension": OSNET_EMBEDDING_DIMENSION,
    "preprocessing": osnet_preprocessing_metadata(),
    "checkpoint_metadata": None,
    "crop_mode": REID_CROP_MODE,
    "production_crop_source": "detector_tracker_bbox",
    "model_eval_mode": None,
    "excluded_classifier_keys": [],
    "active_extractor": None,
    "error": None,
}
