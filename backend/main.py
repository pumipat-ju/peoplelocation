import cv2
import copy
import os
import sys
import threading
import time
import json
import base64
import warnings
import shutil
from contextlib import nullcontext
import uuid
import numpy as np

# OPTIMIZED REID BUILD:
# - OSNet batch inference
# - confirmed-track embedding cache + periodic refresh
# - EMA smoothing for refreshed local-track embeddings
# - CUDA/FP16 YOLO when available
# - ReID inference/cache timing diagnostics
# Camera capture, synchronization, homography and floorplan logic are unchanged.

try:
    from .identity_store import IdentityStore
except ImportError:
    from identity_store import IdentityStore

TORCH_IMPORT_ERROR = None
TORCHREID_IMPORT_ERROR = None
FEATURE_EXTRACTOR_IMPORT_ERROR = None

try:
    import torch
except Exception as error:
    torch = None
    TORCH_IMPORT_ERROR = str(error)

try:
    import torchreid
except Exception as error:
    torchreid = None
    TORCHREID_IMPORT_ERROR = str(error)

FeatureExtractor = None

if torchreid is not None:
    try:
        from torchreid.utils import FeatureExtractor
    except Exception as primary_error:
        try:
            from torchreid.reid.utils import FeatureExtractor
        except Exception as fallback_error:
            FEATURE_EXTRACTOR_IMPORT_ERROR = (
                f"primary={primary_error}; "
                f"fallback={fallback_error}"
            )

import logging
import signal
from urllib.parse import urlparse
from collections import deque
from scipy.optimize import linear_sum_assignment

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

try:
    from .reid_config import (
        OSNET_ARCHITECTURE,
        OSNET_DEFAULT_CHECKPOINT_NAME,
        OSNET_EMBEDDING_DIMENSION,
        OSNET_INPUT_HEIGHT,
        OSNET_INPUT_WIDTH,
        OSNET_PIXEL_MEAN,
        OSNET_PIXEL_STD,
        osnet_preprocessing_metadata,
        read_osnet_checkpoint_metadata,
        validate_osnet_checkpoint_metadata
    )
except ImportError:
    from reid_config import (
        OSNET_ARCHITECTURE,
        OSNET_DEFAULT_CHECKPOINT_NAME,
        OSNET_EMBEDDING_DIMENSION,
        OSNET_INPUT_HEIGHT,
        OSNET_INPUT_WIDTH,
        OSNET_PIXEL_MEAN,
        OSNET_PIXEL_STD,
        osnet_preprocessing_metadata,
        read_osnet_checkpoint_metadata,
        validate_osnet_checkpoint_metadata
    )

try:
    from .reid.crop import (
        extract_person_crop as _reid_extract_person_crop,
        extract_person_crop_without_margin as _reid_extract_person_crop_without_margin,
        extract_person_embedding as _reid_extract_person_embedding,
        get_reid_crop as _reid_get_reid_crop,
    )
    from .reid.config import (
        resolve_reid_checkpoint_path as _reid_resolve_checkpoint_path,
        resolve_reid_crop_mode as _reid_resolve_crop_mode,
    )
    from .reid.similarity import (
        cosine_similarity as _reid_cosine_similarity,
        l2_normalize as _reid_l2_normalize,
    )
    from .reid.gallery import (
        gallery_similarity as _reid_gallery_similarity,
        identity_prototype_candidates as _reid_identity_prototype_candidates,
        normalize_embedding_candidate as _reid_normalize_embedding_candidate,
        robust_identity_prototype as _reid_robust_identity_prototype,
    )
    from .reid.extractor import (
        OSNetFeatureExtractor as _ReIDOSNetFeatureExtractor,
        LightweightAppearanceFeatureExtractor as _ReIDLightweightAppearanceFeatureExtractor,
        build_feature_extractor as _reid_build_feature_extractor,
        configure_extractor_dependencies,
        load_validated_osnet_checkpoint as _reid_load_validated_osnet_checkpoint,
        resolve_reid_device as _reid_resolve_reid_device,
    )
    from .diagnostics import (
        build_reid_quality_metadata,
        safe_json_value,
    )
except ImportError:
    from reid.crop import (
        extract_person_crop as _reid_extract_person_crop,
        extract_person_crop_without_margin as _reid_extract_person_crop_without_margin,
        extract_person_embedding as _reid_extract_person_embedding,
        get_reid_crop as _reid_get_reid_crop,
    )
    from reid.config import (
        resolve_reid_checkpoint_path as _reid_resolve_checkpoint_path,
        resolve_reid_crop_mode as _reid_resolve_crop_mode,
    )
    from reid.similarity import (
        cosine_similarity as _reid_cosine_similarity,
        l2_normalize as _reid_l2_normalize,
    )
    from reid.gallery import (
        gallery_similarity as _reid_gallery_similarity,
        identity_prototype_candidates as _reid_identity_prototype_candidates,
        normalize_embedding_candidate as _reid_normalize_embedding_candidate,
        robust_identity_prototype as _reid_robust_identity_prototype,
    )
    from reid.extractor import (
        OSNetFeatureExtractor as _ReIDOSNetFeatureExtractor,
        LightweightAppearanceFeatureExtractor as _ReIDLightweightAppearanceFeatureExtractor,
        build_feature_extractor as _reid_build_feature_extractor,
        configure_extractor_dependencies,
        load_validated_osnet_checkpoint as _reid_load_validated_osnet_checkpoint,
        resolve_reid_device as _reid_resolve_reid_device,
    )
    from diagnostics import (
        build_reid_quality_metadata,
        safe_json_value,
    )
# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()

from pathlib import Path

try:
    from .embedding_store import EmbeddingStore
    from .embedding_view import router as embedding_view_router, configure_store, configure_embedding_extractor
except ImportError:
    from embedding_store import EmbeddingStore
    from embedding_view import router as embedding_view_router, configure_store, configure_embedding_extractor

embedding_store = EmbeddingStore(
    db_path=Path(__file__).resolve().parent / "database" / "embeddings.sqlite3",
    embedding_dim=512,
)

configure_store(embedding_store)
app.include_router(embedding_view_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# YOLO
# ============================================================

YOLO_MODEL_PATH = "yolov8s.pt"

# ------------------------------------------------------------
# Inference performance
# ------------------------------------------------------------
# Use CUDA automatically when available. Keeping these options here changes
# only model inference; camera capture/synchronization is untouched.
YOLO_DEVICE = (
    0
    if (
        torch is not None
        and torch.cuda.is_available()
    )
    else "cpu"
)
YOLO_USE_HALF = bool(
    torch is not None
    and torch.cuda.is_available()
)
YOLO_IMAGE_SIZE = 640

# BoT-SORT stores persistent state on the YOLO predictor. A YOLO instance is
# therefore created per camera by get_camera_tracking_model(). Keeping this
# compatibility name as None prevents accidental reuse of a shared tracker.
model = None

app.is_running = True


# ============================================================
# GLOBAL VARIABLES
# ============================================================

cameras_lock = threading.Lock()
cameras = {}

MAX_UPLOAD_SIZE = 500 * 1024 * 1024

try:
    LIVE_CAMERA_RECONNECT_INTERVAL_SEC = max(
        0.1,
        float(
            os.getenv(
                "LIVE_CAMERA_RECONNECT_INTERVAL_SEC",
                "1.0"
            )
        )
    )
except (TypeError, ValueError):
    LIVE_CAMERA_RECONNECT_INTERVAL_SEC = 1.0

LIVE_CAMERA_STOP_TIMEOUT_SEC = 3.0

FLOORPLAN_PATH = "static/floorplan.png"
FLOORPLAN_DIR = os.path.join("static", "floorplans")
UPLOAD_DIR = "static/uploads"
TOPOLOGY_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "camera_topology.json"
)

os.makedirs("static", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FLOORPLAN_DIR, exist_ok=True)


def normalize_floorplan_name(value):
    """Return a safe stored filename while preserving a supported extension."""
    raw_name = os.path.basename(str(value or "floorplan.png").strip())
    stem, extension = os.path.splitext(raw_name)
    extension = extension.lower()
    if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
        extension = ".png"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_")
    return f"{safe_stem or 'floorplan'}{extension}"


def floorplan_path_by_name(name):
    safe_name = normalize_floorplan_name(name)
    path = os.path.abspath(os.path.join(FLOORPLAN_DIR, safe_name))
    root = os.path.abspath(FLOORPLAN_DIR)
    if os.path.commonpath([root, path]) != root:
        raise ValueError("Invalid floorplan name")
    return path, safe_name


def available_floorplan_names():
    names = []
    if os.path.isdir(FLOORPLAN_DIR):
        for name in sorted(os.listdir(FLOORPLAN_DIR)):
            path, safe_name = floorplan_path_by_name(name)
            if os.path.isfile(path):
                names.append(safe_name)
    return names


# ============================================================
# RE-ID CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# OSNet
# ------------------------------------------------------------

USE_OSNET = (
    os.getenv(
        "REID_ENABLED",
        "true"
    ).strip().lower()
    not in {"0", "false", "no", "off"}
)

# Shared production/training/evaluation architecture
REID_MODEL_NAME = OSNET_ARCHITECTURE

# Weight ที่ train กับ Market1501
REID_MODEL_PATH_CONFIG = os.getenv(
    "REID_CHECKPOINT_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weights",
        OSNET_DEFAULT_CHECKPOINT_NAME
    )
).strip()

REID_DEVICE_CONFIG = os.getenv(
    "REID_DEVICE",
    "auto"
).strip().lower()

REID_THRESHOLD_SAFETY_MODE = os.getenv(
    "REID_THRESHOLD_SAFETY_MODE",
    "conservative"
).strip().lower()


def resolve_reid_crop_mode(configured_value):
    return _reid_resolve_crop_mode(configured_value)


REID_CROP_MODE = resolve_reid_crop_mode(
    os.getenv("REID_CROP_MODE", "improved")
)

# Validation-only EER operating point from the offline V5 experiment.
# This is diagnostic metadata, not a production matching threshold.
V5_OFFLINE_VALIDATION_THRESHOLD_REFERENCE = 0.583767414

if REID_THRESHOLD_SAFETY_MODE not in {
    "conservative",
    "validated"
}:
    logger.warning(
        "Unknown REID_THRESHOLD_SAFETY_MODE=%s; using conservative",
        REID_THRESHOLD_SAFETY_MODE
    )
    REID_THRESHOLD_SAFETY_MODE = "conservative"


def resolve_reid_checkpoint_path(
    configured_path
):
    return _reid_resolve_checkpoint_path(configured_path)


REID_MODEL_PATH = (
    resolve_reid_checkpoint_path(
        REID_MODEL_PATH_CONFIG
    )
)

REID_RUNTIME_STATUS = {
    "enabled": bool(USE_OSNET),
    "model_architecture": REID_MODEL_NAME,
    "checkpoint_path": REID_MODEL_PATH,
    "checkpoint_name": os.path.basename(
        REID_MODEL_PATH
    ),
    "checkpoint_loaded": False,
    "device": None,
    "fallback_active": False,
    "embedding_dimension": None,
    "expected_embedding_dimension": (
        OSNET_EMBEDDING_DIMENSION
    ),
    "preprocessing": (
        osnet_preprocessing_metadata()
    ),
    "checkpoint_metadata": None,
    "crop_mode": REID_CROP_MODE,
    "production_crop_source": "detector_tracker_bbox",
    "model_eval_mode": None,
    "excluded_classifier_keys": [],
    "offline_v5_validation_threshold_reference": (
        V5_OFFLINE_VALIDATION_THRESHOLD_REFERENCE
    ),
    "threshold_safety_mode": REID_THRESHOLD_SAFETY_MODE,
    "similarity_only_shortcut_enabled": (
        REID_THRESHOLD_SAFETY_MODE
        == "validated"
    ),
    "active_extractor": None,
    "error": None
}

# OSNet รับภาพขนาด H x W
REID_INPUT_H = OSNET_INPUT_HEIGHT
REID_INPUT_W = OSNET_INPUT_WIDTH

# Re-ID configuration source of truth.  These names remain available from
# ``backend.main`` for compatibility while their implementation lives in the
# extracted package.
try:
    from .reid.config import (
        REID_CROP_BOTTOM_MARGIN,
        REID_CROP_MODE,
        REID_CROP_SIDE_MARGIN,
        REID_CROP_TOP_MARGIN,
        REID_DEVICE_CONFIG,
        REID_INFERENCE_INTERVAL,
        REID_INPUT_H,
        REID_INPUT_W,
        REID_LOCAL_EMA_ALPHA,
        REID_MIN_CROP_SIZE,
        REID_MODEL_NAME,
        REID_MODEL_PATH,
        REID_MODEL_PATH_CONFIG,
        REID_REQUIRE_CONFIRMED_TRACK,
        USE_OSNET,
        resolve_reid_checkpoint_path,
        resolve_reid_crop_mode,
    )
except ImportError:
    from reid.config import (
        REID_CROP_BOTTOM_MARGIN,
        REID_CROP_MODE,
        REID_CROP_SIDE_MARGIN,
        REID_CROP_TOP_MARGIN,
        REID_DEVICE_CONFIG,
        REID_INFERENCE_INTERVAL,
        REID_INPUT_H,
        REID_INPUT_W,
        REID_LOCAL_EMA_ALPHA,
        REID_MIN_CROP_SIZE,
        REID_MODEL_NAME,
        REID_MODEL_PATH,
        REID_MODEL_PATH_CONFIG,
        REID_REQUIRE_CONFIRMED_TRACK,
        USE_OSNET,
        resolve_reid_checkpoint_path,
        resolve_reid_crop_mode,
    )


# ------------------------------------------------------------
# Global ReID
# ------------------------------------------------------------

# คนสามารถหายจากกล้องได้กี่วินาที
# แล้วยังพยายามเอา GID เดิมกลับมา
REID_MAX_IDLE_SEC = 30.0
REID_MAX_GALLERY_IDLE_SEC = 120.0

# ------------------------------------------------------------
# Similarity thresholds
# ------------------------------------------------------------

# Cross-camera
REID_CROSS_CAM_THRESHOLD = 0.55
REID_CROSS_CAM_STRONG_THRESHOLD = 0.70

# Same-camera
REID_SAME_CAM_THRESHOLD = 0.45
REID_SAME_CAM_STRONG_THRESHOLD = 0.65


# ------------------------------------------------------------
# Floorplan distance
# ------------------------------------------------------------

# ใช้สำหรับข้ามกล้อง
# ต้องไม่แคบเกินไป เพราะกล้อง 1 -> กล้อง 2
# อาจมีระยะห่างของตำแหน่งบน floorplan
REID_MAP_GATE_CROSS_CAM_PX = 1000.0

# Same camera ใช้ gate แคบกว่า
REID_MAP_GATE_SAME_CAM_PX = 350.0


# ------------------------------------------------------------
# Bounding box size
# ------------------------------------------------------------

REID_SIZE_GATE_RATIO = 0.25


# ------------------------------------------------------------
# Embedding gallery
# ------------------------------------------------------------

REID_GALLERY_SIZE = 12

# ค่า 0.90 หมายถึง embedding หลักจะเปลี่ยนช้า
REID_EMBED_UPDATE_ALPHA = 0.90

# ------------------------------------------------------------
# ReID runtime optimization
# ------------------------------------------------------------
# A confirmed BoT-SORT track reuses its most recent embedding between OSNet
# refreshes. This removes redundant OSNet inference without changing cameras.
REID_INFERENCE_INTERVAL = 5
REID_CACHE_MAX_AGE_FRAMES = 15
REID_LOCAL_EMA_ALPHA = 0.90
REID_REQUIRE_CONFIRMED_TRACK = True


# ------------------------------------------------------------
# Minimum crop
# ------------------------------------------------------------

REID_MIN_CROP_SIZE = 24

# Tracklet / quality gallery.  These gates apply only to identity memory;
# detections are still available to the existing tracker and assignment flow.
REID_TRACKLET_MIN_SAMPLES = 3
REID_TRACKLET_MAX_SAMPLES = 24
REID_GALLERY_DIVERSITY_THRESHOLD = 0.985
REID_MIN_DETECTION_CONFIDENCE = 0.50
REID_MAX_BORDER_CLIP_RATIO = 0.20
REID_MAX_OVERLAP_FOR_GALLERY = 0.0
REID_MIN_BLUR_VARIANCE = 10.0


# ------------------------------------------------------------
# ReID crop
# ------------------------------------------------------------

REID_CROP_SIDE_MARGIN = 0.18
REID_CROP_TOP_MARGIN = 0.10
REID_CROP_BOTTOM_MARGIN = 0.18


# ------------------------------------------------------------
# Same camera cache
# ------------------------------------------------------------

REID_RECENT_SAME_CAM_SEC = 12.0
REID_RECENT_SAME_CAM_THRESHOLD = 0.40


# ------------------------------------------------------------
# Cross camera cache
# ------------------------------------------------------------

REID_RECENT_CROSS_CAM_SEC = 30.0


# ------------------------------------------------------------
# Assignment weights
# ------------------------------------------------------------

# Same camera
ASSIGN_SAME_CAM_APPEARANCE_WEIGHT = 0.42
ASSIGN_SAME_CAM_MOTION_WEIGHT = 0.42
ASSIGN_SAME_CAM_MAP_WEIGHT = 0.10
ASSIGN_SAME_CAM_TIME_WEIGHT = 0.06


# Cross camera
# เน้น ReID มากกว่า motion
ASSIGN_CROSS_CAM_APPEARANCE_WEIGHT = 0.75
ASSIGN_CROSS_CAM_MAP_WEIGHT = 0.15
ASSIGN_CROSS_CAM_TIME_WEIGHT = 0.10


# ------------------------------------------------------------
# Assignment thresholds
# ------------------------------------------------------------

ASSIGN_SAME_CAM_SCORE_THRESHOLD = 0.28

ASSIGN_CROSS_CAM_SCORE_THRESHOLD = 0.42

ASSIGN_STRONG_APPEARANCE_THRESHOLD = 0.78

# Reject otherwise-valid candidates when their evidence is too close to the
# runner-up.  Keeping this explicit favours a temporary split over a false
# identity merge.
ASSIGN_SAME_CAM_MIN_MARGIN = 0.05
ASSIGN_CROSS_CAM_MIN_MARGIN = 0.08

# A cross-camera top-1/top-2 ambiguity is temporary evidence, not proof of a
# new person.  Keep only bounded downstream state while another arrival or a
# stronger aggregate can resolve it.  These are evidence-window bounds; they
# do not weaken any appearance, score, topology, or lifecycle threshold.
AMBIGUOUS_HANDOFF_MIN_SAMPLES = REID_TRACKLET_MIN_SAMPLES
AMBIGUOUS_HANDOFF_MAX_SAMPLES = REID_TRACKLET_MAX_SAMPLES
AMBIGUOUS_HANDOFF_MIN_SOLO_EVENT_SEC = 1.5
AMBIGUOUS_HANDOFF_MAX_EVENT_SEC = 2.0
AMBIGUOUS_HANDOFF_MAX_RECORDS = 512
AMBIGUOUS_HANDOFF_TEMPORAL_TOP1_MIN_RATIO = 0.80
UNRESOLVED_HANDOFF_FORENSIC_MAX_EVENTS = 4096

IDENTITY_PROVISIONAL = "PROVISIONAL"
IDENTITY_ACTIVE = "ACTIVE"
IDENTITY_DORMANT = "DORMANT"
IDENTITY_EXPIRED = "EXPIRED"
IDENTITY_DORMANT_TTL_SEC = 300.0
IDENTITY_TRANSITION_HISTORY_SIZE = 100
IDENTITY_HANDOFF_HISTORY_SIZE = 100
IDENTITY_EXPIRED_SNAPSHOT_RETENTION_SEC = 7 * 24 * 60 * 60
IDENTITY_ASSIGNABLE_STATES = frozenset({
    IDENTITY_PROVISIONAL,
    IDENTITY_ACTIVE,
    IDENTITY_DORMANT,
})
IDENTITY_ALLOWED_TRANSITIONS = {
    IDENTITY_PROVISIONAL: frozenset({
        IDENTITY_ACTIVE,
        IDENTITY_DORMANT,
    }),
    IDENTITY_ACTIVE: frozenset({IDENTITY_DORMANT}),
    IDENTITY_DORMANT: frozenset({
        IDENTITY_ACTIVE,
        IDENTITY_EXPIRED,
    }),
    IDENTITY_EXPIRED: frozenset(),
}


# ------------------------------------------------------------
# Occlusion
# ------------------------------------------------------------

OCCLUSION_IOU_THRESHOLD = 0.50

OCCLUSION_HOLD_SEC = 0.5

OCCLUSION_PREV_IOU_THRESHOLD = 0.30

OCCLUSION_CENTER_DIST_PX = 80.0

ASSIGN_OVERLAP_FREEZE_BONUS = 0.08

ASSIGN_SAME_CAM_BONUS = 0.10

#--------

#--------

LOCAL_TRACK_VERIFY_THRESHOLD = 0.45
LOCAL_TRACK_STRONG_THRESHOLD = 0.65

# A confirmed camera-local tracker may bridge a brief low-quality bootstrap
# while the first durable tracklet prototype is still being collected. This
# never makes provisional evidence globally reusable.
REID_PROVISIONAL_LOCAL_CONTINUITY_SEC = 8.0

# ------------------------------------------------------------
# Debug
# ------------------------------------------------------------

REID_DEBUG = True

# Identity observations are coordinated downstream of capture/tracking.  This
# window never applies to frame acquisition and camera workers never wait for
# another camera to submit an observation.
GLOBAL_ASSIGNMENT_WINDOW_SEC = 0.25
GLOBAL_ASSIGNMENT_MAX_PENDING_CAMERAS = 128
GLOBAL_ASSIGNMENT_MAX_OBSERVATIONS_PER_CAMERA = 256
GLOBAL_ASSIGNMENT_MAX_READY_BATCHES = 4
IDENTITY_DB_PATH = os.getenv(
    "IDENTITY_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "identity_memory.sqlite3"),
)




# ============================================================
# HYBRID V26 RE-ID EXTENSIONS
# Camera / player runtime remains from main(2).py
# ============================================================

TRACKLET_PROTOTYPE_ENABLED = True
TRACKLET_PROTOTYPE_MAX_SAMPLES = 8
TRACKLET_PROTOTYPE_MIN_SAMPLES = 3
TRACKLET_PROTOTYPE_WEIGHT = 0.70
TRACKLET_SUPPORT_WEIGHT = 0.30
TRACKLET_SUPPORT_TOPK = 3
TRACKLET_PROTOTYPE_MIN_CONSENSUS = 0.65
SOFT_LOCAL_CONTINUITY_ENABLED = True
SOFT_LOCAL_CONTINUITY_MAX_GAP_SEC = 0.30
SOFT_LOCAL_CONTINUITY_BONUS = 0.04
SOFT_LOCAL_SWITCH_MIN_GAIN = 0.05
SOFT_LOCAL_CONTINUITY_REQUIRE_CONFIRMED = True
SOFT_LOCAL_CONTINUITY_REQUIRE_SAME_GENERATION = True
TRACKLET_PROTOTYPE_KEEP_RECENT_SAMPLES = True
TRACKLET_PROTOTYPE_REJECT_NEAR_DUPLICATES = False
PERSISTENT_TRACKLET_BUFFER_ENABLED = True
PERSISTENT_TRACKLET_BUFFER_MAX_SAMPLES = 8
PERSISTENT_TRACKLET_BUFFER_MAX_AGE_SEC = 2.0
IDENTITY_PROTOTYPE_ENABLED = True
IDENTITY_PROTOTYPE_MAX_SAMPLES = 7
IDENTITY_PROTOTYPE_MIN_SAMPLES = 2
IDENTITY_PROTOTYPE_WEIGHT = 0.75
IDENTITY_PROTOTYPE_SUPPORT_WEIGHT = 0.25
IDENTITY_PROTOTYPE_MIN_CONSENSUS = 0.70
MERGE_GUARD_ENABLED = True
MERGE_GUARD_APPEARANCE_MARGIN = 0.025
MERGE_GUARD_SAME_CAM_MIN_MARGIN = 0.08
MERGE_GUARD_CROSS_CAM_MIN_MARGIN = 0.12
PAIRWISE_SWAP_CORRECTION_ENABLED = True
PAIRWISE_SWAP_MOTION_WEIGHT = 0.80
PAIRWISE_SWAP_APPEARANCE_WEIGHT = 0.15
PAIRWISE_SWAP_SIZE_WEIGHT = 0.05
PAIRWISE_SWAP_MIN_AVG_GAIN = 0.03
PAIRWISE_SWAP_MIN_ROW_GAIN = 0.02
PAIRWISE_SWAP_MIN_MOTION_GAIN = 0.05
PAIRWISE_SWAP_MAX_GROUP = 12
PAIRWISE_SNAPSHOT_HISTORY = 3
PAIRWISE_SNAPSHOT_MAX_GAP_SEC = 0.35
PAIRWISE_MATRIX_DIAGNOSTICS_ENABLED = True
PAIRWISE_MATRIX_DIAGNOSTIC_MAX_RECORDS = 600
LOCAL_TRANSITION_RECOVERY_ENABLED = True
LOCAL_TRANSITION_MAX_GAP_SEC = 0.45
LOCAL_TRANSITION_HISTORY = 3
LOCAL_TRANSITION_MOTION_WEIGHT = 0.70
LOCAL_TRANSITION_APPEARANCE_WEIGHT = 0.20
LOCAL_TRANSITION_SIZE_WEIGHT = 0.10
LOCAL_TRANSITION_MIN_SCORE = 0.72
LOCAL_TRANSITION_MIN_MARGIN = 0.08
LOCAL_TRANSITION_MIN_MOTION = 0.55
LOCAL_TRANSITION_LOST_TTL_SEC = 1.20
LOCAL_TRANSITION_REAPPEAR_MIN_GAP_SEC = 0.05
LOCAL_TRANSITION_ALLOW_STALE_MAPPING = True
LOCAL_TRANSITION_MAX_LOST_PER_CAMERA = 32
DELAYED_TRANSITION_ENABLED = True
DELAYED_TRANSITION_REQUIRED_VOTES = 3
DELAYED_TRANSITION_MAX_VOTE_GAP_SEC = 0.30
DELAYED_TRANSITION_STRONG_SCORE = 0.86
DELAYED_TRANSITION_STRONG_MOTION = 0.82
DELAYED_TRANSITION_STRONG_MARGIN = 0.14
DELAYED_TRANSITION_NORMAL_SCORE = 0.74
DELAYED_TRANSITION_NORMAL_MOTION = 0.60
DELAYED_TRANSITION_NORMAL_MARGIN = 0.08
WRONG_GID_ESCAPE_ENABLED = True
WRONG_GID_ESCAPE_REQUIRED_VOTES = 3
WRONG_GID_ESCAPE_MAX_VOTE_GAP_SEC = 0.30
WRONG_GID_ESCAPE_MIN_CHALLENGER_SCORE = 0.74
WRONG_GID_ESCAPE_MIN_SCORE_ADVANTAGE = 0.08
WRONG_GID_ESCAPE_STRONG_SCORE = 0.90
WRONG_GID_ESCAPE_STRONG_ADVANTAGE = 0.20
WRONG_GID_ESCAPE_MIN_OWNER_SCORE = 0.45
PAIRWISE_MOTION_DISTANCE_SCALE = 1.25
PAIRWISE_DIAGNOSTIC_TOP_K = 20
PAIRWISE_DIAGNOSTIC_GAIN_LEVELS = (0.02, 0.04, 0.06, 0.08, 0.10)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def frame_to_base64(frame):
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )

    if not ok:
        return None

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def image_file_to_base64(path):
    if not os.path.exists(path):
        return None

    img = cv2.imread(path)

    if img is None:
        return None

    return frame_to_base64(img)


def open_camera_once(camera_url):
    cap = cv2.VideoCapture(
        parse_video_source(camera_url)
    )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    return frame


def parse_json_points(points_text):
    pts = json.loads(points_text)

    if not isinstance(pts, list) or len(pts) != 4:
        raise ValueError("ต้องมี 4 จุด")

    for p in pts:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError("รูปแบบจุดต้องเป็น [x, y]")

    return pts


def json_response(
    success: bool,
    message: str,
    data: dict = None,
    status_code: int = 200
):
    content = {
        "success": success,
        "message": message
    }

    if data:
        content.update(data)

    return JSONResponse(content, status_code=status_code)


def parse_video_source(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid video source")

    if isinstance(value, int):
        if value < 0:
            raise ValueError("Camera index must be zero or greater")

        return value

    if not isinstance(value, str):
        raise ValueError("Video source must be a camera index or URL")

    source = value.strip()

    if not source:
        raise ValueError("Video source is required")

    if source.isdigit():
        return int(source)

    parsed = urlparse(source)

    allowed_schemes = {
        "rtsp",
        "http",
        "https",
        "rtmp"
    }

    if parsed.scheme.lower() not in allowed_schemes:
        raise ValueError(
            f"ไม่รองรับ scheme: {parsed.scheme}"
        )

    return source


def validate_camera_url(url: str):
    return parse_video_source(url)


def mask_video_source(value):
    if not isinstance(value, str):
        return value

    parsed = urlparse(value)

    if parsed.username is None and parsed.password is None:
        return value

    hostname = parsed.hostname or ""

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    try:
        port = parsed.port
    except ValueError:
        port = None

    if port is not None:
        hostname = f"{hostname}:{port}"

    masked_netloc = f"***:***@{hostname}"

    return parsed._replace(
        netloc=masked_netloc
    ).geturl()


def sanitize_source_error(message, source):
    text = str(message)

    if not isinstance(source, str):
        return text

    masked_source = mask_video_source(source)
    text = text.replace(source, masked_source)
    parsed = urlparse(source)

    for secret in (parsed.username, parsed.password):
        if secret:
            text = text.replace(secret, "***")

    return text


def safe_filename(filename: str):
    filename = os.path.basename(filename)

    keepchars = (
        ".",
        "_",
        "-"
    )

    cleaned = "".join(
        c for c in filename
        if c.isalnum() or c in keepchars
    ).strip()

    if not cleaned or cleaned.startswith("."):
        return f"video_{int(time.time())}.mp4"

    return cleaned


def clamp_bbox(
    x1,
    y1,
    x2,
    y2,
    w,
    h
):
    x1 = max(
        0,
        min(int(x1), w - 1)
    )

    y1 = max(
        0,
        min(int(y1), h - 1)
    )

    x2 = max(
        0,
        min(int(x2), w - 1)
    )

    y2 = max(
        0,
        min(int(y2), h - 1)
    )

    if x2 <= x1:
        x2 = min(
            w - 1,
            x1 + 1
        )

    if y2 <= y1:
        y2 = min(
            h - 1,
            y1 + 1
        )

    return x1, y1, x2, y2


def l2_normalize(vec):
    return _reid_l2_normalize(vec)


def cosine_similarity(a, b):
    return _reid_cosine_similarity(a, b)

    return value


try:
    from .topology import (
        TOPOLOGY_SCHEMA_VERSION,
        TOPOLOGY_SUPPORTED_VERSIONS,
        configure_topology_dependencies,
        default_topology_config,
        load_topology_config,
        normalize_topology_config,
        validate_topology_config,
        point_in_polygon,
    )
except ImportError:  # pragma: no cover - direct backend/main.py execution
    from topology import (
        configure_topology_dependencies,
        default_topology_config,
        load_topology_config,
        normalize_topology_config,
        validate_topology_config,
        point_in_polygon,
    )

configure_topology_dependencies(globals())
topology_lock = threading.Lock()
topology_config = load_topology_config()

# Re-ID extractor implementation and fallback now live in backend.reid.
configure_extractor_dependencies(globals())
OSNetFeatureExtractor = _ReIDOSNetFeatureExtractor
LightweightAppearanceFeatureExtractor = _ReIDLightweightAppearanceFeatureExtractor
load_validated_osnet_checkpoint = _reid_load_validated_osnet_checkpoint
resolve_reid_device = _reid_resolve_reid_device
build_feature_extractor = _reid_build_feature_extractor


# ============================================================
# LIGHTWEIGHT FALLBACK
# ============================================================

# OSNET MARKET1501

        # ใช้ FeatureExtractor โดยตรง
        # และระบุ Market1501 weight
        # ----------------------------------------------------









class CameraProcessor:

    def __init__(
        self,
        cam_id,
        src_pts,
        dst_pts
    ):

        self.cam_id = cam_id

        self.src_pts = np.array(
            src_pts,
            dtype=np.float32
        )

        self.dst_pts = np.array(
            dst_pts,
            dtype=np.float32
        )

        if (
            self.src_pts.shape != (4, 2)
            or
            self.dst_pts.shape != (4, 2)
        ):
            raise ValueError(
                "src_pts และ dst_pts ต้องมี 4 จุด"
            )

        self.H, _ = cv2.findHomography(
            self.src_pts,
            self.dst_pts
        )

        if self.H is None:
            raise ValueError(
                "คำนวณ Homography ไม่สำเร็จ"
            )

    def to_floorplan(
        self,
        px,
        py
    ):

        pt = np.array(
            [[[px, py]]],
            dtype=np.float32
        )

        transformed = cv2.perspectiveTransform(
            pt,
            self.H
        )

        map_x, map_y = transformed[0][0]

        return (
            int(map_x),
            int(map_y)
        )

    def draw_calibration_polygon(
        self,
        frame
    ):

        pts = self.src_pts.astype(
            np.int32
        ).reshape(
            (-1, 1, 2)
        )

        cv2.polylines(
            frame,
            [pts],
            True,
            (255, 200, 0),
            2
        )

        for i, p in enumerate(
            self.src_pts.astype(np.int32)
        ):

            x, y = (
                int(p[0]),
                int(p[1])
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                f"P{i + 1}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

        return frame


# ============================================================
# GLOBAL IDENTITY MANAGER
# ============================================================











            # ReID สูงมาก

            # ต้องผ่านทั้ง appearance + score




                        # Local ID + Appearance ตรงกัน
                        # Local ID เดิม แต่ Appearance ไม่ตรง
                        # อย่าเชื่อ Local ID
                        # ปล่อยลง global matching

class GlobalAssignmentCoordinator:
    """Bound identity observations without coordinating camera lifecycles.

    ``submit`` only copies observation metadata into a latest-per-camera slot
    and starts a daemon timer.  Camera processing therefore never waits for a
    missing or late camera; the timer atomically submits whatever observations
    are present to ``assign_global_batch``.
    """

    def __init__(
        self,
        manager_provider,
        window_sec=GLOBAL_ASSIGNMENT_WINDOW_SEC,
        max_pending_cameras=GLOBAL_ASSIGNMENT_MAX_PENDING_CAMERAS,
        max_observations_per_camera=(
            GLOBAL_ASSIGNMENT_MAX_OBSERVATIONS_PER_CAMERA
        ),
        max_ready_batches=GLOBAL_ASSIGNMENT_MAX_READY_BATCHES,
    ):
        self.manager_provider = manager_provider
        self.window_sec = max(0.0, float(window_sec))
        self.max_pending_cameras = max(1, int(max_pending_cameras))
        self.max_observations_per_camera = max(
            1,
            int(max_observations_per_camera)
        )
        self.max_ready_batches = max(1, int(max_ready_batches))
        self.lock = threading.Lock()
        self.pending = {}
        self.ready_batches = deque()
        self.camera_epochs = {}
        self.timer = None
        self.dispatcher_thread = None
        self.inflight_batch_id = None
        self.current_batch_id = None
        self.batch_sequence = 0
        self.replaced_submission_count = 0
        self.capacity_drop_count = 0
        self.ready_batch_drop_count = 0
        self.last_error = None
        self.last_completed_batch_id = None
        self.last_submit_duration_ms = None
        self.last_assignment_duration_ms = None

    def _new_batch_id_locked(self):
        self.batch_sequence += 1
        return f"global-{self.batch_sequence:08d}"

    def _take_pending_locked(self):
        if not self.pending:
            return None

        batch = {
            "batch_id": self.current_batch_id,
            "submissions": self.pending,
        }
        self.pending = {}
        self.current_batch_id = None

        timer = self.timer
        self.timer = None
        if timer is not None:
            timer.cancel()

        return batch

    def _start_timer_locked(self):
        expected_batch_id = self.current_batch_id
        timer = threading.Timer(
            self.window_sec,
            self._flush_timer,
            args=(expected_batch_id,),
        )
        timer.daemon = True
        self.timer = timer
        timer.start()

    def _enqueue_ready_batch_locked(self, batch):
        if len(self.ready_batches) >= self.max_ready_batches:
            self.ready_batches.popleft()
            self.ready_batch_drop_count += 1
        self.ready_batches.append(batch)

        if (
            self.dispatcher_thread is None
            or not self.dispatcher_thread.is_alive()
        ):
            dispatcher = threading.Thread(
                target=self._dispatch_ready_batches,
                daemon=True,
                name="GlobalAssignmentDispatcher",
            )
            self.dispatcher_thread = dispatcher
            dispatcher.start()

    def _dispatch_ready_batches(self):
        while True:
            with self.lock:
                if not self.ready_batches:
                    self.dispatcher_thread = None
                    self.inflight_batch_id = None
                    return
                batch = self.ready_batches.popleft()
                self.inflight_batch_id = batch["batch_id"]

            self._execute_batch(batch)

            with self.lock:
                self.inflight_batch_id = None

    def _event_time_outside_pending_window_locked(self, event_time):
        pending_times = [
            submission["event_time"]
            for submission in self.pending.values()
        ]
        if not pending_times:
            return False
        return (
            event_time < min(pending_times) - self.window_sec
            or event_time > max(pending_times) + self.window_sec
        )

    def submit(
        self,
        cam_name,
        detections,
        prev_assignments=None,
        event_time=None,
    ):
        """Submit without waiting; results become local evidence next frame."""
        submit_started = time.perf_counter()
        if not detections:
            return []

        observation_event_time = (
            time.time()
            if event_time is None
            else float(event_time)
        )
        bounded_detections = [
            {
                **detection,
                "event_time": float(
                    detection.get(
                        "event_time",
                        observation_event_time
                    )
                ),
            }
            for detection in detections[
                :self.max_observations_per_camera
            ]
        ]
        # Snapshot and attach the coordinator epoch before trusted preview.
        # Do not hold this lock while entering the identity manager: batch
        # dispatch uses manager -> coordinator lock ordering.
        with self.lock:
            self.camera_epochs.setdefault(cam_name, 0)
            preview_camera_epoch = self.camera_epochs[cam_name]
        for detection in bounded_detections:
            detection["coordinator_generation"] = (
                preview_camera_epoch
            )

        preview_results = (
            self.manager_provider().preview_trusted_assignments(
                cam_name,
                bounded_detections,
                event_time=observation_event_time,
                blocking=False,
            )
        )
        preview_results.extend(
            None
            for _ in range(len(detections) - len(bounded_detections))
        )
        with self.lock:
            camera_epoch = self.camera_epochs.get(
                cam_name,
                preview_camera_epoch,
            )
            if camera_epoch != preview_camera_epoch:
                # A reset raced the read-only preview.  Never render its stale
                # claim or submit it under the previous camera epoch.
                preview_results = [None for _ in detections]
                for detection in bounded_detections:
                    detection["coordinator_generation"] = camera_epoch

            if self._event_time_outside_pending_window_locked(
                observation_event_time
            ):
                batch = self._take_pending_locked()
                if batch is not None:
                    self._enqueue_ready_batch_locked(batch)

            if self.current_batch_id is None:
                self.current_batch_id = self._new_batch_id_locked()

            if cam_name in self.pending:
                self.replaced_submission_count += 1
            elif len(self.pending) >= self.max_pending_cameras:
                oldest_camera = next(iter(self.pending))
                self.pending.pop(oldest_camera, None)
                self.capacity_drop_count += 1

            reserved_gids = {
                gid
                for pending_camera, submission in self.pending.items()
                if pending_camera != cam_name
                for gid in submission["preview_gids"]
            }
            for index, result in enumerate(preview_results):
                if result is not None and result["gid"] in reserved_gids:
                    preview_results[index] = None

            self.pending[cam_name] = {
                "detections": bounded_detections,
                "prev_assignments": list(prev_assignments or []),
                "event_time": observation_event_time,
                "preview_gids": {
                    result["gid"]
                    for result in preview_results
                    if result is not None
                },
                "camera_epoch": camera_epoch,
            }

            if self.timer is None:
                self._start_timer_locked()

            self.last_submit_duration_ms = (
                (time.perf_counter() - submit_started) * 1000.0
            )

        # The originating frame continues through the existing preview path.
        # Only read-only trusted evidence is visible until the atomic global
        # batch commits; new/ambiguous observations safely remain unlabeled.
        return preview_results

    def _flush_timer(self, expected_batch_id):
        with self.lock:
            if self.current_batch_id != expected_batch_id:
                return
            batch = self._take_pending_locked()
            if batch is not None:
                self._enqueue_ready_batch_locked(batch)

    def _execute_batch(self, batch):
        manager = self.manager_provider()
        assignment_started = time.perf_counter()
        try:
            with manager.lock:
                with self.lock:
                    submissions = {
                        cam_name: submission
                        for cam_name, submission in batch["submissions"].items()
                        if submission["camera_epoch"]
                        == self.camera_epochs.get(cam_name, 0)
                    }
                if not submissions:
                    return

                camera_detections = {
                    cam_name: submission["detections"]
                    for cam_name, submission in submissions.items()
                }
                previous = {
                    cam_name: submission["prev_assignments"]
                    for cam_name, submission in submissions.items()
                }
                event_times = [
                    submission["event_time"]
                    for submission in submissions.values()
                ]
                manager.assign_global_batch(
                    camera_detections,
                    prev_assignments_by_camera=previous,
                    event_time=max(event_times),
                    batch_id=batch["batch_id"],
                    assignment_window_sec=self.window_sec,
                )
        except Exception as error:
            with self.lock:
                self.last_error = str(error)
            logger.error(
                "[REID][GLOBAL] Coordinator batch failed | batch=%s error=%s",
                batch["batch_id"],
                error,
                exc_info=True,
            )
            return

        with self.lock:
            self.last_error = None
            self.last_completed_batch_id = batch["batch_id"]
            self.last_assignment_duration_ms = (
                (time.perf_counter() - assignment_started) * 1000.0
            )

    def flush(self):
        """Synchronously flush pending identity work for tests/shutdown tools."""
        with self.lock:
            ready_batches = list(self.ready_batches)
            self.ready_batches.clear()
            batch = self._take_pending_locked()

        for ready_batch in ready_batches:
            self._execute_batch(ready_batch)
        if batch is not None:
            self._execute_batch(batch)

        return bool(ready_batches or batch is not None)

    def discard_camera(self, cam_name):
        with self.lock:
            self.camera_epochs[cam_name] = (
                self.camera_epochs.get(cam_name, 0) + 1
            )
            removed = self.pending.pop(cam_name, None) is not None
            if not self.pending:
                timer = self.timer
                self.timer = None
                self.current_batch_id = None
                if timer is not None:
                    timer.cancel()
        manager = self.manager_provider()
        discard_unresolved = getattr(
            manager,
            "discard_unresolved_handoffs",
            None,
        )
        if callable(discard_unresolved):
            discard_unresolved(cam_name)
        return removed

    def stop(self):
        """Cancel uncommitted observations without touching camera workers."""
        with self.lock:
            pending_count = sum(
                len(item["detections"])
                for item in self.pending.values()
            )
            pending_count += sum(
                len(submission["detections"])
                for batch in self.ready_batches
                for submission in batch["submissions"].values()
            )
            for cam_name in self.camera_epochs:
                self.camera_epochs[cam_name] = (
                    self.camera_epochs.get(cam_name, 0) + 1
                )
            self.pending = {}
            self.ready_batches.clear()
            self.current_batch_id = None
            timer = self.timer
            self.timer = None
            if timer is not None:
                timer.cancel()
        manager = self.manager_provider()
        discard_unresolved = getattr(
            manager,
            "discard_unresolved_handoffs",
            None,
        )
        if callable(discard_unresolved):
            discard_unresolved()
        else:
            with manager.lock:
                pass
        return pending_count

    def status(self):
        with self.lock:
            manager = self.manager_provider()
            diagnostics = getattr(
                manager,
                "last_global_batch_diagnostics",
                None,
            )
            return {
                "assignment_window_sec": self.window_sec,
                "pending_batch_id": self.current_batch_id,
                "pending_cameras": sorted(self.pending),
                "pending_observation_count": sum(
                    len(item["detections"])
                    for item in self.pending.values()
                ),
                "pending_observations": [
                    {
                        "batch_id": self.current_batch_id,
                        "camera": cam_name,
                        "event_time": submission["event_time"],
                        "track_ids": [
                            detection["tid"]
                            for detection in submission["detections"]
                        ],
                        "generation": submission["camera_epoch"],
                        "assignment_state": "pending",
                    }
                    for cam_name, submission in self.pending.items()
                ],
                "ready_batch_count": len(self.ready_batches),
                "inflight_batch_id": self.inflight_batch_id,
                "last_completed_batch_id": self.last_completed_batch_id,
                "replaced_submission_count": self.replaced_submission_count,
                "capacity_drop_count": self.capacity_drop_count,
                "ready_batch_drop_count": self.ready_batch_drop_count,
                "last_submit_duration_ms": self.last_submit_duration_ms,
                "last_assignment_duration_ms": (
                    self.last_assignment_duration_ms
                ),
                "last_error": self.last_error,
                "last_batch": diagnostics,
            }


# ============================================================
# GLOBAL MAP MANAGER
# ============================================================

class GlobalMapManager:

    def __init__(
        self,
        trail_len=50,
        timeout_sec=2.0,
        floorplan_path=None
    ):

        self.trail_len = trail_len

        self.timeout_sec = timeout_sec

        self.base_map = None

        self.objects = {}

        self.tracks = {}

        self.last_seen = {}

        self.lock = threading.Lock()

        self.floorplan_path = floorplan_path or FLOORPLAN_PATH

        self.load_floorplan()


    def load_floorplan(self):

        if os.path.exists(
            self.floorplan_path
        ):

            img = cv2.imread(
                self.floorplan_path
            )

            if img is not None:

                self.base_map = img

                return


        self.base_map = np.zeros(
            (600, 900, 3),
            dtype=np.uint8
        )

        cv2.putText(
            self.base_map,
            "No Floorplan Uploaded",
            (220, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )


    def update_object(
        self,
        global_id,
        map_x,
        map_y
    ):

        if self.base_map is None:
            return

        with self.lock:

            h, w = (
                self.base_map.shape[:2]
            )

            map_x = max(
                0,
                min(
                    int(map_x),
                    w - 1
                )
            )

            map_y = max(
                0,
                min(
                    int(map_y),
                    h - 1
                )
            )

            self.objects[
                global_id
            ] = (
                map_x,
                map_y
            )

            self.last_seen[
                global_id
            ] = time.time()


            if global_id not in self.tracks:

                self.tracks[
                    global_id
                ] = deque(
                    maxlen=self.trail_len
                )


            self.tracks[
                global_id
            ].append(
                (
                    map_x,
                    map_y
                )
            )


    def cleanup_stale_objects(self):

        now = time.time()

        stale_ids = [

            gid

            for gid, ts
            in self.last_seen.items()

            if (
                now - ts
                >
                self.timeout_sec
            )

        ]

        for gid in stale_ids:

            self.last_seen.pop(
                gid,
                None
            )

            self.objects.pop(
                gid,
                None
            )

            self.tracks.pop(
                gid,
                None
            )


    def draw_map(self):

        with self.lock:

            self.cleanup_stale_objects()

            canvas = (
                self.base_map.copy()
            )


            for gid, (
                mx,
                my
            ) in self.objects.items():

                cv2.circle(
                    canvas,
                    (mx, my),
                    8,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    canvas,
                    f"ID {gid}",
                    (
                        mx + 10,
                        my - 8
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


            return canvas


# ============================================================
# INITIALIZE REID
# ============================================================

appearance_extractor = (
    build_feature_extractor()
)

configure_embedding_extractor(appearance_extractor)

RESET_ID_ON_START = True

if RESET_ID_ON_START:
    if os.path.exists(IDENTITY_DB_PATH):
        os.remove(IDENTITY_DB_PATH)
        logger.info("[IDENTITY] Database reset: %s", IDENTITY_DB_PATH)

# Global identity implementation lives in backend.identity.manager.  Inject the
# production dependency namespace after bootstrap definitions are available so
# extracted methods retain the exact existing runtime behavior.
try:
    from .identity import manager as _identity_manager_module
except ImportError:  # pragma: no cover - direct backend/main.py execution
    from identity import manager as _identity_manager_module

_identity_manager_module.configure_identity_dependencies(globals())
GlobalIdentityManager = _identity_manager_module.GlobalIdentityManager

global_identity_manager = (
    GlobalIdentityManager(
        IdentityStore(IDENTITY_DB_PATH)
    )
)

global_assignment_coordinator = GlobalAssignmentCoordinator(
    lambda: global_identity_manager,
    window_sec=GLOBAL_ASSIGNMENT_WINDOW_SEC,
)

global_map = GlobalMapManager(
    trail_len=1,
    timeout_sec=0.7
)
global_maps_lock = threading.Lock()
global_maps = {}


def get_floorplan_map_manager(floorplan_name):
    if not floorplan_name:
        return global_map
    path, safe_name = floorplan_path_by_name(floorplan_name)
    with global_maps_lock:
        manager = global_maps.get(safe_name)
        if manager is None:
            manager = GlobalMapManager(
                trail_len=1,
                timeout_sec=0.7,
                floorplan_path=path,
            )
            global_maps[safe_name] = manager
        return manager


def reset_global_identity_session(reason="new_playback_session"):
    """Start a fresh Global-ID namespace without changing camera/player workers."""
    global global_identity_manager

    # Flush any pending global-assignment work before replacing the manager.
    try:
        global_assignment_coordinator.flush()
    except Exception as error:
        logger.warning(
            "[IDENTITY] Coordinator flush before reset failed: %s",
            error,
        )

    old_manager = global_identity_manager
    old_store = getattr(old_manager, "identity_store", None)

    # Close the SQLite handle before recreating the identity database.
    if old_store is not None:
        try:
            old_store.close()
        except Exception as error:
            logger.warning(
                "[IDENTITY] Store close before reset failed: %s",
                error,
            )

    if os.path.exists(IDENTITY_DB_PATH):
        try:
            os.remove(IDENTITY_DB_PATH)
        except OSError as error:
            logger.warning(
                "[IDENTITY] Could not remove old DB during replay reset: %s",
                error,
            )

    global_identity_manager = GlobalIdentityManager(
        IdentityStore(IDENTITY_DB_PATH)
    )

    logger.info(
        "[IDENTITY] New identity session | reason=%s | next_gid=%s",
        reason,
        global_identity_manager.next_global_id,
    )

    return {
        "reason": str(reason),
        "next_global_id": int(global_identity_manager.next_global_id),
    }


# ============================================================
# PER-CAMERA BOT-SORT CONTEXT
# ============================================================

def new_camera_tracker_context():
    return {
        "tracking_model": None,
        "tracker_lock": threading.RLock(),
        "tracker_instance_id": None,
        "tracker_generation": 0,
        "tracker_reset_count": 0,
        "tracker_last_reset_reason": None,
        "tracker_created_at": None,
        "tracker_last_frame_index": None,
        "tracker_last_event_time": None,
        "canonical_event_time_floor": None,
        "tracker_source_time_sec": None,
        "tracker_time_offset_sec": 0.0,
        "tracker_last_update": None,
        "downstream_timing": None,
        "video_last_processing_error": None,
        "active_local_tracks": []
    }


def _ensure_camera_tracker_context(
    cam_data
):
    defaults = new_camera_tracker_context()

    for key, value in defaults.items():
        cam_data.setdefault(key, value)

    return cam_data


def canonical_observation_event_time(cam_data, event_time=None):
    """Keep downstream observation metadata on a non-decreasing timeline."""
    candidate = time.time() if event_time is None else float(event_time)
    if not np.isfinite(candidate):
        raise ValueError("Observation event time must be finite")
    previous = cam_data.get("canonical_event_time_floor")
    if (
        isinstance(previous, (int, float))
        and not isinstance(previous, bool)
        and np.isfinite(float(previous))
    ):
        candidate = max(candidate, float(previous))
    cam_data["canonical_event_time_floor"] = candidate
    return candidate


def get_camera_tracking_model(cam_name):
    """Return the camera's private YOLO predictor and BoT-SORT state."""
    with cameras_lock:
        cam_data = cameras.get(cam_name)

        if cam_data is None:
            raise KeyError(
                f"Camera not found: {cam_name}"
            )

        _ensure_camera_tracker_context(
            cam_data
        )
        tracker_lock = cam_data[
            "tracker_lock"
        ]

    with tracker_lock:
        tracking_model = cam_data.get(
            "tracking_model"
        )

        if tracking_model is None:
            tracking_model = YOLO(
                YOLO_MODEL_PATH
            )

            with cameras_lock:
                if cameras.get(cam_name) is not cam_data:
                    raise RuntimeError(
                        f"Camera was removed while creating tracker: {cam_name}"
                    )

                cam_data["tracking_model"] = (
                    tracking_model
                )
                cam_data["tracker_generation"] = int(
                    cam_data.get(
                        "tracker_generation",
                        0
                    )
                ) + 1
                cam_data["tracker_instance_id"] = (
                    f"{cam_name}:"
                    f"{uuid.uuid4().hex[:12]}"
                )
                cam_data["tracker_created_at"] = (
                    time.time()
                )

            logger.info(
                "[TRACKER] Created | camera=%s | instance=%s | generation=%s",
                cam_name,
                cam_data["tracker_instance_id"],
                cam_data["tracker_generation"]
            )

        return tracking_model


def reset_camera_tracker(
    cam_name,
    reason="manual"
):
    """Reset only one camera's tracker and camera-scoped Re-ID mappings."""
    with cameras_lock:
        cam_data = cameras.get(cam_name)

        if cam_data is None:
            return None

        _ensure_camera_tracker_context(
            cam_data
        )
        tracker_lock = cam_data[
            "tracker_lock"
        ]

    with tracker_lock:
        previous_instance = cam_data.get(
            "tracker_instance_id"
        )
        cam_data["tracking_model"] = None
        cam_data["tracker_instance_id"] = None
        cam_data["tracker_created_at"] = None
        cam_data["tracker_last_frame_index"] = None
        cam_data["tracker_last_event_time"] = None
        cam_data["tracker_source_time_sec"] = None
        cam_data["tracker_last_update"] = None
        cam_data["active_local_tracks"] = []
        cam_data["prev_assignments"] = []
        # Local ReID cache is scoped to the current tracker generation.
        # Clearing it prevents a recycled local track ID from inheriting an
        # embedding that belonged to a previous person.
        cam_data["reid_embedding_cache"] = {}
        cam_data["tracker_reset_count"] = int(
            cam_data.get(
                "tracker_reset_count",
                0
            )
        ) + 1
        cam_data["tracker_last_reset_reason"] = (
            str(reason)
        )
        global_assignment_coordinator.discard_camera(
            cam_name
        )
        local_cleanup = (
            global_identity_manager
            .reset_camera_local_state(
                cam_name
            )
        )

    logger.info(
        "[TRACKER] Reset | camera=%s | previous_instance=%s | reason=%s | local_mappings=%s",
        cam_name,
        previous_instance,
        reason,
        local_cleanup["local_mappings_removed"]
    )

    return {
        "camera_id": cam_name,
        "previous_instance_id": previous_instance,
        "reason": str(reason),
        **local_cleanup
    }


def get_camera_tracker_status(
    cam_name,
    cam_data=None
):
    if cam_data is None:
        with cameras_lock:
            cam_data = cameras.get(cam_name)

    if cam_data is None:
        return None

    with cameras_lock:
        _ensure_camera_tracker_context(
            cam_data
        )
        tracker_lock = cam_data[
            "tracker_lock"
        ]

    with tracker_lock:
        tracking_model = cam_data.get(
            "tracking_model"
        )
        predictor = getattr(
            tracking_model,
            "predictor",
            None
        )
        botsort_states = getattr(
            predictor,
            "trackers",
            None
        ) or []

        return {
            "camera_id": cam_name,
            "local_track_scope": "camera",
            "initialized": tracking_model is not None,
            "tracker_instance_id": cam_data.get(
                "tracker_instance_id"
            ),
            "tracker_generation": int(
                cam_data.get(
                    "tracker_generation",
                    0
                )
            ),
            "tracker_reset_count": int(
                cam_data.get(
                    "tracker_reset_count",
                    0
                )
            ),
            "last_reset_reason": cam_data.get(
                "tracker_last_reset_reason"
            ),
            "botsort_state_count": len(
                botsort_states
            ),
            "botsort_state_ids": [
                f"0x{id(state):x}"
                for state in botsort_states
            ],
            "active_local_track_count": len(
                cam_data.get(
                    "active_local_tracks",
                    []
                )
            ),
            "active_local_tracks": list(
                cam_data.get(
                    "active_local_tracks",
                    []
                )
            ),
            "last_frame_index": cam_data.get(
                "tracker_last_frame_index"
            ),
            "last_event_time": cam_data.get(
                "tracker_last_event_time"
            ),
            "source_time_sec": cam_data.get(
                "tracker_source_time_sec"
            ),
            "configured_offset_sec": cam_data.get(
                "tracker_time_offset_sec",
                0.0
            ),
            "last_update": cam_data.get(
                "tracker_last_update"
            ),
            "downstream_timing": cam_data.get(
                "downstream_timing"
            ),
        }

# ============================================================
# MULTI-CAMERA VIDEO SYNCHRONIZER
# ============================================================

class MultiCameraVideoManager:
    def __init__(self):
        self.lock = threading.Lock()

        self.videos = {}
        self.frames = {}
        self.frame_indices = {}
        self.running = {}

        self.thread = None
        self.started = False

    def register_video(
        self,
        cam_name,
        video_path,
        loop_video=False,
        time_offset_sec=0.0
    ):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not cap.isOpened():
            raise RuntimeError(
                f"ไม่สามารถเปิด Video ของ {cam_name}: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prime calibration with the first frame without advancing playback.
        # The worker still begins at frame 1 after the user presses Play.
        ok, initial_frame = cap.read()
        if not ok or initial_frame is None:
            cap.release()
            raise RuntimeError(
                f"Cannot read the first frame of {cam_name}: {video_path}"
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        with self.lock:
            self.videos[cam_name] = {
                "cap": cap,
                "fps": float(fps),
                "total_frames": total_frames,
                "loop_video": bool(loop_video),
                "frame_index": 0,
                "time_offset_sec": float(time_offset_sec),
                "playback_started_at": None,
                "last_source_time_sec": None,
                "last_event_time": None,
                "tracker_reset_pending": False,
            }

            self.frames[cam_name] = initial_frame.copy()
            self.frame_indices[cam_name] = 0
            # Wait for an explicit playback command so multiple uploaded
            # clips can start on the same worker iteration.
            self.running[cam_name] = False

        logger.info(
            f"[SYNC] Registered {cam_name} | "
            f"FPS={fps:.2f} | Frames={total_frames}"
        )
        return initial_frame

    def remove_video(self, cam_name):
        with self.lock:
            data = self.videos.pop(cam_name, None)

            self.frames.pop(cam_name, None)
            self.frame_indices.pop(cam_name, None)
            self.running.pop(cam_name, None)

            if data is not None:
                try:
                    data["cap"].release()
                except Exception:
                    pass

    def get_camera_names(self):
        with self.lock:
            return list(self.videos.keys())

    def get_playback_states(self):
        with self.lock:
            return {
                cam_name: bool(
                    self.running.get(
                        cam_name,
                        False
                    )
                )
                for cam_name in self.videos
            }

    def set_playback(self, camera_names, is_playing):
        names = list(dict.fromkeys(camera_names))

        with self.lock:
            missing = [
                cam_name
                for cam_name in names
                if cam_name not in self.videos
            ]

            if missing:
                raise KeyError(
                    ", ".join(missing)
                )

            for cam_name in names:
                data = self.videos[cam_name]

                if (
                    is_playing
                    and data["total_frames"] > 0
                    and data["frame_index"] >= data["total_frames"]
                ):
                    if not data["loop_video"]:
                        self.running[cam_name] = False
                        continue
                    data["cap"].set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )
                    data["frame_index"] = 0
                    self.frame_indices[cam_name] = 0
                    data["tracker_reset_pending"] = True

                was_playing = self.running.get(cam_name, False)
                if is_playing and not was_playing:
                    data["playback_started_at"] = time.time()
                    data["last_source_time_sec"] = None
                    data["last_event_time"] = None

                self.running[cam_name] = bool(
                    is_playing
                )

            return {
                cam_name: bool(
                    self.running[cam_name]
                )
                for cam_name in names
            }

    def read_synchronized_frames(self):
        """
        อ่าน Frame ของทุก Video ในรอบเดียวกัน

        รอบที่ 1:
            CAM1 -> Frame 1
            CAM2 -> Frame 1
            CAM3 -> Frame 1

        รอบที่ 2:
            CAM1 -> Frame 2
            CAM2 -> Frame 2
            CAM3 -> Frame 2
        """

        with self.lock:

            if not self.videos:
                return None

            result = {}

            for cam_name, data in self.videos.items():

                if not self.running.get(cam_name, False):
                    continue

                cap = data["cap"]

                ret, frame = cap.read()

                if not ret:

                    if data["loop_video"]:

                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                        ret, frame = cap.read()

                        if not ret:
                            self.running[cam_name] = False
                            continue

                        data["frame_index"] = 0
                        data["tracker_reset_pending"] = True
                        data["playback_started_at"] = time.time()
                        data["last_source_time_sec"] = None
                        data["last_event_time"] = None

                    else:
                        self.running[cam_name] = False
                        continue

                data["frame_index"] += 1

                frame_index = data["frame_index"]
                source_time_sec = frame_index / max(data["fps"], 1e-6)
                playback_started_at = data.get("playback_started_at")
                if playback_started_at is None:
                    playback_started_at = time.time()
                    data["playback_started_at"] = playback_started_at
                canonical_event_time = (
                    playback_started_at
                    + source_time_sec
                    + data.get("time_offset_sec", 0.0)
                )
                last_event_time = data.get("last_event_time")
                if last_event_time is not None:
                    canonical_event_time = max(
                        canonical_event_time,
                        last_event_time + (1.0 / max(data["fps"], 1e-6))
                    )
                data["last_source_time_sec"] = source_time_sec
                data["last_event_time"] = canonical_event_time

                self.frames[cam_name] = frame.copy()
                self.frame_indices[cam_name] = frame_index

                result[cam_name] = {
                    "frame": frame.copy(),
                    "frame_index": frame_index,
                    "fps": data["fps"],
                    "source_time_sec": source_time_sec,
                    "event_time": canonical_event_time,
                    "time_offset_sec": data.get("time_offset_sec", 0.0),
                    "source_reset": bool(
                        data.get(
                            "tracker_reset_pending",
                            False
                        )
                    ),
                }
                data["tracker_reset_pending"] = False

            if not result:
                return None

            return result

    def get_frame(self, cam_name):
        with self.lock:

            frame = self.frames.get(cam_name)

            if frame is None:
                return None

            return frame.copy()

    def get_frame_index(self, cam_name):
        with self.lock:
            return self.frame_indices.get(cam_name, 0)

    def get_fps(self, cam_name):
        with self.lock:

            data = self.videos.get(cam_name)

            if data is None:
                return 25.0

            return data["fps"]

    def release_all(self):

        with self.lock:

            for data in self.videos.values():

                try:
                    data["cap"].release()
                except Exception:
                    pass

            self.videos.clear()
            self.frames.clear()
            self.frame_indices.clear()
            self.running.clear()

        logger.info("[SYNC] All videos released")

multi_video_manager = MultiCameraVideoManager()

# ============================================================
# PROCESSED FRAME BUFFER
# ============================================================

processed_frames = {}
processed_frame_locks = {}

video_worker_lock = threading.Lock()
video_worker_running = False
video_worker_thread = None


def publish_processed_frame(
    cam_name,
    annotated_frame,
    cam_data=None
):
    ok, buffer = cv2.imencode(
        ".jpg",
        annotated_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )

    if not ok:
        return False

    frame_bytes = buffer.tobytes()

    with cameras_lock:
        current = cameras.get(cam_name)

        if current is None:
            return False

        if cam_data is not None and current is not cam_data:
            return False

        with video_worker_lock:
            processed_frames[cam_name] = frame_bytes

            condition = processed_frame_locks.setdefault(
                cam_name,
                threading.Condition()
            )

            with condition:
                condition.notify_all()

    return True


# ============================================================
# LIVE CAMERA REALTIME WORKERS
# ============================================================

class LiveCameraWorker:
    """Bounded latest-frame capture and processing for one live source."""

    def __init__(
        self,
        cam_name,
        source,
        reconnect_interval=None
    ):
        self.cam_name = cam_name
        self.source = parse_video_source(source)
        self.reconnect_interval = (
            LIVE_CAMERA_RECONNECT_INTERVAL_SEC
            if reconnect_interval is None
            else max(0.01, float(reconnect_interval))
        )
        self.instance_id = (
            f"{cam_name}:{uuid.uuid4().hex[:12]}"
        )

        self.state_lock = threading.RLock()
        self.frame_condition = threading.Condition(
            self.state_lock
        )
        self.stop_event = threading.Event()
        self.capture = None
        self.capture_thread = None
        self.processing_thread = None
        self.started = False
        self.capture_open = False
        self.processing = False

        # A single replaceable slot gives latest-frame semantics without an
        # unbounded queue when inference is slower than the source FPS.
        self._latest_frame = None
        self._tracker_reset_pending = False
        self._ever_captured = False
        self._open_attempts = 0

        self.frame_index = 0
        self.captured_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.reconnect_count = 0
        self.last_frame_event_time = None
        self.last_frame_monotonic = None
        self.last_processing_started = None
        self.last_processing_finished = None
        self.last_processing_duration_ms = None
        self.last_processing_latency_ms = None
        self.last_capture_error = None
        self.last_processing_error = None
        self._processing_timestamps = deque(maxlen=120)

    def start(self):
        with self.state_lock:
            if (
                self.capture_thread is not None
                and self.capture_thread.is_alive()
            ) or (
                self.processing_thread is not None
                and self.processing_thread.is_alive()
            ):
                return False

            if self.stop_event.is_set():
                raise RuntimeError(
                    "A stopped live worker cannot be started again"
                )

            self.started = True
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name=f"LiveCapture-{self.cam_name}"
            )
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name=f"LiveProcess-{self.cam_name}"
            )
            capture_thread = self.capture_thread
            processing_thread = self.processing_thread

        processing_thread.start()
        capture_thread.start()

        logger.info(
            "[LIVE] Worker started | camera=%s | instance=%s",
            self.cam_name,
            self.instance_id
        )
        return True

    def stop(self, timeout=LIVE_CAMERA_STOP_TIMEOUT_SEC):
        self.stop_event.set()

        with self.frame_condition:
            capture = self.capture
            self._latest_frame = None
            self.frame_condition.notify_all()

        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

        deadline = time.monotonic() + max(0.0, float(timeout))

        for thread in (
            self.capture_thread,
            self.processing_thread
        ):
            if thread is None or thread is threading.current_thread():
                continue

            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

        with self.state_lock:
            self.capture_open = False
            stopped = not any(
                thread is not None and thread.is_alive()
                for thread in (
                    self.capture_thread,
                    self.processing_thread
                )
            )

        if not stopped:
            logger.warning(
                "[LIVE] Worker stop timed out | camera=%s",
                self.cam_name
            )
        else:
            logger.info(
                "[LIVE] Worker stopped | camera=%s",
                self.cam_name
            )

        return stopped

    def _set_capture_error(self, message):
        with self.state_lock:
            self.last_capture_error = sanitize_source_error(
                message,
                self.source
            )
            self.capture_open = False

    def _release_capture(self, capture):
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

        with self.state_lock:
            if self.capture is capture:
                self.capture = None

            self.capture_open = False

    def _capture_diagnostic(self, capture, read_result):
        def capture_property(property_id):
            try:
                return float(capture.get(property_id))
            except Exception:
                return None

        def format_number(value):
            if value is None or not np.isfinite(value):
                return "unknown"

            if value.is_integer():
                return str(int(value))

            return f"{value:.3f}".rstrip("0").rstrip(".")

        try:
            backend = capture.getBackendName()
        except Exception:
            backend = None

        if not backend:
            backend = (
                "V4L2"
                if (
                    sys.platform.startswith("linux")
                    and isinstance(self.source, int)
                )
                else "unknown"
            )

        try:
            opened = bool(capture.isOpened())
        except Exception:
            opened = False

        fourcc_value = capture_property(cv2.CAP_PROP_FOURCC)
        fourcc = "unknown"

        if fourcc_value is not None and np.isfinite(fourcc_value):
            fourcc_code = int(fourcc_value)
            decoded_fourcc = "".join(
                chr((fourcc_code >> (8 * offset)) & 0xFF)
                for offset in range(4)
            )
            fourcc = (
                decoded_fourcc
                if all(32 <= ord(char) <= 126 for char in decoded_fourcc)
                else str(fourcc_code)
            )

        read_value = (
            "not-attempted"
            if read_result is None
            else str(bool(read_result))
        )

        return (
            f"backend={backend} "
            f"device/index={mask_video_source(self.source)} "
            f"fourcc={fourcc} "
            f"width={format_number(capture_property(cv2.CAP_PROP_FRAME_WIDTH))} "
            f"height={format_number(capture_property(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"fps={format_number(capture_property(cv2.CAP_PROP_FPS))} "
            f"opened={opened} "
            f"read={read_value}"
        )

    def _open_capture(self):
        with self.state_lock:
            is_reconnect = self._open_attempts > 0
            self._open_attempts += 1

            if is_reconnect:
                self.reconnect_count += 1

        try:
            if (
                sys.platform.startswith("linux")
                and isinstance(self.source, int)
            ):
                capture = cv2.VideoCapture(
                    self.source,
                    cv2.CAP_V4L2
                )
                capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG")
                )
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 30)
            else:
                capture = cv2.VideoCapture(self.source)

            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            opened = bool(capture.isOpened())
        except Exception as error:
            self._set_capture_error(
                f"Capture open failed: {error}"
            )
            return None

        if not opened:
            self._set_capture_error(
                "Capture did not open | "
                + self._capture_diagnostic(capture, None)
            )
            self._release_capture(capture)
            return None

        if self.stop_event.is_set():
            self._release_capture(capture)
            return None

        with self.state_lock:
            self.capture = capture
            self.capture_open = True
            self.last_capture_error = None

        return capture

    def _publish_captured_frame(
        self,
        frame,
        event_time,
        monotonic_time
    ):
        with self.frame_condition:
            self.frame_index += 1
            self.captured_frames += 1
            self.last_frame_event_time = event_time
            self.last_frame_monotonic = monotonic_time
            self.last_capture_error = None
            self._ever_captured = True

            source_reset = self._tracker_reset_pending
            self._tracker_reset_pending = False

            if self._latest_frame is not None:
                self.dropped_frames += 1
                source_reset = (
                    source_reset
                    or self._latest_frame.get(
                        "source_reset",
                        False
                    )
                )

            self._latest_frame = {
                "frame": frame,
                "frame_index": self.frame_index,
                "event_time": event_time,
                "capture_monotonic": monotonic_time,
                "source_reset": source_reset
            }
            self.frame_condition.notify()

    def _capture_loop(self):
        capture = None

        try:
            while not self.stop_event.is_set():
                if capture is None:
                    capture = self._open_capture()

                    if capture is None:
                        if self.stop_event.wait(
                            self.reconnect_interval
                        ):
                            break

                        continue

                try:
                    success, frame = capture.read()
                except Exception as error:
                    success = False
                    frame = None
                    read_reason = f"Capture read raised: {error}"
                else:
                    read_reason = "Capture read returned no frame"

                if not success or frame is None:
                    if self.stop_event.is_set():
                        break

                    failure_kind = (
                        "Capture first-frame read failed"
                        if not self._ever_captured
                        else "Capture read failed"
                    )
                    read_error = (
                        f"{failure_kind}: {read_reason} | "
                        + self._capture_diagnostic(
                            capture,
                            success
                        )
                    )
                    self._set_capture_error(read_error)

                    with self.state_lock:
                        if self._ever_captured:
                            self._tracker_reset_pending = True

                    self._release_capture(capture)
                    capture = None

                    logger.warning(
                        "[LIVE] Source unavailable; reconnecting | "
                        "camera=%s | error=%s",
                        self.cam_name,
                        sanitize_source_error(
                            read_error,
                            self.source
                        )
                    )

                    if self.stop_event.wait(
                        self.reconnect_interval
                    ):
                        break

                    continue

                self._publish_captured_frame(
                    frame,
                    time.time(),
                    time.monotonic()
                )
        except Exception as error:
            safe_error = sanitize_source_error(
                error,
                self.source
            )
            self._set_capture_error(
                f"Capture worker failed: {safe_error}"
            )
            logger.error(
                "[LIVE] Capture worker failed | camera=%s | error=%s",
                self.cam_name,
                safe_error
            )
        finally:
            self._release_capture(capture)

            with self.frame_condition:
                self.frame_condition.notify_all()

    def _processing_loop(self):
        while not self.stop_event.is_set():
            with self.frame_condition:
                while (
                    self._latest_frame is None
                    and not self.stop_event.is_set()
                ):
                    self.frame_condition.wait(timeout=0.5)

                if self.stop_event.is_set():
                    self._latest_frame = None
                    break

                item = self._latest_frame
                self._latest_frame = None
                self.processing = True

            started_monotonic = time.monotonic()

            with self.state_lock:
                self.last_processing_started = time.time()

            try:
                if item.get("source_reset"):
                    reset_camera_tracker(
                        self.cam_name,
                        reason="live_source_reconnected"
                    )

                with cameras_lock:
                    cam_data = cameras.get(self.cam_name)

                    if cam_data is None:
                        continue

                    cam_data["last_frame"] = item["frame"].copy()
                    cam_data["last_frame_event_time"] = item[
                        "event_time"
                    ]

                annotated_frame = process_camera_frame(
                    self.cam_name,
                    item["frame"],
                    item["frame_index"],
                    event_time=item["event_time"]
                )

                if not publish_processed_frame(
                    self.cam_name,
                    annotated_frame,
                    cam_data=cam_data
                ):
                    raise RuntimeError(
                        "Processed frame could not be published"
                    )

                finished_monotonic = time.monotonic()

                with self.state_lock:
                    self.processed_frames += 1
                    self.last_processing_error = None
                    self._processing_timestamps.append(
                        finished_monotonic
                    )
                    self.last_processing_duration_ms = (
                        (finished_monotonic - started_monotonic)
                        * 1000.0
                    )
                    self.last_processing_latency_ms = (
                        (
                            finished_monotonic
                            - item["capture_monotonic"]
                        )
                        * 1000.0
                    )
            except Exception as error:
                safe_error = sanitize_source_error(
                    error,
                    self.source
                )

                with self.state_lock:
                    self.last_processing_error = safe_error

                if not self.stop_event.is_set():
                    logger.error(
                        "[LIVE] Frame processing failed | camera=%s | frame=%s | error=%s",
                        self.cam_name,
                        item.get("frame_index"),
                        safe_error
                    )
            finally:
                with self.state_lock:
                    self.processing = False
                    self.last_processing_finished = time.time()

    def status(self):
        with self.state_lock:
            capture_alive = bool(
                self.capture_thread is not None
                and self.capture_thread.is_alive()
            )
            processing_alive = bool(
                self.processing_thread is not None
                and self.processing_thread.is_alive()
            )
            now_monotonic = time.monotonic()

            if self.last_frame_monotonic is None:
                last_frame_age_ms = None
            else:
                last_frame_age_ms = max(
                    0.0,
                    (
                        now_monotonic
                        - self.last_frame_monotonic
                    ) * 1000.0
                )

            timestamps = list(
                self._processing_timestamps
            )

            if len(timestamps) >= 2:
                processing_fps = (
                    (len(timestamps) - 1)
                    / max(
                        timestamps[-1] - timestamps[0],
                        1e-6
                    )
                )
            else:
                processing_fps = 0.0

            return {
                "worker_instance_id": self.instance_id,
                "running": bool(
                    not self.stop_event.is_set()
                    and capture_alive
                    and processing_alive
                ),
                "capture_thread_alive": capture_alive,
                "processing_thread_alive": processing_alive,
                "capture_open": bool(self.capture_open),
                "source": mask_video_source(self.source),
                "frame_index": int(self.frame_index),
                "captured_frames": int(self.captured_frames),
                "processed_frames": int(self.processed_frames),
                "dropped_frames": int(self.dropped_frames),
                "frame_queue_capacity": 1,
                "latest_frame_pending": self._latest_frame is not None,
                "processing": bool(self.processing),
                "last_frame_event_time": self.last_frame_event_time,
                "last_frame_age_ms": (
                    None
                    if last_frame_age_ms is None
                    else round(last_frame_age_ms, 3)
                ),
                "last_processing_started": self.last_processing_started,
                "last_processing_finished": self.last_processing_finished,
                "last_processing_duration_ms": (
                    None
                    if self.last_processing_duration_ms is None
                    else round(
                        self.last_processing_duration_ms,
                        3
                    )
                ),
                "last_processing_latency_ms": (
                    None
                    if self.last_processing_latency_ms is None
                    else round(
                        self.last_processing_latency_ms,
                        3
                    )
                ),
                "processing_fps": round(processing_fps, 3),
                "reconnect_count": int(self.reconnect_count),
                "last_error": (
                    self.last_processing_error
                    or self.last_capture_error
                )
            }


class LiveCameraManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.workers = {}

    def start_worker(self, cam_name, source):
        with self.lock:
            existing = self.workers.get(cam_name)

            if existing is not None:
                return existing, False

            worker = LiveCameraWorker(
                cam_name,
                source
            )
            self.workers[cam_name] = worker

        try:
            worker.start()
        except Exception:
            with self.lock:
                if self.workers.get(cam_name) is worker:
                    self.workers.pop(cam_name, None)

            raise

        return worker, True

    def stop_worker(
        self,
        cam_name,
        timeout=LIVE_CAMERA_STOP_TIMEOUT_SEC
    ):
        with self.lock:
            worker = self.workers.pop(
                cam_name,
                None
            )

        if worker is None:
            return False

        worker.stop(timeout=timeout)
        return True

    def restart_worker(self, cam_name, source):
        self.stop_worker(cam_name)
        return self.start_worker(cam_name, source)

    def get_status(self, cam_name):
        with self.lock:
            worker = self.workers.get(cam_name)

        if worker is None:
            return None

        return worker.status()

    def stop_all(self):
        with self.lock:
            workers = list(self.workers.values())
            self.workers.clear()

        for worker in workers:
            worker.stop()


live_camera_manager = LiveCameraManager()

# ============================================================
# STREAMING UTILITIES
# ============================================================

# ============================================================
# PERSON EMBEDDING
# ============================================================

def extract_person_crop(
    frame,
    x1,
    y1,
    x2,
    y2
):
    return _reid_extract_person_crop(frame, x1, y1, x2, y2)


def extract_person_crop_without_margin(
    frame,
    x1,
    y1,
    x2,
    y2
):
    return _reid_extract_person_crop_without_margin(
        frame, x1, y1, x2, y2
    )


def get_reid_crop(
    frame,
    x1,
    y1,
    x2,
    y2,
    crop_mode=None
):
    return _reid_get_reid_crop(frame, x1, y1, x2, y2, crop_mode)


def extract_person_embedding(
    frame,
    x1,
    y1,
    x2,
    y2
):
    """Compatibility wrapper for code paths that still need one embedding."""
    return _reid_extract_person_embedding(
        frame, x1, y1, x2, y2, appearance_extractor
    )


# ============================================================
# BBOX FUNCTIONS
# ============================================================

def bbox_iou(
    boxA,
    boxB
):

    ax1, ay1, ax2, ay2 = boxA

    bx1, by1, bx2, by2 = boxB


    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )


    inter_w = max(
        0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1
    )

    inter_area = (
        inter_w
        *
        inter_h
    )


    areaA = (
        max(
            1,
            ax2 - ax1
        )
        *
        max(
            1,
            ay2 - ay1
        )
    )

    areaB = (
        max(
            1,
            bx2 - bx1
        )
        *
        max(
            1,
            by2 - by1
        )
    )


    union = (
        areaA
        +
        areaB
        -
        inter_area
    )


    if union <= 0:
        return 0.0


    return (
        inter_area
        /
        union
    )


def bbox_center(
    box
):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5
    )


def center_distance(
    boxA,
    boxB
):

    ax, ay = bbox_center(
        boxA
    )

    bx, by = bbox_center(
        boxB
    )

    return float(
        np.hypot(
            ax - bx,
            ay - by
        )
    )


# ============================================================
# FORCED GID MAP
# ============================================================

def build_forced_gid_map(
    cam_name,
    detection_boxes,
    event_time=None,
):

    forced = {}
    try:
        reference_ts = float(
            time.time()
            if event_time is None
            else event_time
        )
    except (TypeError, ValueError, OverflowError):
        return forced
    if not np.isfinite(reference_ts):
        return forced

    cam_state = cameras.get(
        cam_name,
        {}
    )

    prev_assignments = (
        cam_state.get(
            "prev_assignments",
            []
        )
    )


    if (
        not prev_assignments
        or
        not detection_boxes
    ):

        return forced


    pairs = []


    for det_idx, det_box in enumerate(
        detection_boxes
    ):

        for prev in prev_assignments:

            if not isinstance(prev, dict):
                continue

            previous_ts = prev.get("ts")
            if not isinstance(previous_ts, (int, float)):
                continue
            previous_ts = float(previous_ts)
            if not np.isfinite(previous_ts):
                continue
            age = reference_ts - previous_ts
            if age < 0.0 or age > OCCLUSION_HOLD_SEC:
                continue

            prev_gid = prev.get(
                "gid"
            )

            prev_box = prev.get(
                "box"
            )

            if (
                prev_gid is None
                or
                prev_box is None
            ):

                continue


            iou = bbox_iou(
                det_box,
                prev_box
            )

            dist = center_distance(
                det_box,
                prev_box
            )


            if (
                iou
                >=
                OCCLUSION_PREV_IOU_THRESHOLD
                or
                dist
                <=
                OCCLUSION_CENTER_DIST_PX
            ):

                score = (
                    iou * 2.0
                    -
                    (
                        dist
                        /
                        max(
                            OCCLUSION_CENTER_DIST_PX,
                            1.0
                        )
                    )
                    *
                    0.25
                )

                pairs.append(
                    (
                        score,
                        det_idx,
                        prev_gid
                    )
                )


    pairs.sort(
        reverse=True
    )


    used_det = set()

    used_gid = set()


    for (
        score,
        det_idx,
        gid
    ) in pairs:

        if det_idx in used_det:
            continue

        if gid in used_gid:
            continue

        forced[
            det_idx
        ] = gid

        used_det.add(
            det_idx
        )

        used_gid.add(
            gid
        )


    return forced


# ============================================================
# GENERATE CAMERA FRAMES
# ============================================================

def process_camera_frame(
    cam_name,
    frame,
    frame_index,
    event_time=None
):
    with cameras_lock:
        cam_data = cameras.get(cam_name)

        if cam_data is None:
            return frame

        _ensure_camera_tracker_context(
            cam_data
        )
        tracker_lock = cam_data[
            "tracker_lock"
        ]

    # The entire per-camera pipeline is serialized with its tracker state.
    # Other cameras use different locks and can process independently.
    with tracker_lock:
        tracking_model = get_camera_tracking_model(
            cam_name
        )
        return _process_camera_frame_locked(
            cam_name,
            frame,
            frame_index,
            cam_data,
            tracking_model,
            event_time=event_time
        )


def _process_camera_frame_locked(
    cam_name,
    frame,
    frame_index,
    cam_data,
    tracking_model,
    event_time=None
):
    """
    ประมวลผล Frame ของ Camera หนึ่งตัว

    YOLO
    -> BoT-SORT
    -> OSNet ReID
    -> Global ID
    -> Homography
    -> Global Map
    """

    downstream_started = time.perf_counter()
    tracking_duration_ms = 0.0
    reid_feature_duration_ms = 0.0
    coordinator_submit_duration_ms = 0.0
    reid_observation_count = 0
    reid_inference_count = 0
    reid_cache_hit_count = 0
    annotated_frame = frame.copy()

    # Per-local-track appearance cache. This is processing state only; it does
    # not alter capture, synchronization, or camera lifecycle behavior.
    reid_cache = cam_data.setdefault(
        "reid_embedding_cache",
        {}
    )
    active_local_track_ids = []
    event_ts = canonical_observation_event_time(
        cam_data,
        event_time=event_time,
    )

    # --------------------------------------------------------
    # YOLO + BoT-SORT
    # --------------------------------------------------------

    tracking_started = time.perf_counter()
    results = tracking_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.55,
        tracker="botsort.yaml",
        device=YOLO_DEVICE,
        half=YOLO_USE_HALF,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False
    )
    tracking_duration_ms = (
        (time.perf_counter() - tracking_started) * 1000.0
    )

    processor = cam_data.get("processor")
    src_pts = cam_data.get("src_pts")

    if processor is not None:
        annotated_frame = processor.draw_calibration_polygon(
            annotated_frame
        )

    frame_assignments = []

    prev_assignments = cam_data.get(
        "prev_assignments",
        []
    )

    if results and len(results) > 0:

        result = results[0]
        boxes = result.boxes

        if boxes is not None and boxes.xyxy is not None:

            xyxy_list = boxes.xyxy.cpu().numpy()

            track_ids = None

            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                active_local_track_ids = sorted({
                    int(track_id)
                    for track_id in track_ids
                })

            confs = None

            if boxes.conf is not None:
                confs = boxes.conf.cpu().numpy().tolist()

            filtered = []
            pending_detections = []
            reid_batch_crops = []
            reid_batch_indices = []

            # ------------------------------------------------
            # Detection + local ReID cache preparation
            # ------------------------------------------------

            for i, box in enumerate(xyxy_list):

                x1, y1, x2, y2 = box[:4]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                # ROI (existing behavior preserved)
                if processor is not None and src_pts is not None:
                    inside = point_in_polygon(
                        (foot_x, foot_y),
                        src_pts
                    )
                    if not inside:
                        continue

                tid = (
                    int(track_ids[i])
                    if track_ids is not None
                    and i < len(track_ids)
                    else -(
                        (max(int(frame_index), 0) + 1)
                        * 1_000_000
                        + i
                        + 1
                    )
                )

                local_track_confirmed = bool(
                    track_ids is not None
                    and i < len(track_ids)
                )

                conf_val = (
                    float(confs[i])
                    if confs is not None
                    and i < len(confs)
                    else None
                )

                box_wh = (
                    max(1, x2 - x1),
                    max(1, y2 - y1)
                )

                # Existing Homography/floorplan calculation is preserved.
                map_pos = None
                if processor is not None:
                    try:
                        map_x, map_y = processor.to_floorplan(
                            foot_x,
                            foot_y
                        )
                        map_pos = (map_x, map_y)
                    except Exception:
                        map_pos = None

                quality_meta = build_reid_quality_metadata(
                    frame,
                    (x1, y1, x2, y2),
                    conf_val,
                )

                item = {
                    "idx": i,
                    "box": (x1, y1, x2, y2),
                    "foot": (foot_x, foot_y),
                    "tid": tid,
                    "frame_index": int(frame_index),
                    "camera_generation": int(
                        cam_data.get("tracker_generation", 0)
                    ),
                    "local_track_confirmed": local_track_confirmed,
                    "conf": conf_val,
                    "box_wh": box_wh,
                    "emb": None,
                    "reid_fresh": False,
                    **quality_meta,
                    "map_pos": map_pos,
                    "center": bbox_center(
                        (x1, y1, x2, y2)
                    ),
                    "event_time": event_ts,
                }

                need_reid = True

                # Wait for a real BoT-SORT local ID before spending OSNet work.
                if (
                    REID_REQUIRE_CONFIRMED_TRACK
                    and not local_track_confirmed
                ):
                    need_reid = False

                # Reuse a recent embedding for an already-confirmed local track.
                if local_track_confirmed:
                    cached = reid_cache.get(tid)
                    if isinstance(cached, dict):
                        last_reid_frame = int(
                            cached.get("frame_index", -999999)
                        )
                        frame_gap = (
                            int(frame_index) - last_reid_frame
                        )
                        cached_embedding = cached.get("emb")
                        if (
                            cached_embedding is not None
                            and frame_gap >= 0
                            and frame_gap < REID_INFERENCE_INTERVAL
                        ):
                            item["emb"] = cached_embedding
                            need_reid = False
                            reid_cache_hit_count += 1

                pending_index = len(pending_detections)
                pending_detections.append(item)

                if need_reid:
                    crop = get_reid_crop(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                    if crop is not None:
                        reid_batch_crops.append(crop)
                        reid_batch_indices.append(pending_index)

            # ------------------------------------------------
            # OSNet batch inference
            # ------------------------------------------------

            if reid_batch_crops:
                reid_started = time.perf_counter()

                if hasattr(appearance_extractor, "extract_batch"):
                    batch_embeddings = appearance_extractor.extract_batch(
                        reid_batch_crops
                    )
                else:
                    batch_embeddings = [
                        appearance_extractor.extract(crop)
                        for crop in reid_batch_crops
                    ]

                reid_feature_duration_ms += (
                    (time.perf_counter() - reid_started) * 1000.0
                )
                reid_inference_count += len(reid_batch_crops)

                for pending_index, new_emb in zip(
                    reid_batch_indices,
                    batch_embeddings
                ):
                    if new_emb is None:
                        continue

                    item = pending_detections[pending_index]
                    tid = item["tid"]

                    # Smooth refreshed local-track appearance. The long-lived
                    # Global ID gallery/prototype logic remains unchanged.
                    previous = None
                    if item["local_track_confirmed"]:
                        cached = reid_cache.get(tid)
                        if isinstance(cached, dict):
                            previous = cached.get("emb")

                    if previous is not None:
                        try:
                            previous = np.asarray(
                                previous, dtype=np.float32
                            ).reshape(-1)
                            current = np.asarray(
                                new_emb, dtype=np.float32
                            ).reshape(-1)
                            if previous.size == current.size:
                                new_emb = l2_normalize(
                                    REID_LOCAL_EMA_ALPHA * previous
                                    + (1.0 - REID_LOCAL_EMA_ALPHA) * current
                                )
                        except (TypeError, ValueError, OverflowError):
                            pass

                    item["emb"] = new_emb
                    item["reid_fresh"] = True

                    global_identity_manager.add_fresh_tracklet_embedding(
                        cam_name,
                        item["tid"],
                        new_emb,
                        event_ts,
                        generation=item.get(
                            "coordinator_generation",
                            item.get("camera_generation"),
                        ),
                    )

                    if item["local_track_confirmed"]:
                        reid_cache[tid] = {
                            "emb": new_emb,
                            "frame_index": int(frame_index),
                        }

            # Only detections with a usable fresh/cached embedding continue to
            # the existing occlusion/global-ID path.
            filtered = [
                item
                for item in pending_detections
                if item.get("emb") is not None
            ]

            # ------------------------------------------------
            # Occlusion
            # ------------------------------------------------

            overlap_indices = set()

            for a in range(len(filtered)):

                for b in range(a + 1, len(filtered)):

                    if bbox_iou(
                        filtered[a]["box"],
                        filtered[b]["box"]
                    ) >= OCCLUSION_IOU_THRESHOLD:

                        overlap_indices.add(a)
                        overlap_indices.add(b)

            forced_gid_map = build_forced_gid_map(
                cam_name,
                [
                    item["box"]
                    for item in filtered
                ],
                event_time=event_ts,
            )

            for a, item in enumerate(filtered):

                item["overlap"] = (
                    a in overlap_indices
                )

                item["overlap"] = bool(item["overlap"])

                item["forced_gid"] = (
                    forced_gid_map.get(a)
                    if a in overlap_indices
                    else None
                )

            # ------------------------------------------------
            # GLOBAL ID
            # ------------------------------------------------

            reid_observation_count = len(filtered)
            coordinator_submit_started = time.perf_counter()
            assignment_results = (
                global_assignment_coordinator.submit(
                    cam_name,
                    filtered,
                    prev_assignments=prev_assignments,
                    event_time=event_ts
                )
                if filtered
                else []
            )
            coordinator_submit_duration_ms = (
                (
                    time.perf_counter()
                    - coordinator_submit_started
                )
                * 1000.0
            )

            # ------------------------------------------------
            # Draw result
            # ------------------------------------------------

            for a, item in enumerate(filtered):

                res = assignment_results[a]

                x1, y1, x2, y2 = item["box"]

                foot_x, foot_y = item["foot"]

                tid = item["tid"]

                if res is None:
                    label = (
                        f"L{tid}"
                        if item["local_track_confirmed"]
                        else "L?"
                    )
                    label += " | GID pending"
                else:
                    gid = res["gid"]

                    try:
                        embedding_store.save_if_selected(
                            global_id=gid,
                            embedding=item["emb"],
                            camera_name=cam_name,
                        )
                    except Exception as exc:
                        print(f"[EmbeddingDB] save failed: {exc}")    

                    match_score = res["score"]

                    match_source = res["source"]

                    label = f"GID {gid}"

                if item["conf"] is not None:
                    label += f" {item['conf']:.2f}"

                if REID_DEBUG and res is not None:
                    label += (
                        f" | L{tid}"
                        if item[
                            "local_track_confirmed"
                        ]
                        else " | L?"
                    )

                box_color = (
                    (0, 165, 255)
                    if a in overlap_indices
                    else (0, 255, 0)
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2
                )

                if res is None:
                    continue

                if REID_DEBUG:

                    debug_line = (
                        f"{match_source}:"
                        f"{match_score:.2f}"
                    )

                    if a in overlap_indices:
                        debug_line += " | OCC"

                    cv2.putText(
                        annotated_frame,
                        debug_line,
                        (
                            x1,
                            min(
                                frame.shape[0] - 8,
                                y2 + 18
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1
                    )

                cv2.circle(
                    annotated_frame,
                    (foot_x, foot_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                # ------------------------------------------------
                # Global Map
                # ------------------------------------------------

                if item["map_pos"] is not None:

                    map_x, map_y = item["map_pos"]

                    cv2.putText(
                        annotated_frame,
                        f"map=({map_x},{map_y})",
                        (
                            foot_x + 8,
                            foot_y - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

                    get_floorplan_map_manager(
                        cam_data.get("floorplan_name")
                    ).update_object(
                        gid,
                        map_x,
                        map_y
                    )

                frame_assignments.append({
                    "gid": gid,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "center": item["center"],
                    "tid": tid,
                    "cam_name": cam_name,
                    "overlap": (
                        a in overlap_indices
                    ),
                    "ts": event_ts,
                })

    # --------------------------------------------------------
    # Save previous assignments
    # --------------------------------------------------------

    cam_data["prev_assignments"] = (
        prev_assignments + frame_assignments
    )[-60:]

    # Remove embeddings for local tracks that have disappeared long enough to
    # make local-ID reuse unsafe.
    active_set = set(active_local_track_ids)
    stale_tids = []
    for cached_tid, cache_item in list(reid_cache.items()):
        try:
            last_frame = int(
                cache_item.get("frame_index", -999999)
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            stale_tids.append(cached_tid)
            continue

        age = int(frame_index) - last_frame
        if (
            cached_tid not in active_set
            and age > REID_CACHE_MAX_AGE_FRAMES
        ):
            stale_tids.append(cached_tid)

    for cached_tid in stale_tids:
        reid_cache.pop(cached_tid, None)

    cam_data["active_local_tracks"] = (
        active_local_track_ids
    )
    cam_data["tracker_last_frame_index"] = int(
        frame_index
    )
    cam_data["tracker_last_event_time"] = event_ts

    total_downstream_ms = (
        (time.perf_counter() - downstream_started) * 1000.0
    )
    processing_fps = (
        1000.0 / total_downstream_ms
        if total_downstream_ms > 0.0
        else 0.0
    )

    cam_data["downstream_timing"] = {
        "frame_index": int(frame_index),
        "event_time": event_ts,
        "detection_tracking_ms": float(tracking_duration_ms),
        "reid_feature_ms": float(reid_feature_duration_ms),
        "coordinator_submit_ms": float(
            coordinator_submit_duration_ms
        ),
        "total_downstream_ms": float(total_downstream_ms),
        "processing_fps": float(processing_fps),
        "observation_count": int(reid_observation_count),
        "reid_inference_count": int(reid_inference_count),
        "reid_cache_hit_count": int(reid_cache_hit_count),
        "reid_cache_size": int(len(reid_cache)),
    }
    cam_data["tracker_last_update"] = time.time()

    # --------------------------------------------------------
    # Frame information
    # --------------------------------------------------------

    cv2.putText(
        annotated_frame,
        (
            f"{cam_name} "
            f"[Frame {frame_index}] "
            f"ReID:{appearance_extractor.name}"
        ),
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    return annotated_frame

def multi_camera_worker():
    global video_worker_running

    logger.info(
        "[SYNC] Multi-camera worker started"
    )

    last_tick = time.time()

    while app.is_running and video_worker_running:

        # ================================================
        # อ่าน Frame ของทุกกล้องในรอบเดียว
        # ================================================

        frames_data = (
            multi_video_manager
            .read_synchronized_frames()
        )

        if frames_data is None:

            time.sleep(0.01)
            continue

        # ================================================
        # ประมวลผลทุกกล้อง
        # ================================================

        processed_batch = []

        for cam_name, data in frames_data.items():

            frame = data["frame"]

            frame_index = data["frame_index"]

            try:

                if data.get("source_reset"):
                    reset_camera_tracker(
                        cam_name,
                        reason="video_source_rewind"
                    )

                # เก็บ frame ล่าสุด
                with cameras_lock:
                    cam_data = cameras.get(
                        cam_name
                    )

                    if cam_data is None:
                        continue

                    cam_data["last_frame"] = (
                        frame.copy()
                    )

                # YOLO + BoT + ReID + Homography
                annotated_frame = process_camera_frame(
                    cam_name,
                    frame,
                    frame_index,
                    event_time=data["event_time"]
                )

                with cameras_lock:
                    if cameras.get(cam_name) is cam_data:
                        cam_data["tracker_source_time_sec"] = data[
                            "source_time_sec"
                        ]
                        cam_data["tracker_time_offset_sec"] = data[
                            "time_offset_sec"
                        ]

                processed_batch.append((cam_name, annotated_frame, cam_data))
                with cameras_lock:
                    if cameras.get(cam_name) is cam_data:
                        cam_data["video_last_processing_error"] = None

            except Exception as e:

                with cameras_lock:
                    current = cameras.get(cam_name)
                    if current is not None:
                        current["video_last_processing_error"] = str(e)

                logger.error(
                    f"[SYNC] Error processing "
                    f"{cam_name}: {e}",
                    exc_info=True
                )

        # Publish only after every camera in this synchronized read cycle has
        # completed processing.  This prevents the UI from showing camera A
        # from a new cycle while camera B is still displaying the prior one.
        for cam_name, annotated_frame, cam_data in processed_batch:
            publish_processed_frame(
                cam_name,
                annotated_frame,
                cam_data=cam_data
            )

        # ================================================
        # ควบคุม FPS
        # ================================================

        fps_values = []

        for cam_name in frames_data:

            fps_values.append(
                frames_data[cam_name]["fps"]
            )

        if fps_values:

            target_fps = min(fps_values)

            target_delay = (
                1.0 / max(target_fps, 1.0)
            )

            elapsed = time.time() - last_tick

            sleep_time = target_delay - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        last_tick = time.time()

    logger.info(
        "[SYNC] Multi-camera worker stopped"
    )


def start_multi_camera_worker():

    global video_worker_running
    global video_worker_thread

    if video_worker_running:
        return

    video_worker_running = True

    video_worker_thread = threading.Thread(
        target=multi_camera_worker,
        daemon=True,
        name="MultiCameraVideoWorker"
    )

    video_worker_thread.start()

    logger.info(
        "[SYNC] Worker started"
    )


def stop_multi_camera_worker():

    global video_worker_running

    video_worker_running = False

    video_camera_names = (
        multi_video_manager
        .get_camera_names()
    )

    multi_video_manager.release_all()

    for cam_name in video_camera_names:
        reset_camera_tracker(
            cam_name,
            reason="video_worker_stopped"
        )

    logger.info(
        "[SYNC] Worker stopped"
    )

def generate_frames(cam_name: str):

    with cameras_lock:
        cam_data = cameras.get(cam_name)

        if cam_data is None:
            return

        source_type = cam_data.get("source_type")

    # ถ้า Worker ยังไม่ทำงาน ให้เริ่ม
    if source_type == "video":
        start_multi_camera_worker()

    while app.is_running:

        # ============================================
        # รอ Frame ล่าสุดจาก Worker
        # ============================================

        frame_bytes = None

        with video_worker_lock:

            frame_bytes = processed_frames.get(
                cam_name
            )

        if frame_bytes is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        else:

            time.sleep(0.01)

# ============================================================
# GLOBAL MAP STREAM
# ============================================================

def generate_global_map(floorplan_name=None):

    map_manager = get_floorplan_map_manager(floorplan_name)

    while app.is_running:

        canvas = (
            map_manager.draw_map()
        )


        ok, buffer = cv2.imencode(

            ".jpg",

            canvas,

            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                85
            ]

        )


        if not ok:
            continue


        frame_bytes = (
            buffer.tobytes()
        )


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            +
            frame_bytes

            +
            b"\r\n"

        )


        time.sleep(
            0.08
        )


# ============================================================
# ROUTES
# ============================================================

@app.get("/api/status")
async def get_status():

    floorplan_exists = (
        os.path.exists(
            FLOORPLAN_PATH
        )
    )

    cams_data = {}

    playback_states = (
        multi_video_manager
        .get_playback_states()
    )

    with cameras_lock:
        camera_items = list(
            cameras.items()
        )

    for name, cam in camera_items:

        cams_data[name] = {

            "url":
                mask_video_source(
                    cam["url"]
                ),

            "source":
                mask_video_source(
                    cam["url"]
                ),

            "source_type":
                cam.get(
                    "source_type"
                ),

            "loop_video":
                cam.get(
                    "loop_video"
                ),

            "is_playing":
                playback_states.get(
                    name,
                    False
                )
                if cam.get("source_type") == "video"
                else None,

            "has_processor":
                cam.get(
                    "processor"
                )
                is not None,

            "src_pts":
                cam.get(
                    "src_pts"
                ),

            "dst_pts":
                cam.get(
                    "dst_pts"
                ),

            "floorplan_name": cam.get("floorplan_name"),

            "tracker":
                get_camera_tracker_status(
                    name,
                    cam
                ),

            "live_worker":
                live_camera_manager.get_status(
                    name
                )
                if cam.get("source_type") in {
                    "live",
                    "camera"
                }
                else None

            ,"video_last_processing_error": cam.get("video_last_processing_error")

        }


    return JSONResponse({

        "cameras":
            cams_data,

        "floorplan_exists":
            floorplan_exists,

        "floorplans": available_floorplan_names(),

        "reid":
            dict(
                REID_RUNTIME_STATUS
            ),

        "global_assignment":
            global_assignment_coordinator.status(),

        "identity":
            global_identity_manager.identity_state_diagnostics()

    })


# ============================================================
# VIDEO PLAYBACK CONTROL
# ============================================================

@app.post(
    "/api/video_playback"
)
async def video_playback(
    request: Request
):
    try:
        payload = await request.json()
    except Exception:
        return json_response(
            False,
            "Invalid JSON payload",
            status_code=400
        )

    if not isinstance(payload, dict):
        return json_response(
            False,
            "JSON payload must be an object",
            status_code=400
        )

    action = payload.get("action")

    if action not in {"play", "pause"}:
        return json_response(
            False,
            "Action must be 'play' or 'pause'",
            status_code=400
        )

    camera_names = payload.get(
        "camera_names"
    )

    if camera_names is None:
        camera_names = (
            multi_video_manager
            .get_camera_names()
        )
    elif (
        not isinstance(camera_names, list)
        or any(
            not isinstance(name, str)
            for name in camera_names
        )
    ):
        return json_response(
            False,
            "camera_names must be a list of video names",
            status_code=400
        )

    camera_names = list(
        dict.fromkeys(camera_names)
    )

    if not camera_names:
        return json_response(
            False,
            "No video clips selected",
            status_code=400
        )

    # A new playback session means a clip is being played again after it
    # reached the end. Pause -> Play in the middle is intentionally NOT reset.
    replaying_from_end = False

    if action == "play":
        with multi_video_manager.lock:
            for cam_name in camera_names:
                video_data = multi_video_manager.videos.get(cam_name)
                if not isinstance(video_data, dict):
                    continue

                total_frames = int(
                    video_data.get("total_frames", 0) or 0
                )
                frame_index = int(
                    video_data.get("frame_index", 0) or 0
                )

                if (
                    video_data.get("loop_video", False)
                    and total_frames > 0
                    and frame_index >= total_frames
                ):
                    replaying_from_end = True
                    break

    if replaying_from_end:
        reset_global_identity_session(
            reason="video_replay_from_start"
        )

        # Reset only tracker identity state for selected uploaded videos.
        # Capture/player/stream workers themselves are untouched.
        for cam_name in camera_names:
            try:
                reset_camera_tracker(
                    cam_name,
                    reason="video_replay_identity_reset",
                )
            except Exception as error:
                logger.warning(
                    "[IDENTITY] Tracker reset on replay failed | "
                    "camera=%s | error=%s",
                    cam_name,
                    error,
                )

    try:
        playback = (
            multi_video_manager
            .set_playback(
                camera_names,
                action == "play"
            )
        )
    except KeyError as e:
        return json_response(
            False,
            f"Video clip not found: {e.args[0]}",
            status_code=404
        )

    verb = "Playing" if action == "play" else "Paused"

    return json_response(
        True,
        f"{verb} {len(camera_names)} video clip(s)",
        {"playback": playback}
    )


# ============================================================
# FLOORPLAN UPLOAD
# ============================================================

@app.post(
    "/api/upload_floorplan"
)
async def upload_floorplan(
    file: UploadFile = File(...),
    floorplan_name: str = Form(...)
):

    try:

        contents = (
            await file.read()
        )


        decoded = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return json_response(False, "ไฟล์ที่อัปโหลดไม่ใช่รูปภาพที่อ่านได้", status_code=400)

        requested_stem = os.path.splitext(str(floorplan_name).strip())[0]
        if not requested_stem:
            return json_response(False, "กรุณาระบุชื่อ Floorplan", status_code=400)
        original_extension = os.path.splitext(file.filename or "")[1].lower()
        if original_extension not in {".png", ".jpg", ".jpeg", ".webp"}:
            original_extension = ".png"
        stored_path, stored_name = floorplan_path_by_name(
            f"{requested_stem[:80]}{original_extension}"
        )
        if os.path.exists(stored_path):
            return json_response(
                False,
                f"มี Floorplan ชื่อ {stored_name} อยู่แล้ว กรุณาใช้ชื่ออื่น",
                status_code=409,
            )
        with open(stored_path, "wb") as f:

            f.write(
                contents
            )


        # Keep the legacy active-map path for existing map rendering.
        shutil.copyfile(stored_path, FLOORPLAN_PATH)
        global_map.load_floorplan()
        with global_maps_lock:
            existing_manager = global_maps.get(stored_name)
        if existing_manager is not None:
            existing_manager.load_floorplan()


        return json_response(
            True,
            "อัปโหลดแผนผังสำเร็จ",
            {"floorplan_name": stored_name}
        )


    except Exception as e:

        logger.error(
            f"Floorplan upload failed: {e}",
            exc_info=True
        )

        return json_response(

            False,

            f"อัปโหลดไม่สำเร็จ: {e}",

            status_code=500

        )


# ============================================================
# VIDEO UPLOAD
# ============================================================

@app.post(
    "/api/upload_video"
)
async def upload_video(

    name: str = Form(...),

    file: UploadFile = File(...),

    loop_video: bool = Form(False),

    time_offset_sec: float = Form(0.0)

):

    try:

        # Uploaded videos are intentionally finite playback sources.
        loop_video = False

        if not file.filename:

            return json_response(

                False,

                "ไม่พบชื่อไฟล์",

                status_code=400

            )


        ext = (
            os.path.splitext(
                file.filename
            )[1].lower()
        )


        allowed_ext = [

            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm"

        ]


        if ext not in allowed_ext:

            return json_response(

                False,

                "รองรับเฉพาะไฟล์วิดีโอ mp4/avi/mov/mkv/webm",

                status_code=400

            )


        filename = (
            safe_filename(
                file.filename
            )
        )


        save_path = (
            os.path.join(
                UPLOAD_DIR,
                filename
            )
        )


        total_size = 0


        with open(
            save_path,
            "wb"
        ) as f:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += (
                    len(chunk)
                )

                if (
                    total_size
                    >
                    MAX_UPLOAD_SIZE
                ):

                    break

                f.write(
                    chunk
                )


        if (
            total_size
            >
            MAX_UPLOAD_SIZE
        ):

            os.remove(
                save_path
            )

            max_mb = (
                MAX_UPLOAD_SIZE
                //
                (1024 * 1024)
            )

            return json_response(

                False,

                f"ไฟล์ใหญ่เกินขีดจำกัด ({max_mb} MB)",

                status_code=400

            )


        with cameras_lock:

            cameras[name] = {

                "url":
                    save_path,

                "source_type":
                    "video",

                "loop_video":
                    loop_video,

                "processor":
                    None,

                "src_pts":
                    None,

                "dst_pts":
                    None,

                "last_frame":
                    None,

                "prev_assignments":
                    [],

                "time_offset_sec":
                    float(time_offset_sec),

                **new_camera_tracker_context()

            }
        try:

            initial_frame = multi_video_manager.register_video(
                name,
                save_path,
                loop_video,
                time_offset_sec
            )
            with cameras_lock:
                if name in cameras:
                    cameras[name]["last_frame"] = initial_frame.copy()

        except Exception as e:

            with cameras_lock:
                cameras.pop(name, None)

            try:
                os.remove(save_path)
            except Exception:
                pass

            return json_response(
                False,
                f"ไม่สามารถเปิด Video ได้: {e}",
                status_code=500
            )

        # เริ่ม Worker
        start_multi_camera_worker()

        logger.info(

            f"Video uploaded: "
            f"{name} -> {save_path} "
            f"({total_size} bytes)"

        )


        return json_response(

            True,

            "อัปโหลดวิดีโอสำเร็จ"

        )


    except Exception as e:

        logger.error(

            f"Video upload failed: {e}",

            exc_info=True

        )

        return json_response(

            False,

            f"อัปโหลดวิดีโอไม่สำเร็จ: {e}",

            status_code=500

        )


# ============================================================
# FLOORPLAN GET
# ============================================================

@app.get("/api/floorplans")
async def list_floorplans():
    items = [{"name": name} for name in available_floorplan_names()]
    return {"floorplans": items}


@app.delete("/api/floorplans/{floorplan_name}")
async def delete_floorplan(floorplan_name: str):
    path, safe_name = floorplan_path_by_name(floorplan_name)
    if not os.path.isfile(path):
        return json_response(False, "Floorplan not found", status_code=404)

    with cameras_lock:
        cameras_using_map = sorted(
            camera_name
            for camera_name, camera in cameras.items()
            if camera.get("floorplan_name") == safe_name
        )
    if cameras_using_map:
        return json_response(
            False,
            "ไม่สามารถลบ Floorplan ที่ยังมี Calibration ใช้งานอยู่",
            {"cameras": cameras_using_map},
            status_code=409,
        )

    os.remove(path)
    with global_maps_lock:
        global_maps.pop(safe_name, None)
    return json_response(
        True,
        "ลบ Floorplan สำเร็จ",
        {"floorplan_name": safe_name},
    )


@app.get("/api/floorplans/{floorplan_name}/calibrations")
async def floorplan_calibrations(floorplan_name: str, exclude_camera: str = None):
    _, safe_name = floorplan_path_by_name(floorplan_name)
    regions = []
    with cameras_lock:
        camera_items = list(cameras.items())
    for camera_name, camera in camera_items:
        if exclude_camera and camera_name == exclude_camera:
            continue
        if camera.get("floorplan_name") != safe_name:
            continue
        points = camera.get("dst_pts")
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            continue
        try:
            normalized_points = [
                [float(point[0]), float(point[1])]
                for point in points
            ]
        except (TypeError, ValueError, IndexError):
            continue
        regions.append({"camera_name": camera_name, "points": normalized_points})
    return {"floorplan_name": safe_name, "calibrations": regions}

@app.get(
    "/api/get_floorplan"
)
async def get_floorplan(name: str = None):

    selected_path = FLOORPLAN_PATH
    selected_name = None
    if name:
        selected_path, selected_name = floorplan_path_by_name(name)
        if not os.path.isfile(selected_path):
            return JSONResponse({"error": "Floorplan not found"}, status_code=404)

    img_b64 = (
        image_file_to_base64(
            selected_path
        )
    )


    if img_b64 is None:

        return JSONResponse(

            {
                "error":
                    "No floorplan uploaded"
            },

            status_code=404

        )


    return {

        "image_base64": img_b64,
        "floorplan_name": selected_name

    }


# ============================================================
# ADD CAMERA
# ============================================================

@app.post(
    "/api/add_camera"
)
async def add_camera(

    name: str = Form(...),

    url: str = Form(...)

):

    try:

        final_url = (
            parse_video_source(
                url
            )
        )

        cam_data = {

            "url":
                final_url,

            "source_type":
                "live",

            "loop_video":
                False,

            "processor":
                None,

            "src_pts":
                None,

            "dst_pts":
                None,

            "last_frame":
                None,

            "last_frame_event_time":
                None,

            "prev_assignments":
                [],

            **new_camera_tracker_context()

        }

        with cameras_lock:

            if name in cameras:
                return json_response(
                    False,
                    "Camera name already exists",
                    status_code=409
                )

            cameras[name] = cam_data

        try:
            worker, created = (
                live_camera_manager.start_worker(
                    name,
                    final_url
                )
            )

            if not created:
                raise RuntimeError(
                    "Live camera worker already exists"
                )
        except Exception:
            live_camera_manager.stop_worker(name)

            with cameras_lock:
                if cameras.get(name) is cam_data:
                    cameras.pop(name, None)

            raise


        logger.info(

            f"Camera added: "
            f"{name} -> {mask_video_source(final_url)}"

        )


        return json_response(

            True,

            "เพิ่มกล้องสำเร็จ"

        )


    except Exception as e:

        logger.error(

            f"Add camera failed: {e}",

            exc_info=True

        )

        return json_response(

            False,

            f"เพิ่มกล้องไม่สำเร็จ: {e}",

            status_code=400

        )


# ============================================================
# DELETE CAMERA
# ============================================================

@app.delete(
    "/api/delete_camera/{cam_name}"
)
async def delete_camera(
    cam_name: str
):

    with cameras_lock:
        cam = cameras.get(cam_name)

    if cam is not None:

        if cam.get("source_type") == "video":

            # Remove capture state without holding cameras_lock; the video
            # manager has its own lock.
            multi_video_manager.remove_video(cam_name)

            video_path = cam.get("url")

            if (
                isinstance(video_path, str)
                and os.path.exists(video_path)
            ):
                try:
                    os.remove(video_path)

                except OSError as e:

                    logger.warning(
                        f"Failed to delete video file: "
                        f"{video_path}, error: {e}"
                    )

        elif cam.get("source_type") in {
            "live",
            "camera"
        }:
            live_camera_manager.stop_worker(
                cam_name
            )

        reset_camera_tracker(
            cam_name,
            reason="camera_removed"
        )

        with cameras_lock:
            cameras.pop(
                cam_name,
                None
            )

        with video_worker_lock:
            processed_frames.pop(
                cam_name,
                None
            )
            processed_frame_locks.pop(
                cam_name,
                None
            )

        logger.info(

            f"Camera deleted: "
            f"{cam_name}"

        )


        return json_response(

            True,

            "Camera deleted",

            {
                "camera":
                    cam_name
            }

        )


    return json_response(

        False,

        "Camera not found",

        status_code=404

    )


# ============================================================
# RESET ONE CAMERA TRACKER
# ============================================================

@app.post(
    "/api/reset_tracker/{cam_name}"
)
async def reset_tracker(
    cam_name: str
):
    result = reset_camera_tracker(
        cam_name,
        reason="api_reset"
    )

    if result is None:
        return json_response(
            False,
            "Camera not found",
            status_code=404
        )

    return json_response(
        True,
        "Camera tracker reset",
        {
            "tracker_reset": result
        }
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.get(
    "/api/video_feed/{cam_name}"
)
async def video_feed(
    cam_name: str
):

    if cam_name not in cameras:

        return json_response(

            False,

            "Camera not found",

            status_code=404

        )


    return StreamingResponse(

        generate_frames(
            cam_name
        ),

        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )

    )


# ============================================================
# GLOBAL MAP FEED
# ============================================================

@app.get(
    "/api/global_map_feed"
)
async def global_map_feed(name: str = None):

    if name:
        path, safe_name = floorplan_path_by_name(name)
        if not os.path.isfile(path):
            return JSONResponse({"error": "Floorplan not found"}, status_code=404)
        name = safe_name

    return StreamingResponse(

        generate_global_map(name),

        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )

    )


# ============================================================
# CAPTURE FRAME
# ============================================================

@app.get(
    "/api/capture_frame/{cam_name}"
)
async def capture_frame(
    cam_name: str
):

    with cameras_lock:
        cam = cameras.get(
            cam_name
        )

        if cam is not None:
            source_type = cam.get("source_type")
            source = cam.get("url")
            cached_frame = cam.get("last_frame")
            frame = (
                cached_frame.copy()
                if cached_frame is not None
                else None
            )
        else:
            source_type = None
            source = None
            frame = None


    if not cam:

        return json_response(

            False,

            "Camera not found",

            status_code=404

        )


    # A playing uploaded clip already has a worker-owned capture. Reopening it
    # here can fail or race the worker and used to discard a perfectly valid
    # cached frame, leaving the calibration dialog blank.
    if frame is None and source_type == "video":
        frame = open_camera_once(source)


    if frame is None:

        return json_response(

            False,

            "Cannot capture frame",

            status_code=500

        )


    with cameras_lock:
        if cameras.get(cam_name) is cam:
            cam["last_frame"] = frame.copy()


    img_b64 = (
        frame_to_base64(
            frame
        )
    )


    return json_response(

        True,

        "Captured",

        {

            "camera":
                cam_name,

            "image_base64":
                img_b64

        }

    )


# ============================================================
# SAVE CALIBRATION
# ============================================================

@app.post(
    "/api/save_calibration/{cam_name}"
)
async def save_calibration(

    cam_name: str,

    src_pts: str = Form(...),

    dst_pts: str = Form(...),

    floorplan_name: str = Form(...)

):

    cam = cameras.get(
        cam_name
    )


    if not cam:

        return json_response(

            False,

            "Camera not found",

            status_code=404

        )


    try:

        selected_floorplan_path, safe_floorplan_name = floorplan_path_by_name(floorplan_name)
        if not os.path.isfile(selected_floorplan_path):
            return json_response(False, "Floorplan not found", status_code=404)

        parsed_src = (
            parse_json_points(
                src_pts
            )
        )

        parsed_dst = (
            parse_json_points(
                dst_pts
            )
        )


        processor = (
            CameraProcessor(
                cam_name,
                parsed_src,
                parsed_dst
            )
        )


        with cameras_lock:

            cam[
                "src_pts"
            ] = parsed_src

            cam[
                "dst_pts"
            ] = parsed_dst

            cam[
                "processor"
            ] = processor

            cam["floorplan_name"] = safe_floorplan_name

        # The existing application displays one active global map. Selecting a
        # floorplan during calibration makes that map the active display map.
        shutil.copyfile(selected_floorplan_path, FLOORPLAN_PATH)
        global_map.load_floorplan()
        get_floorplan_map_manager(safe_floorplan_name).load_floorplan()


        return json_response(

            True,

            "Saved calibration",

            {

                "camera":
                    cam_name,

                "src_pts":
                    parsed_src,

                "dst_pts":
                    parsed_dst,

                "floorplan_name": safe_floorplan_name

            }

        )


    except Exception as e:

        return json_response(

            False,

            str(e),

            status_code=400

        )


# ============================================================
# CAMERA CONFIG
# ============================================================

@app.get("/api/topology")
async def get_topology():
    with topology_lock:
        config = copy.deepcopy(topology_config)
    validation_error = config.pop("_validation_error", None)
    return {
        "topology": config,
        "valid": validation_error is None,
        "error": validation_error,
    }


@app.put("/api/topology")
async def put_topology(request: Request):
    global topology_config
    try:
        payload = await request.json()
    except Exception:
        return json_response(
            False,
            "Invalid topology JSON payload",
            status_code=400,
        )
    with cameras_lock:
        known_cameras = set(cameras)
    try:
        normalized = normalize_topology_config(
            payload,
            known_cameras=known_cameras,
        )
    except ValueError as error:
        logger.warning("Topology update rejected: %s", error)
        return json_response(
            False,
            str(error),
            status_code=400,
        )
    with topology_lock:
        with open(TOPOLOGY_CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=False, indent=2)
        topology_config = normalized
    return {"topology": normalized}

@app.get(
    "/api/camera_config/{cam_name}"
)
async def camera_config(
    cam_name: str
):

    cam = cameras.get(
        cam_name
    )


    if not cam:

        return json_response(

            False,

            "Camera not found",

            status_code=404

        )


    return json_response(

        True,

        "Success",

        {

            "name":
                cam_name,

            "url":
                cam["url"],

            "source_type":
                cam.get(
                    "source_type",
                    "camera"
                ),

            "loop_video":
                cam.get(
                    "loop_video",
                    False
                ),

            "src_pts":
                cam["src_pts"],

            "dst_pts":
                cam["dst_pts"],

            "has_homography":
                bool(
                    cam[
                        "processor"
                    ]
                    is not None
                )

        }

    )


# ============================================================
# SHUTDOWN
# ============================================================

def stop_background_workers():
    app.is_running = False
    global_assignment_coordinator.stop()
    live_camera_manager.stop_all()
    stop_multi_camera_worker()


@app.on_event("shutdown")
def cleanup_background_workers():
    stop_background_workers()
    global_identity_manager.close()

@app.post("/api/shutdown")
async def shutdown_system():

    stop_background_workers()
    global_identity_manager.close()

    logger.info(
        "Shutdown requested"
    )

    def graceful_exit():

        time.sleep(1)

        os.kill(
            os.getpid(),
            signal.SIGTERM
        )

    threading.Thread(
        target=graceful_exit,
        daemon=True
    ).start()

    return json_response(
        True,
        "shutting down"
    )


# ============================================================
# MAIN
# ============================================================

# Refresh the extracted manager namespace after all downstream helpers (such
# as bbox_center and runtime topology functions) have been defined.
_identity_manager_module.configure_identity_dependencies(globals())

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8899

    )
