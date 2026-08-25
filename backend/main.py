import cv2
import os
import threading
import time
import json
import base64
import warnings
import uuid
import numpy as np

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
UPLOAD_DIR = "static/uploads"
TOPOLOGY_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "camera_topology.json"
)

os.makedirs("static", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
        "..",
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
    path = os.path.expanduser(
        os.path.expandvars(
            configured_path
        )
    )

    if not os.path.isabs(path):
        path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            path
        )

    return os.path.abspath(path)


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
REID_CROSS_CAM_THRESHOLD = 0.62
REID_CROSS_CAM_STRONG_THRESHOLD = 0.76

# Same-camera
REID_SAME_CAM_THRESHOLD = 0.50
REID_SAME_CAM_STRONG_THRESHOLD = 0.70


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
REID_MIN_BLUR_VARIANCE = 20.0


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

IDENTITY_PROVISIONAL = "PROVISIONAL"
IDENTITY_ACTIVE = "ACTIVE"
IDENTITY_DORMANT = "DORMANT"
IDENTITY_EXPIRED = "EXPIRED"
IDENTITY_DORMANT_TTL_SEC = 300.0
IDENTITY_TRANSITION_HISTORY_SIZE = 100
IDENTITY_EXPIRED_SNAPSHOT_RETENTION_SEC = 7 * 24 * 60 * 60


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
    vec = np.asarray(
        vec,
        dtype=np.float32
    ).flatten()

    norm = np.linalg.norm(vec)

    if norm < 1e-8:
        return vec

    return vec / norm


def cosine_similarity(a, b):
    if a is None or b is None:
        return -1.0

    a = l2_normalize(a)
    b = l2_normalize(b)

    return float(np.dot(a, b))


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


def safe_json_value(value):

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def default_topology_config():
    return {"version": 1, "enforce": False, "transitions": []}


def validate_topology_config(config):
    if not isinstance(config, dict):
        raise ValueError("Topology config must be an object")
    if not isinstance(config.get("transitions", []), list):
        raise ValueError("Topology transitions must be a list")
    for rule in config.get("transitions", []):
        required = ("from_camera", "to_camera")
        if not isinstance(rule, dict) or not all(rule.get(key) for key in required):
            raise ValueError("Every transition needs source and destination cameras")
        min_time = float(rule.get("min_travel_sec", 0.0))
        max_time = float(rule.get("max_travel_sec", float("inf")))
        if min_time < 0 or max_time < min_time:
            raise ValueError("Invalid transition travel-time window")
    return True


def load_topology_config():
    if not os.path.isfile(TOPOLOGY_CONFIG_PATH):
        return default_topology_config()
    try:
        with open(TOPOLOGY_CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        validate_topology_config(config)
        return config
    except Exception as error:
        logger.error("Topology config ignored: %s", error)
        return default_topology_config()


topology_lock = threading.Lock()
topology_config = load_topology_config()


# ============================================================
# LIGHTWEIGHT FALLBACK
# ============================================================

class LightweightAppearanceFeatureExtractor:

    def __init__(self):
        self.name = "lightweight"
        self.device = "cpu"
        self.embedding_dimension = (
            (2 * 12 * 4 * 4)
            +
            (16 * 32)
            +
            1
        )

    def _hsv_hist(
        self,
        img_bgr,
        h_bins=12,
        s_bins=4,
        v_bins=4
    ):

        hsv = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2HSV
        )

        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [h_bins, s_bins, v_bins],
            [0, 180, 0, 256, 0, 256]
        )

        hist = cv2.normalize(
            hist,
            hist
        ).flatten().astype(np.float32)

        return hist

    def _region_hist(self, img_bgr):

        h, w = img_bgr.shape[:2]

        upper = img_bgr[
            :max(1, int(h * 0.45)),
            :
        ]

        lower = img_bgr[
            max(0, int(h * 0.45)):,
            :
        ]

        upper_hist = self._hsv_hist(upper)

        lower_hist = self._hsv_hist(lower)

        return np.concatenate(
            [
                upper_hist,
                lower_hist
            ]
        ).astype(np.float32)

    def _shape_feature(self, img_bgr):

        h, w = img_bgr.shape[:2]

        aspect = np.array(
            [w / max(h, 1)],
            dtype=np.float32
        )

        gray = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (16, 32)
        )

        gray = gray.astype(
            np.float32
        ) / 255.0

        coarse = gray.flatten()

        return np.concatenate(
            [
                aspect,
                coarse
            ]
        ).astype(np.float32)

    def extract(self, person_crop):

        if person_crop is None:
            return None

        if person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]

        if (
            h < REID_MIN_CROP_SIZE
            or
            w < REID_MIN_CROP_SIZE
        ):
            return None

        crop = cv2.resize(
            person_crop,
            (64, 128)
        )

        hist_feat = self._region_hist(crop)

        shape_feat = self._shape_feature(crop)

        feat = np.concatenate(
            [
                hist_feat,
                shape_feat
            ]
        ).astype(np.float32)

        return l2_normalize(feat)


# ============================================================
# OSNET MARKET1501
# ============================================================

def resolve_reid_device():
    if REID_DEVICE_CONFIG == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        REID_DEVICE_CONFIG.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "REID_DEVICE requests CUDA but CUDA is unavailable"
        )

    if (
        REID_DEVICE_CONFIG != "cpu"
        and not REID_DEVICE_CONFIG.startswith("cuda")
    ):
        raise ValueError(
            "REID_DEVICE must be auto, cpu, cuda, or cuda:<index>"
        )

    return REID_DEVICE_CONFIG


def load_validated_osnet_checkpoint(
    model,
    checkpoint_path
):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "OSNet checkpoint must contain a state dictionary"
        )

    checkpoint_metadata = (
        read_osnet_checkpoint_metadata(
            checkpoint
        )
    )

    try:
        validate_osnet_checkpoint_metadata(
            checkpoint_metadata,
            expected_architecture=(
                REID_MODEL_NAME
            ),
            expected_embedding_dimension=(
                OSNET_EMBEDDING_DIMENSION
            ),
            expected_preprocessing=(
                osnet_preprocessing_metadata()
            )
        )
    except ValueError as error:
        raise RuntimeError(
            str(error)
        ) from error

    state_dict = checkpoint

    for key in (
        "model_state_dict",
        "state_dict"
    ):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            state_dict = candidate
            break

    normalized_state = {}

    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue

        normalized_key = (
            key[7:]
            if key.startswith("module.")
            else key
        )

        if normalized_key.startswith(
            "classifier."
        ):
            continue

        if torch.is_tensor(value):
            normalized_state[
                normalized_key
            ] = value

    if not normalized_state:
        raise RuntimeError(
            "OSNet checkpoint contains no model tensors"
        )

    model_state = model.state_dict()

    expected_keys = {
        key
        for key in model_state
        if not key.startswith("classifier.")
    }

    checkpoint_keys = set(
        normalized_state
    )

    missing_keys = sorted(
        expected_keys
        -
        checkpoint_keys
    )

    unexpected_keys = sorted(
        checkpoint_keys
        -
        expected_keys
    )

    shape_mismatches = sorted(
        key
        for key in (
            expected_keys
            &
            checkpoint_keys
        )
        if tuple(normalized_state[key].shape)
        != tuple(model_state[key].shape)
    )

    if (
        missing_keys
        or unexpected_keys
        or shape_mismatches
    ):
        raise RuntimeError(
            "OSNet checkpoint is incompatible with "
            f"{REID_MODEL_NAME}; "
            f"missing={missing_keys[:5]}, "
            f"unexpected={unexpected_keys[:5]}, "
            f"shape_mismatch={shape_mismatches[:5]}"
        )

    model.load_state_dict(
        normalized_state,
        strict=False
    )

    return (
        len(normalized_state),
        checkpoint_metadata
    )

class OSNetFeatureExtractor:

    def __init__(self):

        if torch is None:
            raise RuntimeError(
                "PyTorch import failed: "
                f"{TORCH_IMPORT_ERROR}"
            )

        if torchreid is None:
            raise RuntimeError(
                "torchreid import failed: "
                f"{TORCHREID_IMPORT_ERROR}"
            )

        if FeatureExtractor is None:
            raise RuntimeError(
                "FeatureExtractor import failed: "
                f"{FEATURE_EXTRACTOR_IMPORT_ERROR}"
            )

        if not os.path.isfile(REID_MODEL_PATH):
            path_type = (
                "directory"
                if os.path.isdir(REID_MODEL_PATH)
                else "missing"
            )

            raise FileNotFoundError(
                "OSNet checkpoint is not a file "
                f"({path_type}): {REID_MODEL_PATH}"
            )

        self.name = (
            f"{REID_MODEL_NAME}_checkpoint"
        )

        self.device = resolve_reid_device()

        # ----------------------------------------------------
        # ใช้ FeatureExtractor โดยตรง
        # และระบุ Market1501 weight
        # ----------------------------------------------------

        with warnings.catch_warnings():
            # FeatureExtractor does not understand the project's
            # model_state_dict wrapper. The strict loader immediately below
            # is authoritative and reports any real incompatibility.
            warnings.filterwarnings(
                "ignore",
                message=(
                    "The pretrained weights .* cannot be loaded.*"
                )
            )

            self.extractor = FeatureExtractor(
                model_name=REID_MODEL_NAME,
                model_path=REID_MODEL_PATH,
                image_size=(
                REID_INPUT_H,
                REID_INPUT_W
            ),
            pixel_mean=list(
                OSNET_PIXEL_MEAN
            ),
            pixel_std=list(
                OSNET_PIXEL_STD
            ),
            device=self.device,
            verbose=False
            )

        (
            self.loaded_tensor_count,
            self.checkpoint_metadata
        ) = load_validated_osnet_checkpoint(
            self.extractor.model,
            REID_MODEL_PATH
        )

        smoke_crop = np.full(
            (
                REID_INPUT_H,
                REID_INPUT_W,
                3
            ),
            127,
            dtype=np.uint8
        )

        smoke_embedding = self.extract(
            smoke_crop
        )

        if (
            smoke_embedding is None
            or smoke_embedding.size == 0
            or not np.all(
                np.isfinite(smoke_embedding)
            )
            or np.linalg.norm(smoke_embedding) < 1e-8
        ):
            raise RuntimeError(
                "OSNet checkpoint loaded but embedding smoke test failed"
            )

        self.embedding_dimension = int(
            smoke_embedding.size
        )

        if (
            self.embedding_dimension
            != OSNET_EMBEDDING_DIMENSION
        ):
            raise RuntimeError(
                "OSNet embedding dimension differs from shared contract: "
                f"expected={OSNET_EMBEDDING_DIMENSION}, "
                f"inference={self.embedding_dimension}"
            )

        model_dimension = getattr(
            self.extractor.model,
            "feature_dim",
            None
        )

        if (
            model_dimension is not None
            and self.embedding_dimension
            != int(model_dimension)
        ):
            raise RuntimeError(
                "OSNet embedding dimension mismatch: "
                f"model={model_dimension}, "
                f"inference={self.embedding_dimension}"
            )

    def extract(self, person_crop):

        if person_crop is None:
            return None

        if person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]

        if (
            h < REID_MIN_CROP_SIZE
            or
            w < REID_MIN_CROP_SIZE
        ):
            return None

        try:

            # Training/evaluation use PIL RGB. OpenCV crops are BGR, so
            # convert explicitly before applying the shared normalization.
            rgb_crop = cv2.cvtColor(
                person_crop,
                cv2.COLOR_BGR2RGB
            )

            features = self.extractor(
                [rgb_crop]
            )

            if features is None:
                return None

            if hasattr(
                features,
                "detach"
            ):
                feat = (
                    features[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

            else:
                feat = np.asarray(
                    features[0]
                )

            feat = feat.reshape(-1).astype(
                np.float32
            )

            return l2_normalize(feat)

        except Exception as e:

            logger.warning(
                f"OSNet feature extraction failed: {e}"
            )

            return None


# ============================================================
# BUILD REID EXTRACTOR
# ============================================================

def build_feature_extractor():
    global REID_RUNTIME_STATUS

    initialization_error = None

    if USE_OSNET:

        try:

            extractor = OSNetFeatureExtractor()

            REID_RUNTIME_STATUS = {
                "enabled": True,
                "model_architecture": REID_MODEL_NAME,
                "checkpoint_path": REID_MODEL_PATH,
                "checkpoint_name": os.path.basename(
                    REID_MODEL_PATH
                ),
                "checkpoint_loaded": True,
                "device": extractor.device,
                "fallback_active": False,
                "embedding_dimension": (
                    extractor.embedding_dimension
                ),
                "expected_embedding_dimension": (
                    OSNET_EMBEDDING_DIMENSION
                ),
                "preprocessing": (
                    osnet_preprocessing_metadata()
                ),
                "checkpoint_metadata": (
                    extractor.checkpoint_metadata
                ),
                "threshold_safety_mode": (
                    REID_THRESHOLD_SAFETY_MODE
                ),
                "similarity_only_shortcut_enabled": (
                    REID_THRESHOLD_SAFETY_MODE
                    == "validated"
                ),
                "active_extractor": extractor.name,
                "error": None
            }

            logger.info(
                "[ReID] Runtime | enabled=%s | "
                "architecture=%s | checkpoint=%s | "
                "loaded=%s | device=%s | fallback=%s | "
                "embedding_dim=%s | threshold_safety=%s",
                REID_RUNTIME_STATUS["enabled"],
                REID_RUNTIME_STATUS["model_architecture"],
                REID_RUNTIME_STATUS["checkpoint_path"],
                REID_RUNTIME_STATUS["checkpoint_loaded"],
                REID_RUNTIME_STATUS["device"],
                REID_RUNTIME_STATUS["fallback_active"],
                REID_RUNTIME_STATUS["embedding_dimension"],
                REID_RUNTIME_STATUS["threshold_safety_mode"]
            )

            return extractor

        except Exception as e:
            initialization_error = (
                f"{type(e).__name__}: {e}"
            )

            logger.error(
                "[ReID] OSNet initialization failed: %s",
                initialization_error
            )

    extractor = (
        LightweightAppearanceFeatureExtractor()
    )

    REID_RUNTIME_STATUS = {
        "enabled": bool(USE_OSNET),
        "model_architecture": REID_MODEL_NAME,
        "checkpoint_path": REID_MODEL_PATH,
        "checkpoint_name": os.path.basename(
            REID_MODEL_PATH
        ),
        "checkpoint_loaded": False,
        "device": extractor.device,
        "fallback_active": True,
        "embedding_dimension": (
            extractor.embedding_dimension
        ),
        "expected_embedding_dimension": (
            OSNET_EMBEDDING_DIMENSION
        ),
        "preprocessing": (
            osnet_preprocessing_metadata()
        ),
        "checkpoint_metadata": None,
        "threshold_safety_mode": (
            REID_THRESHOLD_SAFETY_MODE
        ),
        "similarity_only_shortcut_enabled": (
            REID_THRESHOLD_SAFETY_MODE
            == "validated"
        ),
        "active_extractor": extractor.name,
        "error": initialization_error
    }

    logger.info(
        "[ReID] Runtime | enabled=%s | "
        "architecture=%s | checkpoint=%s | "
        "loaded=%s | device=%s | fallback=%s | "
        "embedding_dim=%s | threshold_safety=%s | error=%s",
        REID_RUNTIME_STATUS["enabled"],
        REID_RUNTIME_STATUS["model_architecture"],
        REID_RUNTIME_STATUS["checkpoint_path"],
        REID_RUNTIME_STATUS["checkpoint_loaded"],
        REID_RUNTIME_STATUS["device"],
        REID_RUNTIME_STATUS["fallback_active"],
        REID_RUNTIME_STATUS["embedding_dimension"],
        REID_RUNTIME_STATUS["threshold_safety_mode"],
        REID_RUNTIME_STATUS["error"]
    )

    return extractor


# ============================================================
# CAMERA PROCESSOR
# ============================================================

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

class GlobalIdentityManager:

    def __init__(self, identity_store=None):

        # Explicit stores make tests and short-lived model instances isolated.
        # The application singleton below opts into the durable SQLite store.
        self.identity_store = identity_store

        # GID -> identity information
        self.identities = self.identity_store.load_identities() if self.identity_store else {}

        self.next_global_id = max(self.identities, default=0) + 1

        # (camera, local_track_id) -> GID
        self.local_to_global = {}

        # (camera, local_track_id) -> quality-approved embedding samples.
        # This is intentionally camera-local evidence, never camera runtime state.
        self.tracklets = {}

        # Occlusion
        self.occlusion_hold = {}

        # Recent same camera
        self.recent_same_cam = []

        # Recent cross camera
        self.recent_cross_cam = []

        # Re-entrant so the downstream coordinator can atomically validate a
        # camera generation and invoke assign_global_batch under one lock.
        self.lock = threading.RLock()

        # Diagnostics for the most recent synchronized multi-camera decision.
        self.last_global_batch_diagnostics = None

    # ==================

    def reset_camera_local_state(
        self,
        cam_name
    ):
        """Remove camera-local evidence without deleting global identities."""
        with self.lock:
            local_keys = [
                key
                for key in self.local_to_global
                if key[0] == cam_name
            ]
            hold_keys = [
                key
                for key in self.occlusion_hold
                if key[0] == cam_name
            ]

            for key in local_keys:
                self.local_to_global.pop(
                    key,
                    None
                )

            for key in hold_keys:
                self.occlusion_hold.pop(
                    key,
                    None
                )

            for key in local_keys:
                self.tracklets.pop(key, None)

            return {
                "local_mappings_removed": len(
                    local_keys
                ),
                "occlusion_holds_removed": len(
                    hold_keys
                ),
                "tracklets_removed": len(local_keys)
            }

    # ==================

    def _verify_local_track(
        self,
        gid,
        det
    ):
        identity = self.identities.get(gid)

        if identity is None:
            return False, -1.0

        appearance = self._gallery_similarity(
            det["emb"],
            identity
        )

        if appearance >= LOCAL_TRACK_STRONG_THRESHOLD:
            return True, appearance

        if appearance >= LOCAL_TRACK_VERIFY_THRESHOLD:
            return True, appearance

        return False, appearance

    # ========================================================
    # GALLERY SIMILARITY
    # ========================================================

    def _gallery_similarity(
        self,
        emb,
        identity
    ):

        scores = []

        if identity.get(
            "embedding"
        ) is not None:

            scores.append(
                cosine_similarity(
                    emb,
                    identity["embedding"]
                )
            )

        for g in identity.get(
            "gallery",
            []
        ):

            scores.append(
                cosine_similarity(
                    emb,
                    g
                )
            )

        if not scores:
            return -1.0

        scores.sort(
            reverse=True
        )

        # ใช้ top 3
        topk = scores[
            :min(3, len(scores))
        ]

        return float(
            sum(topk) / len(topk)
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self, reference_time=None):

        # Re-ID state is timestamped on the observation timeline.  Wall-clock
        # processing delay must not age a person by more than its source gap.
        now = (
            time.time()
            if reference_time is None
            else float(reference_time)
        )

        # ----------------------------------------------------
        # Global identities
        # ----------------------------------------------------

        for gid, info in self.identities.items():
            state = info.get("state", IDENTITY_ACTIVE)
            last_seen = info.get("last_seen")
            if not isinstance(last_seen, (int, float)):
                self._transition_identity(info, IDENTITY_EXPIRED, now, "invalid_persisted_identity")
                continue
            idle_sec = now - last_seen
            if state in (IDENTITY_PROVISIONAL, IDENTITY_ACTIVE) and idle_sec > REID_MAX_IDLE_SEC:
                self._transition_identity(info, IDENTITY_DORMANT, now, "idle_timeout")
            elif state == IDENTITY_DORMANT and idle_sec > REID_MAX_IDLE_SEC + IDENTITY_DORMANT_TTL_SEC:
                self._transition_identity(info, IDENTITY_EXPIRED, now, "dormant_ttl_expired")

        if self.identity_store:
            self.identity_store.purge_expired_snapshots(
                now - IDENTITY_EXPIRED_SNAPSHOT_RETENTION_SEC
            )


        # ----------------------------------------------------
        # Local -> global
        # ----------------------------------------------------

        stale_local_keys = []

        for key, data in self.local_to_global.items():

            if (
                now - data["last_seen"]
                >
                REID_MAX_IDLE_SEC
            ):

                stale_local_keys.append(
                    key
                )

        for key in stale_local_keys:

            self.local_to_global.pop(
                key,
                None
            )


        # ----------------------------------------------------
        # Occlusion
        # ----------------------------------------------------

        stale_hold_keys = []

        for key, hold in self.occlusion_hold.items():

            if (
                now >
                hold.get(
                    "until_ts",
                    0
                )
            ):

                stale_hold_keys.append(
                    key
                )

        for key in stale_hold_keys:

            self.occlusion_hold.pop(
                key,
                None
            )


        # ----------------------------------------------------
        # Same camera cache
        # ----------------------------------------------------

        self.recent_same_cam = [

            item

            for item in self.recent_same_cam

            if (
                now - item.get(
                    "ts",
                    0
                )
                <=
                REID_RECENT_SAME_CAM_SEC
            )

            and
            item.get("gid")
            in
            self.identities

        ][-300:]


        # ----------------------------------------------------
        # Cross camera cache
        # ----------------------------------------------------

        self.recent_cross_cam = [

            item

            for item in self.recent_cross_cam

            if (
                now - item.get(
                    "ts",
                    0
                )
                <=
                REID_RECENT_CROSS_CAM_SEC
            )

            and
            item.get("gid")
            in
            self.identities

        ][-300:]


    # ========================================================
    # SIZE CHECK
    # ========================================================

    def _size_ratio_ok(
        self,
        box_wh,
        ref_wh
    ):

        if (
            box_wh is None
            or
            ref_wh is None
        ):
            return True

        bw, bh = box_wh
        rw, rh = ref_wh

        if min(
            bw,
            bh,
            rw,
            rh
        ) <= 0:

            return True

        wr = (
            min(bw, rw)
            /
            max(bw, rw)
        )

        hr = (
            min(bh, rh)
            /
            max(bh, rh)
        )

        return (
            wr >= REID_SIZE_GATE_RATIO
            and
            hr >= REID_SIZE_GATE_RATIO
        )


    # ========================================================
    # MAP DISTANCE
    # ========================================================

    def _map_distance(
        self,
        p1,
        p2
    ):

        if (
            p1 is None
            or
            p2 is None
        ):
            return None

        return float(
            np.linalg.norm(
                np.array(
                    p1,
                    dtype=np.float32
                )
                -
                np.array(
                    p2,
                    dtype=np.float32
                )
            )
        )


    # ========================================================
    # MAP SCORE
    # ========================================================

    def _map_score(
        self,
        identity,
        map_pos,
        cross_camera=False
    ):

        prev_pos = identity.get(
            "last_map_pos"
        )

        if (
            map_pos is None
            or
            prev_pos is None
        ):

            return 0.50

        dist = self._map_distance(
            map_pos,
            prev_pos
        )

        if dist is None:
            return 0.50

        if cross_camera:

            gate = (
                REID_MAP_GATE_CROSS_CAM_PX
            )

        else:

            gate = (
                REID_MAP_GATE_SAME_CAM_PX
            )

        score = (
            1.0
            -
            min(
                dist / max(gate, 1.0),
                1.0
            )
        )

        return float(
            max(
                0.0,
                min(1.0, score)
            )
        )


    # ========================================================
    # TIME SCORE
    # ========================================================

    def _time_score(
        self,
        identity,
        now_ts
    ):

        dt = max(
            0.0,
            now_ts
            -
            identity["last_seen"]
        )

        return float(
            max(
                0.0,
                1.0
                -
                min(
                    dt
                    /
                    max(
                        REID_MAX_IDLE_SEC,
                        1e-3
                    ),
                    1.0
                )
            )
        )


    # ========================================================
    # MOTION
    # ========================================================

    def _recent_history_for_gid(
        self,
        cam_name,
        gid,
        prev_assignments,
        limit=3
    ):

        if not prev_assignments:
            return []

        hist = []

        for item in reversed(
            prev_assignments
        ):

            if (
                item.get("cam_name")
                ==
                cam_name
                and
                item.get("gid")
                ==
                gid
            ):

                hist.append(item)

                if len(hist) >= limit:
                    break

        return list(
            reversed(hist)
        )


    def _predict_center(
        self,
        history,
        event_time=None
    ):

        if not history:
            return None

        if len(history) == 1:
            return history[-1].get(
                "center"
            )

        p1 = history[-2].get(
            "center"
        )

        p2 = history[-1].get(
            "center"
        )

        t1 = history[-2].get(
            "ts",
            0.0
        )

        t2 = history[-1].get(
            "ts",
            0.0
        )

        if (
            p1 is None
            or
            p2 is None
        ):

            return p2 or p1

        dt = max(
            t2 - t1,
            1e-3
        )

        vx = (
            p2[0] - p1[0]
        ) / dt

        vy = (
            p2[1] - p1[1]
        ) / dt

        prediction_time = (
            time.time()
            if event_time is None
            else float(event_time)
        )
        horizon = min(
            max(
                prediction_time - t2,
                0.0
            ),
            0.25
        )

        return (
            p2[0] + vx * horizon,
            p2[1] + vy * horizon
        )


    def _motion_score(
        self,
        cam_name,
        gid,
        det_box,
        prev_assignments,
        event_time=None
    ):

        history = (
            self._recent_history_for_gid(
                cam_name,
                gid,
                prev_assignments,
                limit=3
            )
        )

        if not history:
            return 0.50

        pred_center = (
            self._predict_center(
                history,
                event_time=event_time
            )
        )

        if pred_center is None:
            return 0.50

        det_center = bbox_center(
            det_box
        )

        dist = float(
            np.hypot(
                det_center[0]
                -
                pred_center[0],
                det_center[1]
                -
                pred_center[1]
            )
        )

        last_box = history[-1].get(
            "box",
            det_box
        )

        lx1, ly1, lx2, ly2 = (
            last_box
        )

        scale = max(
            40.0,
            np.hypot(
                lx2 - lx1,
                ly2 - ly1
            ) * 0.75
        )

        score = (
            1.0
            -
            min(
                dist / scale,
                1.6
            )
        )

        return float(
            max(
                -0.6,
                min(1.0, score)
            )
        )


    # ========================================================
    # CAN MATCH
    # ========================================================

    def _topology_gate(self, identity, cam_name, det, now_ts):
        if identity.get("last_cam") == cam_name:
            return True, "same_camera"
        with topology_lock:
            config = dict(topology_config)
        if not config.get("enforce", False):
            return True, "topology_disabled"
        travel_sec = max(0.0, now_ts - identity.get("last_seen", now_ts))
        for rule in config.get("transitions", []):
            if (
                rule.get("from_camera") == identity.get("last_cam")
                and rule.get("to_camera") == cam_name
            ):
                if float(rule.get("min_travel_sec", 0.0)) <= travel_sec <= float(
                    rule.get("max_travel_sec", float("inf"))
                ):
                    return True, "topology_allowed"
                return False, "travel_time_outside_window"
        return False, "topology_transition_not_allowed"

    def _can_match(
        self,
        identity,
        cam_name,
        now_ts,
        map_pos,
        box_wh
    ):

        state = identity.get("state", IDENTITY_ACTIVE)
        if state == IDENTITY_EXPIRED:
            return False
        topology_ok, _ = self._topology_gate(
            identity,
            cam_name,
            {"map_pos": map_pos},
            now_ts,
        )
        if not topology_ok:
            return False

        dt = (
            now_ts
            -
            identity["last_seen"]
        )

        max_idle = REID_MAX_IDLE_SEC + (
            IDENTITY_DORMANT_TTL_SEC if state == IDENTITY_DORMANT else 0.0
        )
        if dt > max_idle:
            return False

        if not self._size_ratio_ok(
            box_wh,
            identity.get("box_wh")
        ):
            return False

        prev_pos = identity.get(
            "last_map_pos"
        )

        if (
            map_pos is not None
            and
            prev_pos is not None
        ):

            dist = self._map_distance(
                map_pos,
                prev_pos
            )

            if dist is not None:

                if (
                    identity.get(
                        "last_cam"
                    )
                    ==
                    cam_name
                ):

                    if (
                        dist
                        >
                        REID_MAP_GATE_SAME_CAM_PX
                    ):

                        return False

                else:

                    if (
                        dist
                        >
                        REID_MAP_GATE_CROSS_CAM_PX
                    ):

                        return False

        return True

    def _hard_gate_reason(self, identity, cam_name, det, now_ts):
        """Return a stable audit reason instead of only a boolean gate."""
        if identity.get("state", IDENTITY_ACTIVE) == IDENTITY_EXPIRED:
            return "identity_expired"
        topology_ok, topology_reason = self._topology_gate(
            identity, cam_name, det, now_ts
        )
        if not topology_ok:
            return topology_reason
        if now_ts - identity["last_seen"] > REID_MAX_IDLE_SEC:
            return "identity_idle_expired"
        if not self._size_ratio_ok(det.get("box_wh"), identity.get("box_wh")):
            return "incompatible_box_size"
        previous = identity.get("last_map_pos")
        current = det.get("map_pos")
        if previous is not None and current is not None:
            gate = (REID_MAP_GATE_SAME_CAM_PX if identity.get("last_cam") == cam_name
                    else REID_MAP_GATE_CROSS_CAM_PX)
            if self._map_distance(previous, current) > gate:
                return "incompatible_location"
        return None

    def _tracklet_quality_score(self, det):
        """Normalize available crop-quality evidence to [0, 1]."""
        confidence = float(det.get("detector_confidence", det.get("conf", 1.0)))
        crop_w, crop_h = det.get("crop_size", det.get("box_wh", (0, 0)))
        crop_quality = min(1.0, min(crop_w, crop_h) / max(REID_MIN_CROP_SIZE, 1))
        blur = float(det.get("blur_variance", REID_MIN_BLUR_VARIANCE))
        blur_quality = min(1.0, blur / max(REID_MIN_BLUR_VARIANCE, 1e-6))
        occlusion_quality = 0.5 if det.get("overlap", False) else 1.0
        return float(max(0.0, min(1.0, confidence * crop_quality * blur_quality * occlusion_quality)))

    def _ambiguity_reason(self, pair_cache, idx, candidate_gids, cross_camera):
        """Return (reason, margin) for viable top-1/top-2 candidates."""
        viable = sorted(
            (pair_cache[(idx, gid)]["score"] for gid in candidate_gids
             if (idx, gid) in pair_cache and "score" in pair_cache[(idx, gid)]),
            reverse=True,
        )
        if len(viable) < 2:
            return None, None
        margin = float(viable[0] - viable[1])
        required = (ASSIGN_CROSS_CAM_MIN_MARGIN if cross_camera
                    else ASSIGN_SAME_CAM_MIN_MARGIN)
        return ("ambiguous_top1_top2", margin) if margin < required else (None, margin)


    # ========================================================
    # PAIR SCORE
    # ========================================================

    def _pair_score(
        self,
        gid,
        identity,
        cam_name,
        det,
        now_ts,
        prev_assignments
    ):

        appearance = (
            self._gallery_similarity(
                det["emb"],
                identity
            )
        )

        cross_camera = (
            identity.get("last_cam")
            !=
            cam_name
        )

        map_s = self._map_score(
            identity,
            det.get("map_pos"),
            cross_camera=cross_camera
        )

        time_s = self._time_score(
            identity,
            now_ts
        )
        # Motion is camera-coordinate evidence and is intentionally excluded
        # from cross-camera scoring. Keep an explicit diagnostic value so the
        # pair result is complete in both branches.
        motion = 0.0
        quality = self._tracklet_quality_score(det)
        quality_adjusted_appearance = max(0.0, appearance) * quality

        # ====================================================
        # CROSS CAMERA
        # ====================================================

        if cross_camera:

            appearance_score = quality_adjusted_appearance

            map_score = max(
                0.0,
                map_s
            )

            time_score = max(
                0.0,
                time_s
            )

            total = (
                ASSIGN_CROSS_CAM_APPEARANCE_WEIGHT * appearance_score
                + ASSIGN_CROSS_CAM_MAP_WEIGHT * map_score
                + ASSIGN_CROSS_CAM_TIME_WEIGHT * time_score
            )

            source_type = "cross-camera"

        # ====================================================
        # SAME CAMERA
        # ====================================================

        else:

            motion = (
                self._motion_score(
                    cam_name,
                    gid,
                    det["box"],
                    prev_assignments,
                    event_time=now_ts
                )
            )

            app_w = (
                ASSIGN_SAME_CAM_APPEARANCE_WEIGHT
            )

            motion_w = (
                ASSIGN_SAME_CAM_MOTION_WEIGHT
            )

            map_w = (
                ASSIGN_SAME_CAM_MAP_WEIGHT
            )

            time_w = (
                ASSIGN_SAME_CAM_TIME_WEIGHT
            )

            total = (
                app_w * quality_adjusted_appearance
                +
                motion_w * motion
                +
                map_w * map_s
                +
                time_w * time_s
            )

            if (
                identity.get(
                    "last_cam"
                )
                ==
                cam_name
            ):

                total += (
                    ASSIGN_SAME_CAM_BONUS
                )

            source_type = "same-camera"


        # ====================================================
        # STRONG APPEARANCE BONUS
        # ====================================================

        if (
            cross_camera
            and
            appearance
            >=
            REID_CROSS_CAM_STRONG_THRESHOLD
        ):

            total += 0.08


        # ====================================================
        # OCCLUSION
        # ====================================================

        if det.get(
            "overlap",
            False
        ):

            if (
                det.get(
                    "forced_gid"
                )
                ==
                gid
            ):

                total += (
                    ASSIGN_OVERLAP_FREEZE_BONUS
                )


        return {

            "gid": gid,

            "score": float(total),

            "appearance": float(
                appearance
            ),

            "quality_adjusted_appearance": float(quality_adjusted_appearance),

            "tracklet_quality": float(quality),

            "motion": float(
                motion
            ),

            "map": float(
                map_s
            ),

            "time": float(
                time_s
            ),

            "cross_camera": bool(
                cross_camera
            ),

            "source_type": source_type

        }


    # ========================================================
    # ACCEPT MATCH
    # ========================================================

    def _accept_match(
        self,
        pair,
        identity,
        cam_name,
        det
    ):

        if pair is None:
            return False

        appearance = (
            pair["appearance"]
        )

        score = pair["score"]

        cross_camera = (
            pair["cross_camera"]
        )


        # ====================================================
        # CROSS CAMERA
        # ====================================================

        if cross_camera:

            # ReID สูงมาก
            if (
                REID_THRESHOLD_SAFETY_MODE
                == "validated"
                and
                appearance
                >=
                REID_CROSS_CAM_STRONG_THRESHOLD
            ):

                return True

            # ต้องผ่านทั้ง appearance + score
            if (
                appearance
                >=
                REID_CROSS_CAM_THRESHOLD
                and
                score
                >=
                ASSIGN_CROSS_CAM_SCORE_THRESHOLD
            ):

                return True

            return False


        # ====================================================
        # SAME CAMERA
        # ====================================================

        if (
            REID_THRESHOLD_SAFETY_MODE
            == "validated"
            and
            appearance
            >=
            REID_SAME_CAM_STRONG_THRESHOLD
        ):

            return True

        if (
            identity.get(
                "last_cam"
            )
            ==
            cam_name
            and
            appearance
            >=
            REID_SAME_CAM_THRESHOLD
            and
            score
            >=
            ASSIGN_SAME_CAM_SCORE_THRESHOLD
        ):

            return True

        return (
            appearance
            >=
            REID_SAME_CAM_THRESHOLD
            and
            score
            >=
            ASSIGN_SAME_CAM_SCORE_THRESHOLD
        )


    # ========================================================
    # UPDATE IDENTITY
    # ========================================================

    def _transition_identity(self, identity, new_state, now_ts, reason):
        old_state = identity.get("state", IDENTITY_PROVISIONAL)
        if old_state == new_state:
            return False
        identity["state"] = new_state
        identity["state_updated_at"] = float(now_ts)
        identity["state_reason"] = reason
        history = identity.setdefault("state_transitions", [])
        history.append({
            "from": old_state, "to": new_state,
            "ts": float(now_ts), "reason": reason,
        })
        del history[:-IDENTITY_TRANSITION_HISTORY_SIZE]
        return True

    def identity_state_diagnostics(self):
        counts = {state: 0 for state in (
            IDENTITY_PROVISIONAL, IDENTITY_ACTIVE, IDENTITY_DORMANT, IDENTITY_EXPIRED
        )}
        transitions = []
        for gid, identity in self.identities.items():
            counts[identity.get("state", IDENTITY_ACTIVE)] += 1
            transitions.extend({"gid": gid, **item} for item in identity.get("state_transitions", []))
        return {"state_counts": counts, "recent_transitions": sorted(
            transitions, key=lambda item: item["ts"], reverse=True
        )[:IDENTITY_TRANSITION_HISTORY_SIZE]}

    def _gallery_quality_reason(self, det):
        confidence = det.get("detector_confidence", det.get("conf"))
        crop_w, crop_h = det.get("crop_size", det.get("box_wh", (0, 0)))
        if confidence is None or confidence < REID_MIN_DETECTION_CONFIDENCE:
            return "low_detector_confidence"
        if min(crop_w, crop_h) < REID_MIN_CROP_SIZE:
            return "crop_too_small"
        if det.get("blur_variance", 0.0) < REID_MIN_BLUR_VARIANCE:
            return "blurred_crop"
        if det.get("overlap", False) and REID_MAX_OVERLAP_FOR_GALLERY <= 0.0:
            return "overlap_or_occlusion"
        if det.get("border_clip_ratio", 0.0) > REID_MAX_BORDER_CLIP_RATIO:
            return "border_clipped"
        return None

    def _record_tracklet_sample(self, gid, cam_name, local_id, det, now_ts):
        """Update gallery only from diverse, quality-approved tracklet evidence."""
        local_key = (cam_name, int(local_id))
        reason = self._gallery_quality_reason(det)
        identity = self.identities[gid]
        diagnostics = identity.setdefault("gallery_diagnostics", {
            "accepted_updates": 0,
            "rejected_updates": 0,
            "last_rejection_reason": None,
            "tracklet_sample_count": 0,
            "prototype_quality": 0.0,
        })
        if reason is not None:
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = reason
            return False, reason

        samples = self.tracklets.setdefault(local_key, [])
        embedding = l2_normalize(det["emb"])
        if any(cosine_similarity(embedding, item["emb"]) >= REID_GALLERY_DIVERSITY_THRESHOLD for item in samples):
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = "near_duplicate"
            diagnostics["tracklet_sample_count"] = len(samples)
            return False, "near_duplicate"

        samples.append({"emb": embedding, "ts": now_ts})
        if len(samples) > REID_TRACKLET_MAX_SAMPLES:
            del samples[:-REID_TRACKLET_MAX_SAMPLES]

        prototype = l2_normalize(np.mean([item["emb"] for item in samples], axis=0))
        diagnostics["tracklet_sample_count"] = len(samples)
        diagnostics["prototype_quality"] = float(
            np.mean([cosine_similarity(item["emb"], prototype) for item in samples])
        )
        diagnostics["last_rejection_reason"] = None

        if len(samples) < REID_TRACKLET_MIN_SAMPLES:
            return False, "tracklet_not_mature"

        # A provisional identity must collect quality-approved evidence before
        # its prototype can enter permanent gallery memory.
        if identity.get("state", IDENTITY_PROVISIONAL) == IDENTITY_PROVISIONAL:
            self._transition_identity(identity, IDENTITY_ACTIVE, now_ts, "mature_tracklet")

        gallery = identity.setdefault("gallery", [])
        if any(cosine_similarity(prototype, item) >= REID_GALLERY_DIVERSITY_THRESHOLD for item in gallery):
            diagnostics["rejected_updates"] += 1
            diagnostics["last_rejection_reason"] = "prototype_near_duplicate"
            return False, "prototype_near_duplicate"

        gallery.append(prototype)
        if len(gallery) > REID_GALLERY_SIZE:
            del gallery[:-REID_GALLERY_SIZE]
        identity["embedding"] = prototype
        diagnostics["accepted_updates"] += 1
        if self.identity_store:
            self.identity_store.save_identity(
                gid, identity, "gallery_update", "quality_approved", now_ts
            )
        return True, None

    def _update_identity(
        self,
        gid,
        cam_name,
        emb,
        map_pos,
        box_wh,
        now_ts,
        last_score=None
    ):

        identity = (
            self.identities[gid]
        )

        if identity.get("state") == IDENTITY_DORMANT:
            reason = "cross_camera_recovery" if identity.get("last_cam") != cam_name else "same_camera_recovery"
            self._transition_identity(identity, IDENTITY_ACTIVE, now_ts, reason)

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        identity["last_cam"] = cam_name

        identity["last_seen"] = now_ts

        identity["last_map_pos"] = map_pos

        identity["box_wh"] = box_wh

        if last_score is not None:

            identity["last_score"] = float(
                last_score
            )


    # ========================================================
    # REMEMBER SAME CAMERA
    # ========================================================

    def _remember_recent_same_cam(
        self,
        gid,
        cam_name,
        emb,
        map_pos,
        box_wh,
        now_ts
    ):

        self.recent_same_cam.append({

            "gid": gid,

            "cam_name": cam_name,

            "embedding": l2_normalize(
                emb
            ),

            "map_pos": map_pos,

            "box_wh": box_wh,

            "ts": now_ts

        })

        if len(
            self.recent_same_cam
        ) > 300:

            self.recent_same_cam = (
                self.recent_same_cam[
                    -300:
                ]
            )


    # ========================================================
    # REMEMBER CROSS CAMERA
    # ========================================================

    def _remember_cross_cam(
        self,
        gid,
        cam_name,
        emb,
        map_pos,
        now_ts
    ):

        self.recent_cross_cam.append({

            "gid": gid,

            "cam_name": cam_name,

            "embedding": l2_normalize(
                emb
            ),

            "map_pos": map_pos,

            "ts": now_ts

        })

        if len(
            self.recent_cross_cam
        ) > 300:

            self.recent_cross_cam = (
                self.recent_cross_cam[
                    -300:
                ]
            )


    # ========================================================
    # COMMIT ASSIGNMENT
    # ========================================================

    def _commit_assignment(
        self,
        gid,
        cam_name,
        local_id,
        emb,
        map_pos,
        box_wh,
        now_ts,
        score,
        source
    ):

        local_key = (
            cam_name,
            int(local_id)
        )

        if gid in self.identities:

            self._update_identity(
                gid,
                cam_name,
                emb,
                map_pos,
                box_wh,
                now_ts,
                last_score=score
            )

        else:

            self.identities[gid] = {

                "embedding":
                    l2_normalize(emb),

                # The first frame is a temporary bootstrap for matching.
                # It does not become gallery evidence until its tracklet
                # reaches the quality-gated prototype stage.
                "gallery": [],

                "gallery_diagnostics": {
                    "accepted_updates": 0,
                    "rejected_updates": 0,
                    "last_rejection_reason": None,
                    "tracklet_sample_count": 0,
                    "prototype_quality": 0.0,
                },

                "last_cam":
                    cam_name,

                "last_seen":
                    now_ts,

                "last_map_pos":
                    map_pos,

                "box_wh":
                    box_wh,

                "last_score":
                    float(score),

                "state": IDENTITY_PROVISIONAL,

                "state_updated_at": float(now_ts),

                "state_reason": "new_tracklet",

                "state_transitions": [{
                    "from": None, "to": IDENTITY_PROVISIONAL,
                    "ts": float(now_ts), "reason": "new_tracklet",
                }]

            }


        self.local_to_global[
            local_key
        ] = {

            "gid": gid,

            "last_seen": now_ts

        }


        self._remember_recent_same_cam(
            gid,
            cam_name,
            emb,
            map_pos,
            box_wh,
            now_ts
        )


        if source in ("cross-camera", "global-cross-camera"):

            self._remember_cross_cam(
                gid,
                cam_name,
                emb,
                map_pos,
                now_ts
            )

        if self.identity_store:
            self.identity_store.save_identity(
                gid, self.identities[gid], "assignment", source, now_ts
            )

        return {

            "gid": gid,

            "score": float(score),

            "source": source

        }


    # ========================================================
    # NEW GLOBAL ID
    # ========================================================

    def _new_identity(
        self,
        cam_name,
        local_id,
        emb,
        map_pos,
        box_wh,
        now_ts
    ):

        gid = self.next_global_id

        self.next_global_id += 1

        return self._commit_assignment(

            gid,

            cam_name,

            local_id,

            emb,

            map_pos,

            box_wh,

            now_ts,

            1.0,

            "new"

        )


    # ========================================================
    # RECENT SAME CAMERA MATCH
    # ========================================================

    def _find_recent_same_cam_match(
        self,
        cam_name,
        emb,
        map_pos,
        box_wh,
        now_ts,
        used_gids=None
    ):

        best_gid = None

        best_score = -999.0

        for item in reversed(
            self.recent_same_cam
        ):

            if (
                item.get("cam_name")
                !=
                cam_name
            ):

                continue

            dt = (
                now_ts
                -
                item.get(
                    "ts",
                    0
                )
            )

            if (
                dt
                >
                REID_RECENT_SAME_CAM_SEC
            ):

                continue

            gid = item.get(
                "gid"
            )

            if gid not in self.identities:
                continue

            if (
                used_gids is not None
                and
                gid in used_gids
            ):

                continue

            if not self._size_ratio_ok(
                box_wh,
                item.get("box_wh")
            ):

                continue

            score = cosine_similarity(
                emb,
                item.get(
                    "embedding"
                )
            )

            if (
                map_pos is not None
                and
                item.get("map_pos") is not None
            ):

                dist = self._map_distance(
                    map_pos,
                    item.get("map_pos")
                )

                if dist is not None:

                    score -= (
                        min(
                            dist
                            /
                            max(
                                REID_MAP_GATE_SAME_CAM_PX,
                                1.0
                            ),
                            1.0
                        )
                        *
                        0.10
                    )


            score -= (
                min(
                    dt
                    /
                    max(
                        REID_RECENT_SAME_CAM_SEC,
                        1e-6
                    ),
                    1.0
                )
                *
                0.08
            )


            if (
                score
                >
                best_score
                and
                score
                >=
                REID_RECENT_SAME_CAM_THRESHOLD
            ):

                best_gid = gid

                best_score = score


        return (
            best_gid,
            float(best_score)
        )


    # ========================================================
    # BATCH ASSIGNMENT
    # ========================================================

    def assign_batch(
        self,
        cam_name,
        detections,
        prev_assignments=None,
        event_time=None
    ):

        now_ts = (
            time.time()
            if event_time is None
            else float(event_time)
        )

        prev_assignments = (
            prev_assignments
            or
            []
        )

        results = [
            None
            for _ in detections
        ]


        with self.lock:

            self.cleanup(
                reference_time=now_ts
            )

            used_gids = set()

            pending_indices = []


            # =================================================
            # 1. LOCAL TRACK / OCCLUSION
            # =================================================

            for idx, det in enumerate(
                detections
            ):

                local_key = (
                    cam_name,
                    int(det["tid"])
                )


                # -------------------------------------------------
                # Occlusion hold
                # -------------------------------------------------

                hold = (
                    self.occlusion_hold.get(
                        local_key
                    )
                )

                if hold is not None:

                    if (
                        now_ts
                        <=
                        hold.get(
                            "until_ts",
                            0
                        )
                    ):

                        gid = hold.get(
                            "gid"
                        )

                        if (
                            gid
                            in
                            self.identities
                            and
                            gid
                            not in
                            used_gids
                        ):

                            results[idx] = (
                                self._commit_assignment(
                                    gid,
                                    cam_name,
                                    det["tid"],
                                    det["emb"],
                                    det.get(
                                        "map_pos"
                                    ),
                                    det.get(
                                        "box_wh"
                                    ),
                                    now_ts,
                                    hold.get(
                                        "score",
                                        1.0
                                    ),
                                    "occlusion-hold"
                                )
                            )

                            used_gids.add(
                                gid
                            )

                            continue


                # -------------------------------------------------
                # Existing local track
                # -------------------------------------------------

                existing = self.local_to_global.get(local_key)

                if existing is not None:

                    gid = existing.get("gid")

                    if (
                        gid in self.identities
                        and
                        gid not in used_gids
                    ):

                        identity = self.identities[gid]

                        appearance = self._gallery_similarity(
                            det["emb"],
                            identity
                        )

                        # -----------------------------------------------
                        # Local ID + Appearance ตรงกัน
                        # -----------------------------------------------

                        if appearance >= LOCAL_TRACK_VERIFY_THRESHOLD:

                            results[idx] = (
                                self._commit_assignment(
                                    gid,
                                    cam_name,
                                    det["tid"],
                                    det["emb"],
                                    det.get("map_pos"),
                                    det.get("box_wh"),
                                    now_ts,
                                    appearance,
                                    "local-track-verified"
                                )
                            )

                            used_gids.add(gid)

                            continue

                        # -----------------------------------------------
                        # Local ID เดิม แต่ Appearance ไม่ตรง
                        #
                        # อย่าเชื่อ Local ID
                        # ปล่อยลง global matching
                        # -----------------------------------------------

                        if REID_DEBUG:

                            logger.warning(
                                f"[REID] Local ID conflict | "
                                f"CAM={cam_name} "
                                f"LID={det['tid']} "
                                f"GID={gid} "
                                f"appearance={appearance:.3f}"
                            )

                        self.local_to_global.pop(
                            local_key,
                            None
                        )

                    self.local_to_global.pop(
                        local_key,
                        None
                    )


                # -------------------------------------------------
                # Forced GID
                # -------------------------------------------------

                if (
                    det.get(
                        "overlap",
                        False
                    )
                    and
                    det.get(
                        "forced_gid"
                    )
                    in
                    self.identities
                    and
                    det.get(
                        "forced_gid"
                    )
                    not in
                    used_gids
                ):

                    gid = det.get(
                        "forced_gid"
                    )

                    results[idx] = (
                        self._commit_assignment(
                            gid,
                            cam_name,
                            det["tid"],
                            det["emb"],
                            det.get(
                                "map_pos"
                            ),
                            det.get(
                                "box_wh"
                            ),
                            now_ts,
                            1.0,
                            "occlusion-forced"
                        )
                    )

                    used_gids.add(
                        gid
                    )

                    continue


                pending_indices.append(
                    idx
                )


            # =================================================
            # 2. GLOBAL MATCHING
            # =================================================

            candidate_gids = []

            for gid, identity in (
                self.identities.items()
            ):

                if gid in used_gids:
                    continue

                reusable = False

                for idx in pending_indices:

                    det = detections[idx]

                    if self._can_match(
                        identity,
                        cam_name,
                        now_ts,
                        det.get(
                            "map_pos"
                        ),
                        det.get(
                            "box_wh"
                        )
                    ):

                        reusable = True

                        break

                if reusable:

                    candidate_gids.append(
                        gid
                    )


            pair_cache = {}


            if (
                pending_indices
                and
                candidate_gids
            ):

                score_matrix = np.full(

                    (
                        len(
                            pending_indices
                        ),
                        len(
                            candidate_gids
                        )
                    ),

                    -1e6,

                    dtype=np.float32

                )


                for r, idx in enumerate(
                    pending_indices
                ):

                    det = detections[idx]

                    for c, gid in enumerate(
                        candidate_gids
                    ):

                        gate_reason = self._hard_gate_reason(
                            self.identities[gid], cam_name, det, now_ts
                        )
                        if gate_reason is not None:
                            pair_cache[(idx, gid)] = {"gate_failure": gate_reason}
                            continue
                        pair = (
                            self._pair_score(
                                gid,
                                self.identities[
                                    gid
                                ],
                                cam_name,
                                det,
                                now_ts,
                                prev_assignments
                            )
                        )

                        pair_cache[
                            (idx, gid)
                        ] = pair

                        score_matrix[
                            r,
                            c
                        ] = pair[
                            "score"
                        ]


                # Hungarian assignment

                row_ind, col_ind = (
                    linear_sum_assignment(
                        -score_matrix
                    )
                )

                matched_rows = set()


                for r, c in zip(
                    row_ind.tolist(),
                    col_ind.tolist()
                ):

                    idx = (
                        pending_indices[r]
                    )

                    gid = (
                        candidate_gids[c]
                    )

                    pair = pair_cache[
                        (idx, gid)
                    ]

                    det = detections[
                        idx
                    ]

                    identity = (
                        self.identities.get(
                            gid
                        )
                    )

                    if identity is None:
                        continue


                    if not self._accept_match(
                        pair,
                        identity,
                        cam_name,
                        det
                    ):

                        continue


                    if gid in used_gids:
                        continue


                    # With unvalidated thresholds, do not let Hungarian
                    # tie-breaking turn multiple acceptable cross-camera
                    # candidates into an identity merge. A measured margin
                    # can replace this conservative rejection only after the
                    # threshold report is validation-backed.
                    if (
                        REID_THRESHOLD_SAFETY_MODE
                        == "conservative"
                        and pair["cross_camera"]
                    ):
                        acceptable_cross_camera = 0

                        for candidate_gid in candidate_gids:
                            if candidate_gid in used_gids:
                                continue

                            candidate_identity = (
                                self.identities.get(
                                    candidate_gid
                                )
                            )
                            candidate_pair = pair_cache.get(
                                (idx, candidate_gid)
                            )

                            if (
                                candidate_identity is not None
                                and candidate_pair is not None
                                and candidate_pair["cross_camera"]
                                and self._accept_match(
                                    candidate_pair,
                                    candidate_identity,
                                    cam_name,
                                    det
                                )
                            ):
                                acceptable_cross_camera += 1

                        if acceptable_cross_camera > 1:
                            continue


                    # ---------------------------------------------
                    # Source
                    # ---------------------------------------------

                    if pair[
                        "cross_camera"
                    ]:

                        source = (
                            "cross-camera"
                        )

                    else:

                        source = (
                            "batch-match"
                        )


                    results[idx] = (
                        self._commit_assignment(
                            gid,
                            cam_name,
                            det["tid"],
                            det["emb"],
                            det.get(
                                "map_pos"
                            ),
                            det.get(
                                "box_wh"
                            ),
                            now_ts,
                            pair["score"],
                            source
                        )
                    )


                    used_gids.add(
                        gid
                    )

                    matched_rows.add(
                        idx
                    )


                pending_indices = [

                    idx

                    for idx in pending_indices

                    if idx not in matched_rows

                ]


            # =================================================
            # 3. SAME CAMERA CACHE
            # =================================================

            still_pending = []


            for idx in pending_indices:

                det = detections[idx]

                gid, recent_score = (
                    self._find_recent_same_cam_match(
                        cam_name,
                        det["emb"],
                        det.get(
                            "map_pos"
                        ),
                        det.get(
                            "box_wh"
                        ),
                        now_ts,
                        used_gids=used_gids
                    )
                )


                if (
                    gid is not None
                    and
                    gid in self.identities
                ):

                    results[idx] = (
                        self._commit_assignment(
                            gid,
                            cam_name,
                            det["tid"],
                            det["emb"],
                            det.get(
                                "map_pos"
                            ),
                            det.get(
                                "box_wh"
                            ),
                            now_ts,
                            recent_score,
                            "same-cam-cache"
                        )
                    )

                    used_gids.add(
                        gid
                    )

                else:

                    still_pending.append(
                        idx
                    )


            # =================================================
            # 4. CREATE NEW ID
            # =================================================

            for idx in still_pending:

                det = detections[idx]

                results[idx] = (
                    self._new_identity(
                        cam_name,
                        det["tid"],
                        det["emb"],
                        det.get(
                            "map_pos"
                        ),
                        det.get(
                            "box_wh"
                        ),
                        now_ts
                    )
                )


            # =================================================
            # 5. OCCLUSION HOLD
            # =================================================

            for idx, det in enumerate(
                detections
            ):

                if results[idx] is None:
                    continue

                if det.get(
                    "overlap",
                    False
                ) and det.get(
                    "local_track_confirmed",
                    True
                ):

                    local_key = (
                        cam_name,
                        int(det["tid"])
                    )

                    self.occlusion_hold[
                        local_key
                    ] = {

                        "gid":
                            results[idx]["gid"],

                        "until_ts":
                            now_ts
                            +
                            OCCLUSION_HOLD_SEC,

                        "score":
                            float(
                                results[idx][
                                    "score"
                                ]
                            )

                    }


            # Detector rows without a confirmed BoT-SORT ID receive a
            # frame-unique ephemeral key. They may use Re-ID evidence for the
            # current frame, but must not become local-ID history.
            for det in detections:
                if det.get(
                    "local_track_confirmed",
                    True
                ):
                    continue

                ephemeral_key = (
                    cam_name,
                    int(det["tid"])
                )
                self.local_to_global.pop(
                    ephemeral_key,
                    None
                )
                self.occlusion_hold.pop(
                    ephemeral_key,
                    None
                )

            # A successful identity assignment is deliberately separate from
            # gallery admission.  Thus every frame can retain normal tracking
            # behaviour while only a mature, quality-approved tracklet alters
            # the long-lived appearance memory.
            for idx, result in enumerate(results):
                if result is None:
                    continue
                accepted, reason = self._record_tracklet_sample(
                    result["gid"],
                    cam_name,
                    detections[idx]["tid"],
                    detections[idx],
                    now_ts,
                )
                result["gallery_update_accepted"] = accepted
                result["gallery_rejection_reason"] = reason


        return results


    def _trusted_assignment_claim(
        self,
        cam_name,
        detection,
        event_time,
        drop_conflicting_local=False,
    ):
        """Return existing local/occlusion evidence without global scoring."""
        local_key = (cam_name, int(detection["tid"]))
        hold = self.occlusion_hold.get(local_key)
        if (
            hold is not None
            and event_time <= hold.get("until_ts", 0)
            and hold.get("gid") in self.identities
        ):
            return {
                "gid": hold["gid"],
                "score": float(hold.get("score", 1.0)),
                "source": "occlusion-hold",
                "priority": 3,
            }

        existing = self.local_to_global.get(local_key)
        if existing is not None:
            gid = existing.get("gid")
            if gid in self.identities:
                appearance = self._gallery_similarity(
                    detection["emb"],
                    self.identities[gid]
                )
                if appearance >= LOCAL_TRACK_VERIFY_THRESHOLD:
                    return {
                        "gid": gid,
                        "score": float(appearance),
                        "source": "local-track-verified",
                        "priority": 2,
                    }
                if REID_DEBUG:
                    logger.warning(
                        "[REID] Local ID conflict | CAM=%s LID=%s GID=%s appearance=%.3f",
                        cam_name,
                        detection["tid"],
                        gid,
                        appearance,
                    )
            if drop_conflicting_local:
                self.local_to_global.pop(local_key, None)

        forced_gid = detection.get("forced_gid")
        if (
            detection.get("overlap", False)
            and forced_gid in self.identities
        ):
            return {
                "gid": forced_gid,
                "score": 1.0,
                "source": "occlusion-forced",
                "priority": 1,
            }

        return None

    def preview_trusted_assignments(
        self,
        cam_name,
        detections,
        event_time=None,
        blocking=True,
    ):
        """Expose already-established evidence while global work is pending.

        This read-only preview keeps normal labels/map updates visible after a
        track has been globally established.  It never creates, matches, or
        mutates an identity; the coordinator's global batch remains the sole
        production assignment decision.
        """
        row_event_time = (
            time.time()
            if event_time is None
            else float(event_time)
        )
        results = [None for _ in detections]
        used_gids = set()

        acquired = self.lock.acquire(
            blocking=bool(blocking)
        )
        if not acquired:
            return results

        try:
            for index, detection in enumerate(detections):
                claim = self._trusted_assignment_claim(
                    cam_name,
                    detection,
                    float(detection.get("event_time", row_event_time)),
                    drop_conflicting_local=False,
                )
                if claim is None or claim["gid"] in used_gids:
                    continue
                results[index] = {
                    "gid": claim["gid"],
                    "score": claim["score"],
                    "source": claim["source"],
                    "gallery_update_accepted": False,
                    "gallery_rejection_reason": "pending_global_batch",
                }
                used_gids.add(claim["gid"])
        finally:
            self.lock.release()

        return results


    # ========================================================
    # GLOBAL MULTI-CAMERA BATCH ASSIGNMENT
    # ========================================================

    def assign_global_batch(
        self,
        camera_detections,
        prev_assignments_by_camera=None,
        event_time=None,
        batch_id=None,
        assignment_window_sec=None,
    ):
        """Match simultaneous detections from all cameras in one assignment.

        ``camera_detections`` is a mapping of camera name to that camera's
        detection list.  This is deliberately model-only: callers decide how
        to form the bounded rendezvous window, while this method makes one
        atomic identity decision for the supplied window.
        """
        default_event_time = (
            time.time()
            if event_time is None
            else float(event_time)
        )
        previous = prev_assignments_by_camera or {}
        results = {
            cam_name: [None for _ in detections]
            for cam_name, detections in camera_detections.items()
        }
        rows = [
            (
                cam_name,
                index,
                detection,
                float(
                    detection.get(
                        "event_time",
                        default_event_time
                    )
                ),
            )
            for cam_name, detections in camera_detections.items()
            for index, detection in enumerate(detections)
        ]
        event_times = [row[3] for row in rows]
        batch_event_time = (
            max(event_times)
            if event_times
            else default_event_time
        )
        resolved_batch_id = (
            str(batch_id)
            if batch_id is not None
            else f"global-{batch_event_time:.6f}"
        )

        with self.lock:
            previous_local_mappings = {
                row: (
                    dict(mapping)
                    if mapping is not None
                    else None
                )
                for row, (cam_name, _, detection, _) in enumerate(rows)
                for mapping in [
                    self.local_to_global.get(
                        (cam_name, int(detection["tid"]))
                    )
                ]
            }
            self.cleanup(
                reference_time=batch_event_time
            )
            used_gids = set()
            assigned_rows = set()
            rejections = []

            # Resolve trusted camera-local evidence before building the
            # Hungarian matrix.  Its GID is then excluded from every
            # conflicting row in this global decision.
            fixed_claims = []
            for row, (cam_name, _, detection, row_event_time) in enumerate(rows):
                claim = self._trusted_assignment_claim(
                    cam_name,
                    detection,
                    row_event_time,
                    drop_conflicting_local=True,
                )

                if claim is not None:
                    fixed_claims.append((row, claim))

            fixed_claims.sort(
                key=lambda item: (
                    -item[1]["priority"],
                    -item[1]["score"],
                    item[0],
                )
            )

            for row, claim in fixed_claims:
                cam_name, index, detection, row_event_time = rows[row]
                gid = claim["gid"]
                if gid in used_gids:
                    rejections.append({
                        "row": row,
                        "camera": cam_name,
                        "track_id": detection["tid"],
                        "gid": gid,
                        "reason": "trusted_gid_conflict",
                    })
                    continue

                results[cam_name][index] = self._commit_assignment(
                    gid,
                    cam_name,
                    detection["tid"],
                    detection["emb"],
                    detection.get("map_pos"),
                    detection.get("box_wh"),
                    row_event_time,
                    claim["score"],
                    claim["source"],
                )
                used_gids.add(gid)
                assigned_rows.add(row)

            pending_rows = [
                row
                for row in range(len(rows))
                if row not in assigned_rows
            ]
            candidate_gids = [
                gid
                for gid in self.identities
                if gid not in used_gids
            ]
            score_matrix = np.full(
                (len(pending_rows), len(candidate_gids)),
                -1e6,
                dtype=np.float32
            )
            pair_cache = {}

            for matrix_row, row in enumerate(pending_rows):
                cam_name, _, detection, row_event_time = rows[row]
                for column, gid in enumerate(candidate_gids):
                    identity = self.identities[gid]
                    gate_reason = self._hard_gate_reason(
                        identity,
                        cam_name,
                        detection,
                        row_event_time
                    )
                    if gate_reason is not None:
                        pair_cache[(row, gid)] = {"gate_failure": gate_reason}
                        continue

                    pair = self._pair_score(
                        gid,
                        identity,
                        cam_name,
                        detection,
                        row_event_time,
                        previous.get(cam_name, []),
                    )
                    pair_cache[(row, gid)] = pair
                    score_matrix[matrix_row, column] = pair["score"]

            candidate_details_by_row = {}
            top1_top2_margin_by_row = {}
            for row in pending_rows:
                viable_scores = sorted(
                    [
                        float(pair_cache[(row, gid)]["score"])
                        for gid in candidate_gids
                        if (
                            (row, gid) in pair_cache
                            and "score" in pair_cache[(row, gid)]
                        )
                    ],
                    reverse=True,
                )
                top1_top2_margin_by_row[row] = (
                    float(viable_scores[0] - viable_scores[1])
                    if len(viable_scores) >= 2
                    else None
                )
                candidate_details_by_row[row] = [
                    {
                        "gid": gid,
                        "hard_gate_passed": (
                            "gate_failure" not in pair_cache[(row, gid)]
                        ),
                        "hard_gate_reason": pair_cache[(row, gid)].get(
                            "gate_failure"
                        ),
                        "appearance": pair_cache[(row, gid)].get(
                            "appearance"
                        ),
                        "score": pair_cache[(row, gid)].get("score"),
                        "motion": pair_cache[(row, gid)].get("motion"),
                    }
                    for gid in candidate_gids
                    if (row, gid) in pair_cache
                ]

            selected = []
            if pending_rows and candidate_gids:
                row_ind, col_ind = linear_sum_assignment(-score_matrix)
                for matrix_row, column in zip(
                    row_ind.tolist(),
                    col_ind.tolist()
                ):
                    row = pending_rows[matrix_row]
                    gid = candidate_gids[column]
                    pair = pair_cache.get((row, gid))
                    cam_name, _, detection, _ = rows[row]
                    identity = self.identities.get(gid)
                    if (
                        pair is None
                        or "score" not in pair
                        or identity is None
                    ):
                        continue
                    if not self._accept_match(pair, identity, cam_name, detection):
                        rejections.append({
                            "row": row,
                            "camera": cam_name,
                            "track_id": detection["tid"],
                            "gid": gid,
                            "reason": "acceptance_threshold",
                            "score": float(pair["score"]),
                        })
                        continue
                    ambiguity_reason, margin = self._ambiguity_reason(
                        pair_cache, row, candidate_gids, pair["cross_camera"]
                    )
                    if ambiguity_reason is not None:
                        pair["reject_reason"] = ambiguity_reason
                        pair["top1_top2_margin"] = margin
                        rejections.append({
                            "row": row,
                            "camera": cam_name,
                            "track_id": detection["tid"],
                            "gid": gid,
                            "reason": ambiguity_reason,
                            "score": float(pair["score"]),
                            "top1_top2_margin": margin,
                        })
                        continue
                    selected.append((row, gid, pair))

            # Commit only after Hungarian has selected the complete global
            # one-to-one set.  Existing identities omitted from this set are
            # intentionally left untouched so their lifecycle continues.
            for row, gid, pair in selected:
                cam_name, index, detection, row_event_time = rows[row]
                source = "global-cross-camera" if pair["cross_camera"] else "global-batch"
                results[cam_name][index] = self._commit_assignment(
                    gid, cam_name, detection["tid"], detection["emb"],
                    detection.get("map_pos"), detection.get("box_wh"),
                    row_event_time,
                    pair["score"], source,
                )
                used_gids.add(gid)
                assigned_rows.add(row)

            # Preserve the existing same-camera cache fallback, while keeping
            # every GID already used by this global decision unavailable.
            for row in pending_rows:
                if row in assigned_rows:
                    continue
                cam_name, index, detection, row_event_time = rows[row]
                gid, recent_score = self._find_recent_same_cam_match(
                    cam_name,
                    detection["emb"],
                    detection.get("map_pos"),
                    detection.get("box_wh"),
                    row_event_time,
                    used_gids=used_gids,
                )
                if gid is None or gid not in self.identities:
                    continue
                results[cam_name][index] = self._commit_assignment(
                    gid,
                    cam_name,
                    detection["tid"],
                    detection["emb"],
                    detection.get("map_pos"),
                    detection.get("box_wh"),
                    row_event_time,
                    recent_score,
                    "same-cam-cache",
                )
                used_gids.add(gid)
                assigned_rows.add(row)

            new_identity_reasons = {}
            for row, (cam_name, index, detection, row_event_time) in enumerate(rows):
                if row in assigned_rows:
                    continue
                row_rejections = [
                    rejection["reason"]
                    for rejection in rejections
                    if rejection.get("row") == row
                ]
                row_pairs = [
                    pair_cache[(row, gid)]
                    for gid in candidate_gids
                    if (row, gid) in pair_cache
                ]
                if not candidate_gids:
                    new_reason = "no_eligible_candidate"
                elif row_rejections:
                    new_reason = row_rejections[-1]
                elif row_pairs and all(
                    "gate_failure" in pair
                    for pair in row_pairs
                ):
                    new_reason = "all_candidates_hard_gated"
                else:
                    new_reason = "unmatched_global_assignment"
                results[cam_name][index] = self._new_identity(
                    cam_name, detection["tid"], detection["emb"],
                    detection.get("map_pos"), detection.get("box_wh"),
                    row_event_time,
                )
                results[cam_name][index]["assignment_reason"] = new_reason
                new_identity_reasons[row] = new_reason
                assigned_rows.add(row)

            for cam_name, index, detection, row_event_time in rows:
                result = results[cam_name][index]
                if (
                    detection.get("overlap", False)
                    and detection.get("local_track_confirmed", True)
                ):
                    self.occlusion_hold[(cam_name, int(detection["tid"]))] = {
                        "gid": result["gid"],
                        "until_ts": row_event_time + OCCLUSION_HOLD_SEC,
                        "score": float(result["score"]),
                    }

                if not detection.get("local_track_confirmed", True):
                    ephemeral_key = (cam_name, int(detection["tid"]))
                    self.local_to_global.pop(ephemeral_key, None)
                    self.occlusion_hold.pop(ephemeral_key, None)

                accepted, reason = self._record_tracklet_sample(
                    result["gid"],
                    cam_name,
                    detection["tid"],
                    detection,
                    row_event_time,
                )
                result["gallery_update_accepted"] = accepted
                result["gallery_rejection_reason"] = reason

            gate_failures = [
                {
                    "camera": rows[row][0],
                    "track_id": rows[row][2]["tid"],
                    "gid": gid,
                    "reason": pair["gate_failure"],
                    "row": row,
                }
                for (row, gid), pair in pair_cache.items()
                if "gate_failure" in pair
            ]
            assignments = [
                {
                    "row": row,
                    "camera": cam_name,
                    "track_id": detection["tid"],
                    "gid": results[cam_name][index]["gid"],
                    "score": float(results[cam_name][index]["score"]),
                    "source": results[cam_name][index]["source"],
                    "identity_state": self.identities.get(
                        results[cam_name][index]["gid"],
                        {}
                    ).get("state", IDENTITY_PROVISIONAL),
                    "assignment_state": "committed",
                    "reason": results[cam_name][index].get(
                        "assignment_reason",
                        results[cam_name][index]["source"],
                    ),
                }
                for row, (cam_name, index, detection, _) in enumerate(rows)
            ]
            self.last_global_batch_diagnostics = {
                "batch_id": resolved_batch_id,
                "event_time": batch_event_time,
                "window_start_event_time": (
                    min(event_times) if event_times else batch_event_time
                ),
                "window_end_event_time": (
                    max(event_times) if event_times else batch_event_time
                ),
                "assignment_window_sec": (
                    GLOBAL_ASSIGNMENT_WINDOW_SEC
                    if assignment_window_sec is None
                    else float(assignment_window_sec)
                ),
                "cameras": sorted(camera_detections),
                "observation_count": len(rows),
                "rows": [
                    {
                        "row": row,
                        "camera": cam_name,
                        "track_id": detection["tid"],
                        "event_time": row_event_time,
                        "sequence_index": detection.get(
                            "frame_index",
                            detection.get("sequence_index"),
                        ),
                        "previous_local_mapping": (
                            previous_local_mappings.get(row)
                        ),
                        "candidate_gids": [
                            item["gid"]
                            for item in candidate_details_by_row.get(row, [])
                        ],
                        "candidates": candidate_details_by_row.get(row, []),
                        "top1_top2_margin": top1_top2_margin_by_row.get(row),
                        "assignment_state": "committed",
                        "batch_id": resolved_batch_id,
                        "generation": detection.get(
                            "coordinator_generation",
                            detection.get("camera_generation"),
                        ),
                        "final_gid": results[cam_name][index]["gid"],
                        "final_state": self.identities.get(
                            results[cam_name][index]["gid"],
                            {},
                        ).get("state", IDENTITY_PROVISIONAL),
                        "new_identity_reason": new_identity_reasons.get(row),
                    }
                    for row, (
                        cam_name,
                        index,
                        detection,
                        row_event_time,
                    ) in enumerate(rows)
                ],
                "candidate_gids": candidate_gids,
                "gate_failures": gate_failures,
                "rejections": rejections,
                "assignments": assignments,
                "selected": [
                    {
                        "camera": rows[row][0],
                        "track_id": rows[row][2]["tid"],
                        "gid": gid,
                        "score": float(pair["score"]),
                    }
                    for row, gid, pair in selected
                ],
            }

            logger.info(
                "[REID][GLOBAL] batch=%s cameras=%s observations=%d assignments=%s rejections=%d",
                resolved_batch_id,
                ",".join(sorted(camera_detections)) or "none",
                len(rows),
                ",".join(
                    f"{item['camera']}:{item['track_id']}->G{item['gid']}({item['source']})"
                    for item in assignments
                ) or "none",
                len(gate_failures) + len(rejections),
            )

        return results


    # ========================================================
    # RESOLVE ID
    # ========================================================

    def resolve_identity(
        self,
        cam_name,
        local_id,
        emb,
        map_pos=None,
        box_wh=None,
        forbidden_gids=None,
        forced_gid=None,
        allow_new=True
    ):

        det = {

            "tid":
                local_id,

            "emb":
                emb,

            "map_pos":
                map_pos,

            "box_wh":
                box_wh,

            "box":
                (
                    0,
                    0,
                    box_wh[0]
                    if box_wh
                    else 1,
                    box_wh[1]
                    if box_wh
                    else 1
                ),

            "overlap":
                forced_gid is not None,

            "forced_gid":
                forced_gid

        }

        result = self.assign_batch(
            cam_name,
            [det],
            prev_assignments=[]
        )[0]

        return (
            result["gid"],
            result["score"],
            result["source"]
        )


# ============================================================
# DOWNSTREAM GLOBAL ASSIGNMENT COORDINATOR
# ============================================================

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
            if self._event_time_outside_pending_window_locked(
                observation_event_time
            ):
                batch = self._take_pending_locked()
                if batch is not None:
                    self._enqueue_ready_batch_locked(batch)

            if self.current_batch_id is None:
                self.current_batch_id = self._new_batch_id_locked()

            self.camera_epochs.setdefault(cam_name, 0)
            camera_epoch = self.camera_epochs[cam_name]
            for detection in bounded_detections:
                detection["coordinator_generation"] = camera_epoch

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
        timeout_sec=2.0
    ):

        self.trail_len = trail_len

        self.timeout_sec = timeout_sec

        self.base_map = None

        self.objects = {}

        self.tracks = {}

        self.last_seen = {}

        self.lock = threading.Lock()

        self.load_floorplan()


    def load_floorplan(self):

        if os.path.exists(
            FLOORPLAN_PATH
        ):

            img = cv2.imread(
                FLOORPLAN_PATH
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

global_identity_manager = (
    GlobalIdentityManager(IdentityStore(IDENTITY_DB_PATH))
)

global_assignment_coordinator = GlobalAssignmentCoordinator(
    lambda: global_identity_manager,
    window_sec=GLOBAL_ASSIGNMENT_WINDOW_SEC,
)

global_map = GlobalMapManager(
    trail_len=1,
    timeout_sec=0.7
)


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
        loop_video=True,
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

    def _open_capture(self):
        with self.state_lock:
            is_reconnect = self._open_attempts > 0
            self._open_attempts += 1

            if is_reconnect:
                self.reconnect_count += 1

        try:
            capture = cv2.VideoCapture(self.source)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            opened = bool(capture.isOpened())
        except Exception as error:
            self._set_capture_error(
                f"Capture open failed: {error}"
            )
            return None

        if not opened:
            self._set_capture_error("Capture did not open")
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
                    read_error = f"Capture read failed: {error}"
                else:
                    read_error = "Capture read returned no frame"

                if not success or frame is None:
                    if self.stop_event.is_set():
                        break

                    self._set_capture_error(read_error)

                    with self.state_lock:
                        if self._ever_captured:
                            self._tracker_reset_pending = True

                    self._release_capture(capture)
                    capture = None

                    logger.warning(
                        "[LIVE] Source unavailable; reconnecting | camera=%s",
                        self.cam_name
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

def point_in_polygon(
    point,
    polygon_pts
):

    if polygon_pts is None:
        return True

    poly = np.array(
        polygon_pts,
        dtype=np.int32
    )

    return (
        cv2.pointPolygonTest(
            poly,
            point,
            False
        )
        >=
        0
    )


# ============================================================
# PERSON EMBEDDING
# ============================================================

def extract_person_embedding(
    frame,
    x1,
    y1,
    x2,
    y2
):

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = (
        clamp_bbox(
            x1,
            y1,
            x2,
            y2,
            w,
            h
        )
    )


    bw = x2 - x1

    bh = y2 - y1


    if (
        bw <= 1
        or
        bh <= 1
    ):

        return None


    # --------------------------------------------------------
    # Crop margin
    # --------------------------------------------------------

    sx = int(
        bw
        *
        REID_CROP_SIDE_MARGIN
    )

    sy_top = int(
        bh
        *
        REID_CROP_TOP_MARGIN
    )

    sy_bottom = int(
        bh
        *
        REID_CROP_BOTTOM_MARGIN
    )


    cx1 = x1 + sx

    cx2 = x2 - sx

    cy1 = y1 + sy_top

    cy2 = y2 - sy_bottom


    cx1, cy1, cx2, cy2 = (
        clamp_bbox(
            cx1,
            cy1,
            cx2,
            cy2,
            w,
            h
        )
    )


    crop = frame[
        cy1:cy2,
        cx1:cx2
    ]


    if (
        crop is None
        or
        crop.size == 0
    ):

        return None


    return appearance_extractor.extract(
        crop
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
    detection_boxes
):

    forced = {}

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
    annotated_frame = frame.copy()
    active_local_track_ids = []
    event_ts = (
        time.time()
        if event_time is None
        else float(event_time)
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

            # ------------------------------------------------
            # Detection
            # ------------------------------------------------

            for i, box in enumerate(xyxy_list):

                x1, y1, x2, y2 = box[:4]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                # ROI
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

                # ------------------------------------------------
                # OSNet / ReID
                # ------------------------------------------------

                reid_started = time.perf_counter()
                emb = extract_person_embedding(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2
                )
                reid_feature_duration_ms += (
                    (time.perf_counter() - reid_started) * 1000.0
                )

                if emb is None:
                    continue

                quality_meta = build_reid_quality_metadata(
                    frame,
                    (x1, y1, x2, y2),
                    conf_val,
                )

                # ------------------------------------------------
                # Homography
                # ------------------------------------------------

                map_pos = None

                if processor is not None:

                    try:

                        map_x, map_y = processor.to_floorplan(
                            foot_x,
                            foot_y
                        )

                        map_pos = (
                            map_x,
                            map_y
                        )

                    except Exception:

                        map_pos = None

                filtered.append({
                    "idx": i,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "foot": (
                        foot_x,
                        foot_y
                    ),
                    "tid": tid,
                    "frame_index": int(frame_index),
                    "camera_generation": int(
                        cam_data.get("tracker_generation", 0)
                    ),
                    "local_track_confirmed": (
                        local_track_confirmed
                    ),
                    "conf": conf_val,
                    "box_wh": box_wh,
                    "emb": emb,
                    **quality_meta,
                    "map_pos": map_pos,
                    "center": bbox_center(
                        (x1, y1, x2, y2)
                    ),
                })

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
                ]
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

                if res is None:
                    continue

                x1, y1, x2, y2 = item["box"]

                foot_x, foot_y = item["foot"]

                tid = item["tid"]

                gid = res["gid"]

                match_score = res["score"]

                match_source = res["source"]

                label = f"GID {gid}"

                if item["conf"] is not None:
                    label += f" {item['conf']:.2f}"

                if REID_DEBUG:
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

                    global_map.update_object(
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
    cam_data["active_local_tracks"] = (
        active_local_track_ids
    )
    cam_data["tracker_last_frame_index"] = int(
        frame_index
    )
    cam_data["tracker_last_event_time"] = event_ts
    cam_data["downstream_timing"] = {
        "frame_index": int(frame_index),
        "event_time": event_ts,
        "detection_tracking_ms": float(tracking_duration_ms),
        "reid_feature_ms": float(reid_feature_duration_ms),
        "coordinator_submit_ms": float(
            coordinator_submit_duration_ms
        ),
        "total_downstream_ms": float(
            (time.perf_counter() - downstream_started) * 1000.0
        ),
        "observation_count": int(reid_observation_count),
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

def generate_global_map():

    while app.is_running:

        canvas = (
            global_map.draw_map()
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

        "reid":
            dict(
                REID_RUNTIME_STATUS
            ),

        "global_assignment":
            global_assignment_coordinator.status()

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
    file: UploadFile = File(...)
):

    try:

        contents = (
            await file.read()
        )


        with open(
            FLOORPLAN_PATH,
            "wb"
        ) as f:

            f.write(
                contents
            )


        global_map.load_floorplan()


        return json_response(
            True,
            "อัปโหลดแผนผังสำเร็จ"
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

    loop_video: bool = Form(True),

    time_offset_sec: float = Form(0.0)

):

    try:

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

@app.get(
    "/api/get_floorplan"
)
async def get_floorplan():

    img_b64 = (
        image_file_to_base64(
            FLOORPLAN_PATH
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

        "image_base64":
            img_b64

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
async def global_map_feed():

    return StreamingResponse(

        generate_global_map(),

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

    dst_pts: str = Form(...)

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


        return json_response(

            True,

            "Saved calibration",

            {

                "camera":
                    cam_name,

                "src_pts":
                    parsed_src,

                "dst_pts":
                    parsed_dst

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
        return {"topology": topology_config}


@app.put("/api/topology")
async def put_topology(request: Request):
    global topology_config
    payload = await request.json()
    validate_topology_config(payload)
    normalized = {
        "version": int(payload.get("version", 1)),
        "enforce": bool(payload.get("enforce", False)),
        "transitions": payload.get("transitions", []),
    }
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

@app.post("/api/shutdown")
async def shutdown_system():

    stop_background_workers()

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

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8899

    )
