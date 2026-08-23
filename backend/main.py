import cv2
import os
import threading
import time
import json
import base64
import numpy as np

try:
    import torch
    import torchreid
    from torchreid.utils import FeatureExtractor
except ImportError:
    torch = None
    torchreid = None
    FeatureExtractor = None

import logging
import signal
from urllib.parse import urlparse
from collections import deque
from scipy.optimize import linear_sum_assignment

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


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

model = YOLO(YOLO_MODEL_PATH)

app.is_running = True


# ============================================================
# GLOBAL VARIABLES
# ============================================================

cameras_lock = threading.Lock()
cameras = {}

MAX_UPLOAD_SIZE = 500 * 1024 * 1024

FLOORPLAN_PATH = "static/floorplan.png"
UPLOAD_DIR = "static/uploads"

os.makedirs("static", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# RE-ID CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# OSNet
# ------------------------------------------------------------

USE_OSNET = True

# ใช้ OSNet x0.25
REID_MODEL_NAME = "osnet_x0_25"

# Weight ที่ train กับ Market1501
REID_MODEL_PATH = "osnet_x0_25_market1501.pt"

# OSNet รับภาพขนาด H x W
REID_INPUT_H = 256
REID_INPUT_W = 128


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
    cap = cv2.VideoCapture(camera_url)

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


def validate_camera_url(url: str):
    if url.isdigit():
        return int(url)

    parsed = urlparse(url)

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

    return url


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


def safe_json_value(value):

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


# ============================================================
# LIGHTWEIGHT FALLBACK
# ============================================================

class LightweightAppearanceFeatureExtractor:

    def __init__(self):
        self.name = "lightweight"

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

class OSNetFeatureExtractor:

    def __init__(self):

        if torch is None:
            raise RuntimeError(
                "ไม่พบ PyTorch"
            )

        if torchreid is None:
            raise RuntimeError(
                "ไม่พบ torchreid"
            )

        if FeatureExtractor is None:
            raise RuntimeError(
                "ไม่พบ torchreid.utils.FeatureExtractor"
            )

        if not os.path.exists(REID_MODEL_PATH):
            raise FileNotFoundError(
                f"ไม่พบ ReID weight: {REID_MODEL_PATH}"
            )

        self.name = (
            f"{REID_MODEL_NAME}_Market1501"
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "[ReID] Loading OSNet..."
        )

        print(
            f"[ReID] Model: {REID_MODEL_NAME}"
        )

        print(
            f"[ReID] Weight: {REID_MODEL_PATH}"
        )

        print(
            f"[ReID] Device: {self.device}"
        )

        # ----------------------------------------------------
        # ใช้ FeatureExtractor โดยตรง
        # และระบุ Market1501 weight
        # ----------------------------------------------------

        self.extractor = FeatureExtractor(
            model_name=REID_MODEL_NAME,
            model_path=REID_MODEL_PATH,
            image_size=(
                REID_INPUT_H,
                REID_INPUT_W
            ),
            device=self.device
        )

        print(
            "[ReID] OSNet Market1501 loaded successfully"
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

            # FeatureExtractor รับ BGR image จาก OpenCV ได้
            features = self.extractor(
                [person_crop]
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

    if USE_OSNET:

        try:

            extractor = OSNetFeatureExtractor()

            print(
                f"[ReID] Using: {extractor.name}"
            )

            return extractor

        except Exception as e:

            print(
                "[ReID] OSNet unavailable:"
            )

            print(
                f"[ReID] {e}"
            )

            print(
                "[ReID] Fallback to lightweight extractor"
            )

    extractor = (
        LightweightAppearanceFeatureExtractor()
    )

    print(
        f"[ReID] Using fallback: {extractor.name}"
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

    def __init__(self):

        self.next_global_id = 1

        # GID -> identity information
        self.identities = {}

        # (camera, local_track_id) -> GID
        self.local_to_global = {}

        # Occlusion
        self.occlusion_hold = {}

        # Recent same camera
        self.recent_same_cam = []

        # Recent cross camera
        self.recent_cross_cam = []

        self.lock = threading.Lock()

    # ==================

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

    def cleanup(self):

        now = time.time()

        # ----------------------------------------------------
        # Global identities
        # ----------------------------------------------------

        stale_global_ids = []

        for gid, info in self.identities.items():

            if (
                now - info["last_seen"]
                >
                REID_MAX_IDLE_SEC
            ):

                stale_global_ids.append(
                    gid
                )

        for gid in stale_global_ids:

            self.identities.pop(
                gid,
                None
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
        history
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

        horizon = min(
            max(
                time.time() - t2,
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
        prev_assignments
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
                history
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

    def _can_match(
        self,
        identity,
        cam_name,
        now_ts,
        map_pos,
        box_wh
    ):

        dt = (
            now_ts
            -
            identity["last_seen"]
        )

        if dt > REID_MAX_IDLE_SEC:
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

        # ====================================================
        # CROSS CAMERA
        # ====================================================

        if cross_camera:

            appearance_score = max(
                0.0,
                appearance
            )

            map_score = max(
                0.0,
                map_s
            )

            time_score = max(
                0.0,
                time_s
            )

            total = (
                0.75 * appearance_score
                +
                0.15 * map_score
                +
                0.10 * time_score
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
                    prev_assignments
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
                app_w * appearance
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
            score
            >=
            ASSIGN_SAME_CAM_SCORE_THRESHOLD
        ):

            return True

        return (
            score
            >=
            ASSIGN_SAME_CAM_SCORE_THRESHOLD
        )


    # ========================================================
    # UPDATE IDENTITY
    # ========================================================

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

        # ----------------------------------------------------
        # Slow EMA
        # ----------------------------------------------------

        old_emb = identity.get(
            "embedding"
        )

        if old_emb is not None:

            identity["embedding"] = (
                l2_normalize(
                    REID_EMBED_UPDATE_ALPHA
                    *
                    old_emb
                    +
                    (
                        1.0
                        -
                        REID_EMBED_UPDATE_ALPHA
                    )
                    *
                    emb
                )
            )

        else:

            identity["embedding"] = (
                l2_normalize(emb)
            )


        # ----------------------------------------------------
        # Gallery
        # ----------------------------------------------------

        gallery = identity.setdefault(
            "gallery",
            []
        )

        gallery.append(
            l2_normalize(emb)
        )

        if len(gallery) > REID_GALLERY_SIZE:

            identity["gallery"] = (
                gallery[
                    -REID_GALLERY_SIZE:
                ]
            )


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

                "gallery": [
                    l2_normalize(emb)
                ],

                "last_cam":
                    cam_name,

                "last_seen":
                    now_ts,

                "last_map_pos":
                    map_pos,

                "box_wh":
                    box_wh,

                "last_score":
                    float(score)

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


        if source == "cross-camera":

            self._remember_cross_cam(
                gid,
                cam_name,
                emb,
                map_pos,
                now_ts
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
        prev_assignments=None
    ):

        now_ts = time.time()

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

            self.cleanup()

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
    GlobalIdentityManager()
)

global_map = GlobalMapManager(
    trail_len=1,
    timeout_sec=0.7
)

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

    def register_video(self, cam_name, video_path, loop_video=True):
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

        with self.lock:
            self.videos[cam_name] = {
                "cap": cap,
                "fps": float(fps),
                "total_frames": total_frames,
                "loop_video": bool(loop_video),
                "frame_index": 0,
            }

            self.frames[cam_name] = None
            self.frame_indices[cam_name] = 0
            # Wait for an explicit playback command so multiple uploaded
            # clips can start on the same worker iteration.
            self.running[cam_name] = False

        logger.info(
            f"[SYNC] Registered {cam_name} | "
            f"FPS={fps:.2f} | Frames={total_frames}"
        )

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

                    else:
                        self.running[cam_name] = False
                        continue

                data["frame_index"] += 1

                frame_index = data["frame_index"]

                self.frames[cam_name] = frame.copy()
                self.frame_indices[cam_name] = frame_index

                result[cam_name] = {
                    "frame": frame.copy(),
                    "frame_index": frame_index,
                    "fps": data["fps"],
                }

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

def process_camera_frame(cam_name, frame, frame_index):
    """
    ประมวลผล Frame ของ Camera หนึ่งตัว

    YOLO
    -> BoT-SORT
    -> OSNet ReID
    -> Global ID
    -> Homography
    -> Global Map
    """

    cam_data = cameras.get(cam_name)

    if cam_data is None:
        return frame

    annotated_frame = frame.copy()

    # --------------------------------------------------------
    # YOLO + BoT-SORT
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.55,
        tracker="botsort.yaml",
        verbose=False
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
                    else i
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

                emb = extract_person_embedding(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2
                )

                if emb is None:
                    continue

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
                    "conf": conf_val,
                    "box_wh": box_wh,
                    "emb": emb,
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

                item["forced_gid"] = (
                    forced_gid_map.get(a)
                    if a in overlap_indices
                    else None
                )

            # ------------------------------------------------
            # GLOBAL ID
            # ------------------------------------------------

            assignment_results = (
                global_identity_manager.assign_batch(
                    cam_name,
                    filtered,
                    prev_assignments=prev_assignments
                )
                if filtered
                else []
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
                    label += f" | L{tid}"

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
                    "ts": time.time(),
                })

    # --------------------------------------------------------
    # Save previous assignments
    # --------------------------------------------------------

    cameras[cam_name]["prev_assignments"] = (
        prev_assignments + frame_assignments
    )[-60:]

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

        for cam_name, data in frames_data.items():

            frame = data["frame"]

            frame_index = data["frame_index"]

            try:

                # เก็บ frame ล่าสุด
                cameras[cam_name]["last_frame"] = (
                    frame.copy()
                )

                # YOLO + BoT + ReID + Homography
                annotated_frame = process_camera_frame(
                    cam_name,
                    frame,
                    frame_index
                )

                # ========================================
                # Encode JPEG
                # ========================================

                ok, buffer = cv2.imencode(
                    ".jpg",
                    annotated_frame,
                    [
                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),
                        80
                    ]
                )

                if not ok:
                    continue

                frame_bytes = buffer.tobytes()

                # ========================================
                # Save processed frame
                # ========================================

                with video_worker_lock:

                    processed_frames[cam_name] = (
                        frame_bytes
                    )

                    if cam_name not in processed_frame_locks:
                        processed_frame_locks[cam_name] = (
                            threading.Condition()
                        )

                    condition = (
                        processed_frame_locks[cam_name]
                    )

                    with condition:
                        condition.notify_all()

            except Exception as e:

                logger.error(
                    f"[SYNC] Error processing "
                    f"{cam_name}: {e}",
                    exc_info=True
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

    multi_video_manager.release_all()

    logger.info(
        "[SYNC] Worker stopped"
    )

def generate_frames(cam_name: str):

    if cam_name not in cameras:
        return

    # ถ้า Worker ยังไม่ทำงาน ให้เริ่ม
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


    for name, cam in cameras.items():

        cams_data[name] = {

            "url":
                cam["url"],

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
                )

        }


    return JSONResponse({

        "cameras":
            cams_data,

        "floorplan_exists":
            floorplan_exists

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

    loop_video: bool = Form(True)

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
                    []

            }
        try:

            multi_video_manager.register_video(
                name,
                save_path,
                loop_video
            )

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
            validate_camera_url(
                url
            )
        )


        with cameras_lock:

            cameras[name] = {

                "url":
                    final_url,

                "source_type":
                    "camera",

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

                "prev_assignments":
                    []

            }


        logger.info(

            f"Camera added: "
            f"{name} -> {final_url}"

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

        if cam_name in cameras:

            cam = cameras[
                cam_name
            ]


            if cam.get("source_type") == "video":

                # เอาออกจาก Multi-Camera Worker
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


            del cameras[
                cam_name
            ]


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

    cam = cameras.get(
        cam_name
    )


    if not cam:

        return json_response(

            False,

            "Camera not found",

            status_code=404

        )


    frame = open_camera_once(
        cam["url"]
    )


    if frame is None:

        return json_response(

            False,

            "Cannot capture frame",

            status_code=500

        )


    cam[
        "last_frame"
    ] = frame.copy()


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

@app.post("/api/shutdown")
async def shutdown_system():

    app.is_running = False

    stop_multi_camera_worker()

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
