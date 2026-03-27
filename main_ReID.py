import cv2
import os
import threading
import time
import json
import base64
import numpy as np
import torch
import torchreid
from collections import deque
from scipy.optimize import linear_sum_assignment

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
    

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = YOLO("yolov8s.pt")
app.is_running = True

# -----------------------------
# Global vars
# -----------------------------
cameras = {}

FLOORPLAN_PATH = "static/floorplan.png"
UPLOAD_DIR = "static/uploads"

os.makedirs("static", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Re-ID config
# -----------------------------
# หมายเหตุ:
# เวอร์ชันนี้รองรับ OSNet ผ่าน torchreid
# ถ้ายังไม่ได้ติดตั้ง torchreid จะ fallback ไปใช้ lightweight feature extractor อัตโนมัติ
USE_OSNET = True
OSNET_MODEL_NAME = "osnet_x1_0"
OSNET_INPUT_W = 128
OSNET_INPUT_H = 256
REID_MAX_IDLE_SEC = 25.0
REID_SIM_THRESHOLD = 0.62
REID_STRONG_SIM_THRESHOLD = 0.75
REID_SAME_CAM_RELINK_THRESHOLD = 0.50
REID_MAP_GATE_PX = 260.0
REID_SIZE_GATE_RATIO = 0.35
REID_EMBED_UPDATE_ALPHA = 0.90
REID_MIN_CROP_SIZE = 24
REID_GALLERY_SIZE = 6
REID_DEBUG = True
REID_RECENT_SAME_CAM_SEC = 12.0
REID_RECENT_SAME_CAM_THRESHOLD = 0.36
REID_CROP_SIDE_MARGIN = 0.18
REID_CROP_TOP_MARGIN = 0.10
REID_CROP_BOTTOM_MARGIN = 0.18
OCCLUSION_IOU_THRESHOLD = 0.35
OCCLUSION_HOLD_SEC = 0.8
OCCLUSION_PREV_IOU_THRESHOLD = 0.20
OCCLUSION_CENTER_DIST_PX = 80.0
ASSIGN_SCORE_THRESHOLD = 0.28
ASSIGN_STRONG_APPEARANCE_THRESHOLD = 0.82
ASSIGN_OVERLAP_FREEZE_BONUS = 0.22
ASSIGN_SAME_CAM_BONUS = 0.10
ASSIGN_MOTION_WEIGHT = 0.42
ASSIGN_APPEARANCE_WEIGHT = 0.42
ASSIGN_MAP_WEIGHT = 0.10
ASSIGN_TIME_WEIGHT = 0.06


# -----------------------------
# Utilities
# -----------------------------
def frame_to_base64(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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



def safe_filename(filename: str):
    keepchars = (" ", ".", "_", "-")
    cleaned = "".join(c for c in filename if c.isalnum() or c in keepchars).strip()
    return cleaned or f"video_{int(time.time())}.mp4"



def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2



def l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm



def cosine_similarity(a, b):
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


# -----------------------------
# Appearance extractor
# -----------------------------
class LightweightAppearanceFeatureExtractor:
    def __init__(self):
        self.name = "lightweight"

    def _hsv_hist(self, img_bgr, h_bins=12, s_bins=4, v_bins=4):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def _region_hist(self, img_bgr):
        h, w = img_bgr.shape[:2]
        upper = img_bgr[: max(1, int(h * 0.45)), :]
        lower = img_bgr[max(0, int(h * 0.45)): , :]
        upper_hist = self._hsv_hist(upper)
        lower_hist = self._hsv_hist(lower)
        return np.concatenate([upper_hist, lower_hist]).astype(np.float32)

    def _shape_feature(self, img_bgr):
        h, w = img_bgr.shape[:2]
        aspect = np.array([w / max(h, 1)], dtype=np.float32)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 32))
        gray = gray.astype(np.float32) / 255.0
        coarse = gray.flatten()
        return np.concatenate([aspect, coarse]).astype(np.float32)

    def extract(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        if h < REID_MIN_CROP_SIZE or w < REID_MIN_CROP_SIZE:
            return None

        crop = cv2.resize(person_crop, (64, 128))
        hist_feat = self._region_hist(crop)
        shape_feat = self._shape_feature(crop)
        feat = np.concatenate([hist_feat, shape_feat]).astype(np.float32)
        return l2_normalize(feat)


class OSNetFeatureExtractor:
    def __init__(self, model_name=OSNET_MODEL_NAME):
        if torch is None or torchreid is None:
            raise RuntimeError("torch หรือ torchreid ยังไม่พร้อม")

        self.name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=True,
            use_gpu=(self.device == "cuda")
        )
        self.model.eval()
        self.model.to(self.device)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _preprocess(self, person_crop):
        img = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (OSNET_INPUT_W, OSNET_INPUT_H))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img).float()

    def extract(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        if h < REID_MIN_CROP_SIZE or w < REID_MIN_CROP_SIZE:
            return None

        tensor = self._preprocess(person_crop).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        feat = feat.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return l2_normalize(feat)


def build_feature_extractor():
    if USE_OSNET:
        try:
            extractor = OSNetFeatureExtractor()
            print(f"[ReID] Using OSNet extractor: {extractor.name} on {extractor.device}")
            return extractor
        except Exception as e:
            print(f"[ReID] OSNet unavailable, fallback to lightweight extractor: {e}")
    extractor = LightweightAppearanceFeatureExtractor()
    print(f"[ReID] Using fallback extractor: {extractor.name}")
    return extractor


# -----------------------------
# Camera Processor
# -----------------------------
class CameraProcessor:
    def __init__(self, cam_id, src_pts, dst_pts):
        self.cam_id = cam_id
        self.src_pts = np.array(src_pts, dtype=np.float32)
        self.dst_pts = np.array(dst_pts, dtype=np.float32)

        if self.src_pts.shape != (4, 2) or self.dst_pts.shape != (4, 2):
            raise ValueError("src_pts และ dst_pts ต้องมี 4 จุด")

        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)
        if self.H is None:
            raise ValueError("คำนวณ Homography ไม่สำเร็จ")

    def to_floorplan(self, px, py):
        pt = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        map_x, map_y = transformed[0][0]
        return int(map_x), int(map_y)

    def draw_calibration_polygon(self, frame):
        pts = self.src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2)

        for i, p in enumerate(self.src_pts.astype(np.int32)):
            x, y = int(p[0]), int(p[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"P{i+1}", (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame


# -----------------------------
# Cross-camera Global ID Manager
# -----------------------------
class GlobalIdentityManager:
    def __init__(self):
        self.next_global_id = 1
        self.identities = {}
        self.local_to_global = {}
        self.occlusion_hold = {}
        self.recent_same_cam = []
        self.lock = threading.Lock()

    def _gallery_similarity(self, emb, identity):
        scores = []
        if identity.get("embedding") is not None:
            scores.append(cosine_similarity(emb, identity["embedding"]))
        for g in identity.get("gallery", []):
            scores.append(cosine_similarity(emb, g))
        if not scores:
            return -1.0
        scores.sort(reverse=True)
        topk = scores[: min(3, len(scores))]
        return float(sum(topk) / len(topk))

    def cleanup(self):
        now = time.time()
        stale_global_ids = []
        for gid, info in self.identities.items():
            if now - info["last_seen"] > max(REID_MAX_IDLE_SEC, 15.0):
                stale_global_ids.append(gid)
        for gid in stale_global_ids:
            self.identities.pop(gid, None)

        stale_local_keys = []
        for key, data in self.local_to_global.items():
            if now - data["last_seen"] > max(REID_MAX_IDLE_SEC, 15.0):
                stale_local_keys.append(key)
        for key in stale_local_keys:
            self.local_to_global.pop(key, None)

        stale_hold_keys = []
        for key, hold in self.occlusion_hold.items():
            if now > hold.get("until_ts", 0):
                stale_hold_keys.append(key)
        for key in stale_hold_keys:
            self.occlusion_hold.pop(key, None)

        kept_recent = []
        for item in self.recent_same_cam:
            if now - item.get("ts", 0) <= REID_RECENT_SAME_CAM_SEC and item.get("gid") in self.identities:
                kept_recent.append(item)
        self.recent_same_cam = kept_recent[-200:]

    def _remember_recent_same_cam(self, gid, cam_name, emb, map_pos, box_wh, now_ts):
        self.recent_same_cam.append({
            "gid": gid,
            "cam_name": cam_name,
            "embedding": l2_normalize(emb),
            "map_pos": map_pos,
            "box_wh": box_wh,
            "ts": now_ts,
        })
        if len(self.recent_same_cam) > 200:
            self.recent_same_cam = self.recent_same_cam[-200:]

    def _find_recent_same_cam_match(self, cam_name, emb, map_pos, box_wh, now_ts, used_gids=None):
        best_gid = None
        best_score = -999.0
        for item in reversed(self.recent_same_cam):
            if item.get("cam_name") != cam_name:
                continue
            dt = now_ts - item.get("ts", 0)
            if dt > REID_RECENT_SAME_CAM_SEC:
                continue
            gid = item.get("gid")
            if gid not in self.identities:
                continue
            if used_gids is not None and gid in used_gids:
                continue
            if not self._size_ratio_ok(box_wh, item.get("box_wh")):
                continue
            score = cosine_similarity(emb, item.get("embedding"))
            prev_pos = item.get("map_pos")
            if map_pos is not None and prev_pos is not None:
                dist = self._map_distance(map_pos, prev_pos)
                if dist is not None:
                    score -= min(dist / max(REID_MAP_GATE_PX * 1.5, 1.0), 1.0) * 0.10
            score -= min(dt / max(REID_RECENT_SAME_CAM_SEC, 1e-6), 1.0) * 0.08
            if score > best_score and score >= REID_RECENT_SAME_CAM_THRESHOLD:
                best_gid = gid
                best_score = score
        return best_gid, float(best_score)

    def _map_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return None
        return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))

    def _size_ratio_ok(self, box_wh, ref_wh):
        if box_wh is None or ref_wh is None:
            return True
        bw, bh = box_wh
        rw, rh = ref_wh
        if min(bw, bh, rw, rh) <= 0:
            return True
        wr = min(bw, rw) / max(bw, rw)
        hr = min(bh, rh) / max(bh, rh)
        return wr >= REID_SIZE_GATE_RATIO and hr >= REID_SIZE_GATE_RATIO

    def _can_match(self, identity, cam_name, now_ts, map_pos, box_wh):
        dt = now_ts - identity["last_seen"]
        if dt > REID_MAX_IDLE_SEC:
            return False
        if not self._size_ratio_ok(box_wh, identity.get("box_wh")):
            return False
        prev_pos = identity.get("last_map_pos")
        if map_pos is not None and prev_pos is not None:
            dist = self._map_distance(map_pos, prev_pos)
            if dist is not None:
                if identity.get("last_cam") == cam_name:
                    if dist > REID_MAP_GATE_PX * 1.5:
                        return False
                else:
                    if dist > REID_MAP_GATE_PX * 2.0:
                        return False
        return True

    def _recent_history_for_gid(self, cam_name, gid, prev_assignments, limit=3):
        if not prev_assignments:
            return []
        hist = []
        for item in reversed(prev_assignments):
            if item.get("cam_name") == cam_name and item.get("gid") == gid:
                hist.append(item)
                if len(hist) >= limit:
                    break
        return list(reversed(hist))

    def _predict_center(self, history):
        if not history:
            return None
        if len(history) == 1:
            return history[-1].get("center")
        p1 = history[-2].get("center")
        p2 = history[-1].get("center")
        t1 = history[-2].get("ts", 0.0)
        t2 = history[-1].get("ts", 0.0)
        if p1 is None or p2 is None:
            return p2 or p1
        dt = max(t2 - t1, 1e-3)
        vx = (p2[0] - p1[0]) / dt
        vy = (p2[1] - p1[1]) / dt
        horizon = min(max(time.time() - t2, 0.0), 0.25)
        return (p2[0] + vx * horizon, p2[1] + vy * horizon)

    def _motion_score(self, cam_name, gid, det_box, prev_assignments):
        history = self._recent_history_for_gid(cam_name, gid, prev_assignments, limit=3)
        if not history:
            return 0.50
        pred_center = self._predict_center(history)
        if pred_center is None:
            return 0.50
        det_center = bbox_center(det_box)
        dist = float(np.hypot(det_center[0] - pred_center[0], det_center[1] - pred_center[1]))
        last_box = history[-1].get("box", det_box)
        lx1, ly1, lx2, ly2 = last_box
        scale = max(40.0, np.hypot(lx2 - lx1, ly2 - ly1) * 0.75)
        score = 1.0 - min(dist / scale, 1.6)
        return float(max(-0.6, min(1.0, score)))

    def _time_score(self, identity, now_ts):
        dt = max(0.0, now_ts - identity["last_seen"])
        return float(max(0.0, 1.0 - min(dt / max(REID_MAX_IDLE_SEC, 1e-3), 1.0)))

    def _map_score(self, identity, map_pos):
        prev_pos = identity.get("last_map_pos")
        if map_pos is None or prev_pos is None:
            return 0.50
        dist = self._map_distance(map_pos, prev_pos)
        if dist is None:
            return 0.50
        score = 1.0 - min(dist / max(REID_MAP_GATE_PX * 1.8, 1.0), 1.4)
        return float(max(-0.4, min(1.0, score)))

    def _pair_score(self, gid, identity, cam_name, det, now_ts, prev_assignments):
        appearance = self._gallery_similarity(det["emb"], identity)
        motion = self._motion_score(cam_name, gid, det["box"], prev_assignments)
        map_s = self._map_score(identity, det.get("map_pos"))
        time_s = self._time_score(identity, now_ts)

        app_w = ASSIGN_APPEARANCE_WEIGHT
        motion_w = ASSIGN_MOTION_WEIGHT
        if det.get("overlap", False):
            app_w -= 0.14
            motion_w += 0.14

        total = app_w * appearance + motion_w * motion + ASSIGN_MAP_WEIGHT * map_s + ASSIGN_TIME_WEIGHT * time_s

        if identity.get("last_cam") == cam_name:
            total += ASSIGN_SAME_CAM_BONUS
            if now_ts - identity.get("last_seen", 0) <= 1.5:
                total += 0.05

        if det.get("forced_gid") == gid and det.get("overlap", False):
            total += ASSIGN_OVERLAP_FREEZE_BONUS

        return {
            "gid": gid,
            "score": float(total),
            "appearance": float(appearance),
            "motion": float(motion),
            "map": float(map_s),
            "time": float(time_s),
        }

    def _accept_match(self, pair, identity, cam_name, det):
        if pair is None:
            return False
        if pair["appearance"] >= ASSIGN_STRONG_APPEARANCE_THRESHOLD:
            return True
        if identity.get("last_cam") == cam_name and pair["score"] >= max(ASSIGN_SCORE_THRESHOLD - 0.06, 0.18):
            return True
        if det.get("overlap", False) and det.get("forced_gid") == pair["gid"] and pair["score"] >= max(ASSIGN_SCORE_THRESHOLD - 0.08, 0.16):
            return True
        return pair["score"] >= ASSIGN_SCORE_THRESHOLD

    def _commit_assignment(self, gid, cam_name, local_id, emb, map_pos, box_wh, now_ts, score, source):
        local_key = (cam_name, int(local_id))
        if gid in self.identities:
            self._update_identity(gid, cam_name, emb, map_pos, box_wh, now_ts, last_score=score)
        else:
            self.identities[gid] = {
                "embedding": l2_normalize(emb),
                "gallery": [l2_normalize(emb)],
                "last_cam": cam_name,
                "last_seen": now_ts,
                "last_map_pos": map_pos,
                "box_wh": box_wh,
                "last_score": float(score),
            }
        self.local_to_global[local_key] = {"gid": gid, "last_seen": now_ts}
        self._remember_recent_same_cam(gid, cam_name, emb, map_pos, box_wh, now_ts)
        return {"gid": gid, "score": float(score), "source": source}

    def _new_identity(self, cam_name, local_id, emb, map_pos, box_wh, now_ts):
        gid = self.next_global_id
        self.next_global_id += 1
        return self._commit_assignment(gid, cam_name, local_id, emb, map_pos, box_wh, now_ts, 1.0, "new")

    def assign_batch(self, cam_name, detections, prev_assignments=None):
        now_ts = time.time()
        prev_assignments = prev_assignments or []
        results = [None] * len(detections)

        with self.lock:
            self.cleanup()
            used_gids = set()
            pending_indices = []

            # 1) lock by local track / hold / forced gid first
            for idx, det in enumerate(detections):
                local_key = (cam_name, int(det["tid"]))

                hold = self.occlusion_hold.get(local_key)
                if hold is not None and now_ts <= hold.get("until_ts", 0):
                    gid = hold.get("gid")
                    if gid in self.identities and gid not in used_gids:
                        results[idx] = self._commit_assignment(
                            gid, cam_name, det["tid"], det["emb"], det.get("map_pos"), det.get("box_wh"),
                            now_ts, hold.get("score", 1.0), "occlusion-hold"
                        )
                        used_gids.add(gid)
                        continue

                existing = self.local_to_global.get(local_key)
                if existing is not None:
                    gid = existing.get("gid")
                    if gid in self.identities and gid not in used_gids:
                        results[idx] = self._commit_assignment(
                            gid, cam_name, det["tid"], det["emb"], det.get("map_pos"), det.get("box_wh"),
                            now_ts, self.identities[gid].get("last_score", 1.0), "local-track"
                        )
                        used_gids.add(gid)
                        continue
                    self.local_to_global.pop(local_key, None)

                if det.get("overlap", False) and det.get("forced_gid") in self.identities and det.get("forced_gid") not in used_gids:
                    gid = det.get("forced_gid")
                    results[idx] = self._commit_assignment(
                        gid, cam_name, det["tid"], det["emb"], det.get("map_pos"), det.get("box_wh"),
                        now_ts, 1.0, "occlusion-forced"
                    )
                    used_gids.add(gid)
                    continue

                pending_indices.append(idx)

            # 2) batch assignment for remaining detections vs existing gids
            candidate_gids = []
            for gid, identity in self.identities.items():
                if gid in used_gids:
                    continue
                reusable = False
                for idx in pending_indices:
                    det = detections[idx]
                    if self._can_match(identity, cam_name, now_ts, det.get("map_pos"), det.get("box_wh")):
                        reusable = True
                        break
                if reusable:
                    candidate_gids.append(gid)

            pair_cache = {}
            if pending_indices and candidate_gids:
                score_matrix = np.full((len(pending_indices), len(candidate_gids)), -1e6, dtype=np.float32)
                for r, idx in enumerate(pending_indices):
                    det = detections[idx]
                    for c, gid in enumerate(candidate_gids):
                        pair = self._pair_score(gid, self.identities[gid], cam_name, det, now_ts, prev_assignments)
                        pair_cache[(idx, gid)] = pair
                        score_matrix[r, c] = pair["score"]

                row_ind, col_ind = linear_sum_assignment(-score_matrix)
                matched_rows = set()
                for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                    idx = pending_indices[r]
                    gid = candidate_gids[c]
                    pair = pair_cache[(idx, gid)]
                    det = detections[idx]
                    identity = self.identities.get(gid)
                    if identity is None:
                        continue
                    if not self._accept_match(pair, identity, cam_name, det):
                        continue
                    if gid in used_gids:
                        continue
                    results[idx] = self._commit_assignment(
                        gid, cam_name, det["tid"], det["emb"], det.get("map_pos"), det.get("box_wh"),
                        now_ts, pair["score"], "batch-match"
                    )
                    used_gids.add(gid)
                    matched_rows.add(idx)

                pending_indices = [idx for idx in pending_indices if idx not in matched_rows]

            # 3) recent same-camera fallback
            still_pending = []
            for idx in pending_indices:
                det = detections[idx]
                gid, recent_score = self._find_recent_same_cam_match(
                    cam_name, det["emb"], det.get("map_pos"), det.get("box_wh"), now_ts, used_gids=used_gids
                )
                if gid is not None and gid in self.identities:
                    results[idx] = self._commit_assignment(
                        gid, cam_name, det["tid"], det["emb"], det.get("map_pos"), det.get("box_wh"),
                        now_ts, recent_score, "same-cam-cache"
                    )
                    used_gids.add(gid)
                else:
                    still_pending.append(idx)

            # 4) create new ids
            for idx in still_pending:
                det = detections[idx]
                results[idx] = self._new_identity(
                    cam_name, det["tid"], det["emb"], det.get("map_pos"), det.get("box_wh"), now_ts
                )

            # refresh hold for overlap
            for idx, det in enumerate(detections):
                if results[idx] is None:
                    continue
                if det.get("overlap", False):
                    local_key = (cam_name, int(det["tid"]))
                    self.occlusion_hold[local_key] = {
                        "gid": results[idx]["gid"],
                        "until_ts": now_ts + OCCLUSION_HOLD_SEC,
                        "score": float(results[idx]["score"]),
                    }

        return results

    def resolve_identity(self, cam_name, local_id, emb, map_pos=None, box_wh=None, forbidden_gids=None, forced_gid=None, allow_new=True):
        det = {
            "tid": local_id,
            "emb": emb,
            "map_pos": map_pos,
            "box_wh": box_wh,
            "box": (0, 0, box_wh[0] if box_wh else 1, box_wh[1] if box_wh else 1),
            "overlap": forced_gid is not None,
            "forced_gid": forced_gid,
        }
        result = self.assign_batch(cam_name, [det], prev_assignments=[])[0]
        return result["gid"], result["score"], result["source"]

    def _update_identity(self, gid, cam_name, emb, map_pos, box_wh, now_ts, last_score=None):
        identity = self.identities[gid]
        identity["embedding"] = l2_normalize(
            REID_EMBED_UPDATE_ALPHA * identity["embedding"] + (1.0 - REID_EMBED_UPDATE_ALPHA) * emb
        )
        gallery = identity.setdefault("gallery", [])
        gallery.append(l2_normalize(emb))
        if len(gallery) > REID_GALLERY_SIZE:
            identity["gallery"] = gallery[-REID_GALLERY_SIZE:]
        identity["last_cam"] = cam_name
        identity["last_seen"] = now_ts
        identity["last_map_pos"] = map_pos
        identity["box_wh"] = box_wh
        if last_score is not None:
            identity["last_score"] = float(last_score)


# -----------------------------
# Global Map Manager
# -----------------------------
class GlobalMapManager:
    def __init__(self, trail_len=50, timeout_sec=2.0):
        self.trail_len = trail_len
        self.timeout_sec = timeout_sec

        self.base_map = None
        self.objects = {}
        self.tracks = {}
        self.last_seen = {}
        self.lock = threading.Lock()

        self.load_floorplan()

    def load_floorplan(self):
        if os.path.exists(FLOORPLAN_PATH):
            img = cv2.imread(FLOORPLAN_PATH)
            if img is not None:
                self.base_map = img
                return

        self.base_map = np.zeros((600, 900, 3), dtype=np.uint8)
        cv2.putText(
            self.base_map,
            "No Floorplan Uploaded",
            (220, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )

    def update_object(self, global_id, map_x, map_y):
        if self.base_map is None:
            return

        with self.lock:
            h, w = self.base_map.shape[:2]
            map_x = max(0, min(int(map_x), w - 1))
            map_y = max(0, min(int(map_y), h - 1))

            self.objects[global_id] = (map_x, map_y)
            self.last_seen[global_id] = time.time()

            if global_id not in self.tracks:
                self.tracks[global_id] = deque(maxlen=self.trail_len)

            self.tracks[global_id].append((map_x, map_y))

    def cleanup_stale_objects(self):
        now = time.time()
        stale_ids = [
            gid for gid, ts in self.last_seen.items()
            if now - ts > self.timeout_sec
        ]
        for gid in stale_ids:
            self.last_seen.pop(gid, None)
            self.objects.pop(gid, None)
            self.tracks.pop(gid, None)

    def draw_map(self):
        with self.lock:
            self.cleanup_stale_objects()
            canvas = self.base_map.copy()

            for gid, (mx, my) in self.objects.items():
                cv2.circle(canvas, (mx, my), 8, (0, 255, 0), -1)
                cv2.putText(canvas, f"ID {gid}", (mx + 10, my - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return canvas


appearance_extractor = build_feature_extractor()
global_identity_manager = GlobalIdentityManager()
global_map = GlobalMapManager(trail_len=1, timeout_sec=0.7)


# -----------------------------
# Streaming
# -----------------------------
def point_in_polygon(point, polygon_pts):
    if polygon_pts is None:
        return True
    poly = np.array(polygon_pts, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0



def extract_person_embedding(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    sx = int(bw * REID_CROP_SIDE_MARGIN)
    sy_top = int(bh * REID_CROP_TOP_MARGIN)
    sy_bottom = int(bh * REID_CROP_BOTTOM_MARGIN)

    cx1 = x1 + sx
    cx2 = x2 - sx
    cy1 = y1 + sy_top
    cy2 = y2 - sy_bottom
    cx1, cy1, cx2, cy2 = clamp_bbox(cx1, cy1, cx2, cy2, w, h)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop is None or crop.size == 0:
        return None
    return appearance_extractor.extract(crop)


def bbox_iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    areaB = max(1, bx2 - bx1) * max(1, by2 - by1)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def center_distance(boxA, boxB):
    ax, ay = bbox_center(boxA)
    bx, by = bbox_center(boxB)
    return float(np.hypot(ax - bx, ay - by))


def build_forced_gid_map(cam_name, detection_boxes):
    forced = {}
    cam_state = cameras.get(cam_name, {})
    prev_assignments = cam_state.get("prev_assignments", [])
    if not prev_assignments or not detection_boxes:
        return forced

    pairs = []
    for det_idx, det_box in enumerate(detection_boxes):
        for prev in prev_assignments:
            prev_gid = prev.get("gid")
            prev_box = prev.get("box")
            if prev_gid is None or prev_box is None:
                continue
            iou = bbox_iou(det_box, prev_box)
            dist = center_distance(det_box, prev_box)
            if iou >= OCCLUSION_PREV_IOU_THRESHOLD or dist <= OCCLUSION_CENTER_DIST_PX:
                score = iou * 2.0 - (dist / max(OCCLUSION_CENTER_DIST_PX, 1.0)) * 0.25
                pairs.append((score, det_idx, prev_gid))

    pairs.sort(reverse=True)
    used_det = set()
    used_gid = set()
    for score, det_idx, gid in pairs:
        if det_idx in used_det or gid in used_gid:
            continue
        forced[det_idx] = gid
        used_det.add(det_idx)
        used_gid.add(gid)
    return forced


def generate_frames(cam_name: str):
    cam_data = cameras.get(cam_name)
    if not cam_data:
        return

    source = cam_data["url"]
    source_type = cam_data.get("source_type", "camera")
    loop_video = cam_data.get("loop_video", True)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps is None or video_fps <= 0 or np.isnan(video_fps):
        video_fps = 25.0
    frame_delay = 1.0 / video_fps

    while app.is_running:
        success, frame = cap.read()

        if not success:
            if source_type == "video" and loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        cameras[cam_name]["last_frame"] = frame.copy()

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.55,
            tracker="botsort.yaml",
            verbose=False
        )

        annotated_frame = frame.copy()
        processor = cameras[cam_name].get("processor")
        src_pts = cameras[cam_name].get("src_pts")

        if processor is not None:
            annotated_frame = processor.draw_calibration_polygon(annotated_frame)

        frame_assignments = []
        prev_assignments = cameras[cam_name].get("prev_assignments", [])

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
                for i, box in enumerate(xyxy_list):
                    x1, y1, x2, y2 = box[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)

                    if processor is not None and src_pts is not None:
                        inside = point_in_polygon((foot_x, foot_y), src_pts)
                        if not inside:
                            continue

                    tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else i
                    conf_val = float(confs[i]) if confs is not None and i < len(confs) else None
                    box_wh = (max(1, x2 - x1), max(1, y2 - y1))
                    emb = extract_person_embedding(frame, x1, y1, x2, y2)
                    if emb is None:
                        continue

                    map_pos = None
                    if processor is not None:
                        try:
                            map_x, map_y = processor.to_floorplan(foot_x, foot_y)
                            map_pos = (map_x, map_y)
                        except Exception:
                            map_pos = None

                    filtered.append({
                        "idx": i,
                        "box": (x1, y1, x2, y2),
                        "foot": (foot_x, foot_y),
                        "tid": tid,
                        "conf": conf_val,
                        "box_wh": box_wh,
                        "emb": emb,
                        "map_pos": map_pos,
                        "center": bbox_center((x1, y1, x2, y2)),
                    })

                overlap_indices = set()
                for a in range(len(filtered)):
                    for b in range(a + 1, len(filtered)):
                        if bbox_iou(filtered[a]["box"], filtered[b]["box"]) >= OCCLUSION_IOU_THRESHOLD:
                            overlap_indices.add(a)
                            overlap_indices.add(b)

                forced_gid_map = build_forced_gid_map(cam_name, [item["box"] for item in filtered])
                for a, item in enumerate(filtered):
                    item["overlap"] = a in overlap_indices
                    item["forced_gid"] = forced_gid_map.get(a) if a in overlap_indices else None

                assignment_results = global_identity_manager.assign_batch(cam_name, filtered, prev_assignments=prev_assignments) if filtered else []

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

                    box_color = (0, 165, 255) if a in overlap_indices else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
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
                        debug_line = f"{match_source}:{match_score:.2f}"
                        if a in overlap_indices:
                            debug_line += " | OCC"
                        cv2.putText(
                            annotated_frame,
                            debug_line,
                            (x1, min(frame.shape[0] - 8, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 0),
                            1,
                        )

                    cv2.circle(annotated_frame, (foot_x, foot_y), 5, (0, 0, 255), -1)

                    if item["map_pos"] is not None:
                        map_x, map_y = item["map_pos"]
                        cv2.putText(
                            annotated_frame,
                            f"map=({map_x},{map_y})",
                            (foot_x + 8, foot_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2
                        )
                        global_map.update_object(gid, map_x, map_y)

                    frame_assignments.append({
                        "gid": gid,
                        "box": (x1, y1, x2, y2),
                        "center": item["center"],
                        "tid": tid,
                        "cam_name": cam_name,
                        "overlap": a in overlap_indices,
                        "ts": time.time(),
                    })

        cameras[cam_name]["prev_assignments"] = (prev_assignments + frame_assignments)[-60:]

        cv2.putText(
            annotated_frame,
            f"{cam_name} [{source_type}] ReID:{appearance_extractor.name}",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        ok, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        if source_type == "video":
            time.sleep(frame_delay)

    cap.release()


def generate_global_map():
    while app.is_running:
        canvas = global_map.draw_map()
        ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.08)


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    floorplan_exists = os.path.exists(FLOORPLAN_PATH)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "cameras": cameras,
            "floorplan_exists": floorplan_exists
        }
    )


@app.post("/upload_floorplan")
async def upload_floorplan(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open(FLOORPLAN_PATH, "wb") as f:
            f.write(contents)

        global_map.load_floorplan()
        return HTMLResponse("<script>alert('อัปโหลดแผนผังสำเร็จ'); window.location.href='/';</script>")
    except Exception as e:
        return HTMLResponse(f"<script>alert('อัปโหลดไม่สำเร็จ: {str(e)}'); window.location.href='/';</script>")


@app.post("/upload_video")
async def upload_video(
    name: str = Form(...),
    file: UploadFile = File(...),
    loop_video: bool = Form(True)
):
    try:
        if not file.filename:
            return HTMLResponse("<script>alert('ไม่พบชื่อไฟล์'); window.location.href='/';</script>")

        ext = os.path.splitext(file.filename)[1].lower()
        allowed_ext = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        if ext not in allowed_ext:
            return HTMLResponse("<script>alert('รองรับเฉพาะไฟล์วิดีโอ mp4/avi/mov/mkv/webm'); window.location.href='/';</script>")

        filename = safe_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)

        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        cameras[name] = {
            "url": save_path,
            "source_type": "video",
            "loop_video": loop_video,
            "processor": None,
            "src_pts": None,
            "dst_pts": None,
            "last_frame": None
        }

        return HTMLResponse("<script>alert('อัปโหลดวิดีโอสำเร็จ'); window.location.href='/';</script>")
    except Exception as e:
        return HTMLResponse(f"<script>alert('อัปโหลดวิดีโอไม่สำเร็จ: {str(e)}'); window.location.href='/';</script>")


@app.get("/get_floorplan")
async def get_floorplan():
    img_b64 = image_file_to_base64(FLOORPLAN_PATH)
    if img_b64 is None:
        return JSONResponse({"error": "No floorplan uploaded"}, status_code=404)
    return {"image_base64": img_b64}


@app.post("/add_camera")
async def add_camera(
    name: str = Form(...),
    url: str = Form(...)
):
    try:
        final_url = int(url) if url.isdigit() else url

        cameras[name] = {
            "url": final_url,
            "source_type": "camera",
            "loop_video": False,
            "processor": None,
            "src_pts": None,
            "dst_pts": None,
            "last_frame": None
        }

        return HTMLResponse("<script>alert('เพิ่มกล้องสำเร็จ'); window.location.href='/';</script>")
    except Exception as e:
        return HTMLResponse(f"<script>alert('เพิ่มกล้องไม่สำเร็จ: {str(e)}'); window.location.href='/';</script>")


@app.post("/delete_camera/{cam_name}")
async def delete_camera(cam_name: str):
    if cam_name in cameras:
        cam = cameras[cam_name]
        if cam.get("source_type") == "video":
            video_path = cam.get("url")
            if isinstance(video_path, str) and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass

        del cameras[cam_name]
        return JSONResponse({"status": "deleted", "camera": cam_name})

    return JSONResponse({"error": "Camera not found"}, status_code=404)


@app.get("/video_feed/{cam_name}")
async def video_feed(cam_name: str):
    if cam_name not in cameras:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    return StreamingResponse(
        generate_frames(cam_name),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/global_map_feed")
async def global_map_feed():
    return StreamingResponse(
        generate_global_map(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/capture_frame/{cam_name}")
async def capture_frame(cam_name: str):
    cam = cameras.get(cam_name)
    if not cam:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    frame = open_camera_once(cam["url"])
    if frame is None:
        return JSONResponse({"error": "Cannot capture frame"}, status_code=500)

    cam["last_frame"] = frame.copy()
    img_b64 = frame_to_base64(frame)

    return {
        "camera": cam_name,
        "image_base64": img_b64
    }


@app.post("/save_calibration/{cam_name}")
async def save_calibration(
    cam_name: str,
    src_pts: str = Form(...),
    dst_pts: str = Form(...)
):
    cam = cameras.get(cam_name)
    if not cam:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    try:
        parsed_src = parse_json_points(src_pts)
        parsed_dst = parse_json_points(dst_pts)

        processor = CameraProcessor(cam_name, parsed_src, parsed_dst)

        cam["src_pts"] = parsed_src
        cam["dst_pts"] = parsed_dst
        cam["processor"] = processor

        return JSONResponse({
            "status": "success",
            "camera": cam_name,
            "src_pts": parsed_src,
            "dst_pts": parsed_dst
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/camera_config/{cam_name}")
async def camera_config(cam_name: str):
    cam = cameras.get(cam_name)
    if not cam:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    return {
        "name": cam_name,
        "url": cam["url"],
        "source_type": cam.get("source_type", "camera"),
        "loop_video": cam.get("loop_video", False),
        "src_pts": cam["src_pts"],
        "dst_pts": cam["dst_pts"],
        "has_homography": bool(cam["processor"] is not None)
    }


@app.post("/shutdown")
async def shutdown_system():
    app.is_running = False

    def kill_server():
        time.sleep(1)
        os._exit(0)

    threading.Thread(target=kill_server).start()
    return {"status": "shutting down"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
