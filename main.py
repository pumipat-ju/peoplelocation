import cv2
import os
import threading
import time
import json
import base64
import numpy as np
from collections import deque

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
os.makedirs("static", exist_ok=True)


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
# Global Map Manager
# -----------------------------
class GlobalMapManager:
    def __init__(self, trail_len=50, timeout_sec=2.0):
        self.trail_len = trail_len
        self.timeout_sec = timeout_sec

        self.base_map = None
        self.objects = {}      # {gid: (mx, my)}
        self.tracks = {}       # {gid: deque([(mx,my), ...])}
        self.last_seen = {}    # {gid: ts}

        self.load_floorplan()

    def load_floorplan(self):
        if os.path.exists(FLOORPLAN_PATH):
            img = cv2.imread(FLOORPLAN_PATH)
            if img is not None:
                self.base_map = img
                return

        # fallback ถ้ายังไม่มี floorplan
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
        self.cleanup_stale_objects()

        canvas = self.base_map.copy()

        # # trajectory
        # for gid, pts in self.tracks.items():
        #     pts_list = list(pts)
        #     if len(pts_list) >= 2:
        #         for i in range(1, len(pts_list)):
        #             cv2.line(canvas, pts_list[i - 1], pts_list[i], (255, 180, 0), 2)

        # current positions
        for gid, (mx, my) in self.objects.items():
            display_id = gid.split("-")[-1]
            cv2.circle(canvas, (mx, my), 8, (0, 255, 0), -1)
            # outline
            # cv2.putText(canvas, gid, (mx + 10, my - 8),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4)
            tracker_id = gid.split("-")[-1]
            # text
            cv2.putText(canvas, f"ID {tracker_id}", (mx + 10, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        return canvas


global_map = GlobalMapManager(trail_len=1, timeout_sec=0.5)  # ปรับให้แสดงเฉพาะตำแหน่งปัจจุบันและลบออกเร็วขึ้น


# -----------------------------
# Streaming
# -----------------------------
def point_in_polygon(point, polygon_pts):
    """
    point: (x, y)
    polygon_pts: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    if polygon_pts is None:
        return True  # ถ้ายังไม่ได้ calibrate ให้ผ่านไว้ก่อน

    poly = np.array(polygon_pts, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0

def generate_frames(cam_name: str):
    cam_data = cameras.get(cam_name)
    if not cam_data:
        return

    camera_url = cam_data["url"]
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while app.is_running:
        success, frame = cap.read()
        if not success:
            break

        cameras[cam_name]["last_frame"] = frame.copy()

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="botsort.yaml",
            verbose=False
        )

        annotated_frame = frame.copy()
        processor = cameras[cam_name].get("processor")
        src_pts = cameras[cam_name].get("src_pts")

        # วาดกรอบ calibration ไว้ก่อน
        if processor is not None:
            annotated_frame = processor.draw_calibration_polygon(annotated_frame)

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

                for i, box in enumerate(xyxy_list):
                    x1, y1, x2, y2 = box[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # ใช้จุดเท้าเป็นตัวแทนตำแหน่งคน
                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)

                    # ถ้ามี calibration แล้ว ให้ track เฉพาะคนที่อยู่ใน polygon
                    if processor is not None and src_pts is not None:
                        inside = point_in_polygon((foot_x, foot_y), src_pts)
                        if not inside:
                            continue

                    # track id
                    if track_ids is not None and i < len(track_ids):
                        tid = int(track_ids[i])
                        gid = f"{cam_name}-{tid}"   # เอาไว้ใช้ภายในระบบ
                        label = f"ID {tid}"         # เอาไว้โชว์บนภาพ
                    else:
                        tid = i
                        gid = f"{cam_name}-{tid}"
                        label = f"ID {tid}"

                    # confidence
                    if confs is not None and i < len(confs):
                        label += f" {confs[i]:.2f}"

                    # วาดเฉพาะคนที่อยู่ในพื้นที่ calibration
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(annotated_frame, (foot_x, foot_y), 5, (0, 0, 255), -1)

                    if processor is not None:
                        try:
                            map_x, map_y = processor.to_floorplan(foot_x, foot_y)

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

                        except Exception as e:
                            cv2.putText(
                                annotated_frame,
                                f"H error: {str(e)}",
                                (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
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
        "src_pts": cam["src_pts"],
        "dst_pts": cam["dst_pts"],
        "has_homography": cam["processor"] is not None
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