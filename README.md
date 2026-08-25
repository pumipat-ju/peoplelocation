# Multi-Camera People Location & Re-Identification

ระบบตรวจจับ ติดตาม และระบุตัวบุคคลแบบไม่ระบุชื่อข้ามกล้อง พร้อมแสดงตำแหน่งบน Floorplan ผ่านหน้าเว็บ

> Production backend ของโปรเจกต์คือ `backend/main.py` ส่วน `main.py` ที่ root เป็นโค้ดเดิมและไม่ควรใช้เป็น source of truth

## ความสามารถหลัก

- ตรวจจับบุคคลด้วย YOLOv8
- ติดตามบุคคลภายในกล้องด้วย BoT-SORT
- สร้าง Global ID ข้ามกล้องด้วย OSNet Re-ID
- จับคู่แบบ one-to-one ด้วย Hungarian assignment
- ลดการสลับ ID ระหว่างการบังกันด้วย occlusion/anti-ID-swap logic
- รองรับกล้อง RTSP/HTTP, webcam และไฟล์วิดีโอ
- อัปโหลดวิดีโอหลายไฟล์พร้อม progress และข้อความผิดพลาดจาก backend
- Calibrate กล้องกับ Floorplan ด้วย 4 จุด พร้อมเส้นเชื่อมและเส้นช่วยเล็ง
- แปลงตำแหน่งจากภาพกล้องไปยัง Floorplan ด้วย homography
- เก็บข้อมูล runtime แยกจาก source code เพื่อใช้งานกับ Docker volume

Global ID เป็นการจดจำจากรูปลักษณ์ในวิดีโอ ไม่ใช่การยืนยันชื่อหรืออัตลักษณ์จริงของบุคคล และประสิทธิภาพอาจลดลงเมื่อเปลี่ยนเสื้อผ้า มุมกล้อง หรือสภาพแสงอย่างมาก

## สถาปัตยกรรม

```text
Camera / Video
      |
      v
YOLO person detection
      |
      v
BoT-SORT local tracking
      |
      v
OSNet appearance embedding
      |
      v
Hungarian global assignment
      |
      +--> Global ID memory
      |
      v
Homography --> Floorplan
```

## โครงสร้างโปรเจกต์

```text
PeopleLocation/
|- backend/
|  |- main.py                 # Production FastAPI backend
|  |- Dockerfile
|  |- requirements.txt
|  |- static/                 # Runtime uploads/floorplan (ไม่ commit)
|  `- data/                   # Runtime database (ไม่ commit)
|- frontend/
|  |- src/
|  |- Dockerfile
|  `- package.json
|- reid_training/             # เตรียมข้อมูล, train และ evaluate OSNet
|- models/osnet/              # รายงานผลขนาดเล็ก; weights/arrays ไม่ commit
|- weights/                   # Local model weights (ไม่ commit)
|- docker-compose.yml
`- README.md
```

## สิ่งที่ต้องติดตั้ง

วิธีที่แนะนำ:

- Docker Desktop พร้อม Docker Compose
- พื้นที่ว่างสำหรับ Docker image และ model weights

สำหรับ Development แบบไม่ใช้ Docker:

- Python 3.10+
- Node.js 18+
- npm

## เตรียม Model Weights

ไฟล์ model ไม่ถูกเก็บใน Git ให้สร้างโฟลเดอร์ `weights/` แล้ววางไฟล์ดังนี้:

```text
weights/
|- yolov8s.pt
`- osnet_x1_0_market1501.pth
```

ชื่อไฟล์ต้องตรงกับ `docker-compose.yml`:

| Model | Host path | Container path |
|---|---|---|
| YOLOv8s | `weights/yolov8s.pt` | `/app/yolov8s.pt` |
| OSNet x1.0 | `weights/osnet_x1_0_market1501.pth` | `/app/weights/osnet_x1_0_market1501.pth` |

หาก OSNet weight หายหรือโหลดไม่ได้ backend จะ fallback ไปใช้ lightweight appearance extractor ซึ่งเหมาะกับการสำรองการทำงาน แต่ไม่ควรใช้เป็นผล Re-ID หลัก

## เริ่มใช้งานด้วย Docker

จาก directory หลักของโปรเจกต์:

```powershell
docker compose up -d --build
```

ตรวจสถานะ:

```powershell
docker compose ps
docker compose logs -f backend
```

เปิดใช้งาน:

- Web UI: <http://localhost:3000>
- Backend API: <http://localhost:8899>
- OpenAPI docs: <http://localhost:8899/docs>

หยุดระบบ:

```powershell
docker compose down
```

ข้อมูลใน `backend/static/` และ `backend/data/` จะยังอยู่หลัง recreate container เพราะถูก mount เป็น volume จากเครื่อง host

## วิธีใช้งาน

1. เปิดหน้าเว็บที่ `http://localhost:3000`
2. อัปโหลดภาพ Floorplan
3. เพิ่ม camera stream หรืออัปโหลดไฟล์วิดีโอ
4. กดปุ่ม Calibrate ของกล้อง
5. คลิก 4 จุดบนภาพกล้อง และ 4 จุดที่สัมพันธ์กันบน Floorplan ด้วยลำดับเดียวกัน
6. กด Save Calibration แล้วตรวจตำแหน่ง Global ID บนแผนผัง

### การอัปโหลดวิดีโอ

- รองรับ `.mp4`, `.avi`, `.mov`, `.mkv` และ `.webm`
- จำกัดขนาดไม่เกิน 500 MB ต่อไฟล์
- เมื่อเลือกหนึ่งไฟล์ ชื่อกล้องจะตรงกับ Camera Name
- เมื่อเลือกหลายไฟล์ ระบบจะตั้งชื่อเป็น `<prefix>_1`, `<prefix>_2`, ...
- วิดีโอและ Floorplan เป็นข้อมูล runtime และถูก ignore จาก Git

### การ Calibration

- ใช้จุดบริเวณพื้นหรือจุดอ้างอิงที่มองเห็นชัด
- เลือกจุดทั้งสองภาพด้วยลำดับเดียวกัน
- เส้นทึบแสดงจุดที่เลือกแล้ว
- เส้นประแสดงแนวจากจุดล่าสุดไปยังตำแหน่งเมาส์
- เมื่อครบ 4 จุด ระบบจะแสดงกรอบพื้นที่ calibration

## Development แบบไม่ใช้ Docker

### Backend

```powershell
cd backend
python -m venv venv
./venv/Scripts/Activate.ps1
python -m pip install --upgrade pip setuptools wheel Cython
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
python main.py
```

สำหรับ manual mode ให้วาง OSNet checkpoint ที่ `weights/osnet_x1_0_market1501.pth` จาก repository root

### Frontend

เปิด terminal อีกหน้าหนึ่ง:

```powershell
cd frontend
npm ci
npm run dev
```

Vite development server จะเปิดที่ `http://localhost:5173` ตามค่าเริ่มต้น และ frontend ปัจจุบันเชื่อม backend ที่ `http://localhost:8899`

## การตรวจสอบก่อน Commit

ตรวจ syntax ของ backend:

```powershell
python -m py_compile backend/main.py
```

ตรวจ production build ของ frontend:

```powershell
cd frontend
npm ci
npm run build
```

ตรวจรายการที่จะ commit:

```powershell
git status --short
git diff --cached --stat
git diff --cached
```

ไม่ควรใช้ `git add -f` กับ dataset, uploaded video, database, model weight หรือไฟล์ `.npz`

## OSNet Training และ Evaluation

เครื่องมือสำหรับเตรียม dataset, fine-tune, evaluate และวิเคราะห์ threshold อยู่ใน `reid_training/` อ่านขั้นตอนโดยละเอียดที่ [reid_training/README.md](reid_training/README.md)

ข้อมูลที่สร้างจากการ train จะไม่ถูก commit:

- `datasets/`, `dataset_raw/`, `dataset_custom/`, `reid_dataset/`
- `runs/`, `checkpoints/`, `wandb/`
- `*.pt`, `*.pth`, `*.npz`
- `reid_training/sequences.json`

สามารถเก็บไฟล์รายงานผลขนาดเล็ก เช่น `models/osnet/*baseline.json` ใน Git เพื่อใช้เปรียบเทียบผลทดลองได้ แต่ไม่ควรอ้างว่า accuracy ดีขึ้นหากยังไม่มีผลวัดจาก test split ที่เหมาะสม

## Troubleshooting

### Backend แจ้งว่าไม่พบ OSNet weight

ตรวจชื่อไฟล์:

```text
weights/osnet_x1_0_market1501.pth
```

จากนั้น rebuild backend:

```powershell
docker compose up -d --build backend
docker compose logs --tail 100 backend
```

### Upload failed

ตรวจว่า backend ทำงานที่ port `8899` และดูรายละเอียดจาก:

```powershell
docker compose logs --tail 100 backend
```

### Port ถูกใช้งานอยู่

ตรวจ port `3000` และ `8899` หรือเปลี่ยน port ฝั่งซ้ายใน `docker-compose.yml`

## ข้อมูลที่ไม่ควร Commit

- วิดีโอ, Floorplan และภาพบุคคลจากระบบจริง
- SQLite database รวมถึงไฟล์ `-wal` และ `-shm`
- `.env`, token, password หรือ URL กล้องที่มี credential
- Python virtual environment, `node_modules/` และ build output
- Dataset, model weights และ training artifacts ขนาดใหญ่

ตรวจ `.gitignore` และ staged files ทุกครั้งก่อน push โดยเฉพาะเมื่อ repository มีข้อมูลจากกล้องจริง

## Production OSNet checkpoint loading

The production backend resolves the OSNet checkpoint independently of the
current working directory. Local development defaults to
`weights/osnet_x1_0_market1501.pth` at the repository root. Override the
runtime configuration when needed:

```powershell
$env:REID_ENABLED = "true"
$env:REID_CHECKPOINT_PATH = "C:\path\to\osnet_x1_0_market1501.pth"
$env:REID_DEVICE = "auto"
python backend/main.py
```

Docker Compose mounts `${MODEL_WEIGHTS_DIR:-./weights}` read-only at
`/app/weights` and defaults `REID_CHECKPOINT_PATH` to
`/app/weights/osnet_x1_0_market1501.pth`. Copy `.env.example` to `.env` to
override these values. The checkpoint itself is intentionally excluded from
Git.

Check the active runtime without relying on startup logs:

```powershell
Invoke-RestMethod http://localhost:8899/api/status | ConvertTo-Json -Depth 5
```

The `reid` object reports the configured architecture and checkpoint, whether
the checkpoint passed validation, the active device and extractor, fallback
state, embedding dimension, and the initialization error when degraded.

Run the production embedding smoke test from the repository root:

```powershell
python backend/osnet_smoke_test.py
```

The smoke test exits non-zero if OSNet is disabled, the checkpoint is missing
or incompatible, fallback is active, or a valid embedding cannot be created.

## Per-camera BoT-SORT state

Each camera/source owns a private YOLO predictor and BoT-SORT state. Local
track IDs are scoped to that camera; only the Re-ID layer assigns Global IDs
across cameras. Inspect `cameras.<name>.tracker` in `/api/status` for the
tracker instance ID, BoT-SORT state IDs, generation/reset count, active local
tracks, and last processed frame.

Reset one source without resetting other cameras or deleting Global ID memory:

```powershell
Invoke-RestMethod -Method Post http://localhost:8899/api/reset_tracker/<camera-name>
```

## Live camera realtime processing

Live sources use a background worker per camera. Numeric source strings are
converted to OpenCV device indices (`"0"` becomes `cv2.VideoCapture(0)`), while
RTSP/HTTP sources remain URL strings. Capture and inference are decoupled by a
single replaceable frame slot: if inference is slower than the camera, stale
frames are counted as dropped instead of building an ever-growing queue.

Each processed live frame follows the normal production path:

```text
VideoCapture -> latest-frame slot -> process_camera_frame()
             -> per-camera YOLO/BoT-SORT -> Global Re-ID -> JPEG cache
```

On a read failure the worker releases the broken capture and retries after
`LIVE_CAMERA_RECONNECT_INTERVAL_SEC` (default `1.0` second). A successful
reconnect resets only that camera's local tracker state. Global identity memory
and all other camera trackers are preserved. Removing a camera or shutting down
the server stops both worker threads and releases the capture handle.

### Manual webcam smoke test

Run this test from the local Python environment so source `0` refers to the
host webcam. A backend inside Docker must be given access to the camera device;
otherwise index `0` is the container's device namespace, not automatically the
Windows host webcam.

1. Start the backend yourself, then add a camera in the frontend with source
   `0`.
2. Check runtime diagnostics:

```powershell
Invoke-RestMethod http://localhost:8899/api/status |
    ConvertTo-Json -Depth 8
```

3. Under `cameras.<name>`, verify `source_type` is `live`,
   `live_worker.running` and `capture_open` are true, and `frame_index`,
   `captured_frames`, and `processed_frames` continue increasing.
4. Open the live preview and verify bounding boxes, Local IDs, and Global IDs
   update continuously.
5. If cameras `0` and `1` are available, add both and verify their
   `tracker.tracker_instance_id` and `live_worker.worker_instance_id` values are
   different.

The status also reports `dropped_frames`, `processing_fps`,
`last_frame_age_ms`, processing duration/latency, reconnect count, and the last
worker error. Credentials embedded in RTSP URLs are masked in status responses
and camera-add logs.

## Deadline regression workflow

Run the safe production-path regression suite from the repository root. The
runner forces SQLite identity state into memory, disables OSNet model loading,
blocks real device access in its import guard, and uses fakes for live/uploaded
sources. Tests run from an isolated temporary runtime directory and fingerprint
protected runtime data before and after the suite. It does not start services,
containers, webcams, or RTSP streams.

```powershell
backend\venv\Scripts\python.exe -m backend.deadline_regression `
    --output deadline_regression_report.json
```

The command prints a short summary and writes the complete machine-readable
JSON report. The report includes test counts, scenario coverage, safe runtime
configuration, topology schema/path metadata, protected floorplan/topology
fingerprints, and project-specific event metrics. These metrics are explicitly
not standard IDF1 or HOTA.

Verify the frontend upload-offset payload and production build separately:

```powershell
cd frontend
npm.cmd test
npm.cmd run build
```

Automated regression tests never replace the manual real-device checks for
Docker device passthrough, live preview, calibration, uploaded playback, and a
physical two-camera handoff.
