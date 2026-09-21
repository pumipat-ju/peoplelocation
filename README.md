# PeopleLocation

ระบบติดตามตำแหน่งบุคคลและเชื่อมโยง Global ID ข้ามกล้อง (Cross-Camera Person Re-Identification) แบบเรียลไทม์ รองรับกล้องสด, network stream และวิดีโอที่อัปโหลด พร้อมแสดงผลบนภาพจากกล้องและ floorplan

> Production backend คือ `backend/main.py` ส่วน `main.py` ที่ repository root ไม่ใช่ source of truth

Global ID เป็นรหัสภายในสำหรับเชื่อมโยง track ที่ระบบประเมินว่าเป็นบุคคลเดียวกัน ไม่ใช่การระบุตัวบุคคลจริง และผลลัพธ์ควรถูกประเมินกับข้อมูลจากสถานที่ติดตั้งจริงก่อนใช้งานเชิงตัดสินใจ

## Architecture

```text
Live camera / RTSP / HTTP / uploaded video
                    |
                    v
          YOLOv8 person detection
                    |
                    v
     Per-camera BoT-SORT local tracking
                    |
                    v
       OSNet x1.0 Re-ID (512-D, L2 normalized)
                    |
                    v
   Quality gallery + multi-evidence candidate scoring
                    |
                    v
 Topology/time gates + ambiguity/occlusion safeguards
                    |
                    v
 Hungarian one-to-one global assignment
                    |
                    v
 Global ID State Machine (PROVISIONAL/ACTIVE/DORMANT/EXPIRED)
          |                         |
          v                         v
 SQLite persistence        Homography / floorplan
```

องค์ประกอบหลักของระบบปัจจุบัน:

- **YOLOv8s** ตรวจจับบุคคล โดยเลือก CUDA และ FP16 อัตโนมัติเมื่อ PyTorch มองเห็น GPU
- **BoT-SORT** เก็บ tracker state แยกต่อกล้อง เพื่อไม่ให้ Local Track ID ปะปนกัน
- **OSNet `osnet_x1_0`** สร้าง embedding 512 มิติจาก person crop
- **Global assignment** รวม appearance, track continuity, motion/map distance, เวลา, topology, quality และ presence ก่อนทำ Hungarian one-to-one assignment
- **Global ID State Machine** ควบคุมเส้นทางตัดสินใจและ lifecycle ของ identity ตั้งแต่ `PROVISIONAL` ไปยัง `ACTIVE`, `DORMANT` และ `EXPIRED`
- **Topology** จำกัดทิศทางการย้ายกล้อง, เวลาเดินทางต่ำสุด/สูงสุด และการมองเห็นพร้อมกันระหว่างกล้อง
- **SQLite** เก็บ identity memory และข้อมูลที่ต้องใช้ข้ามการ restart

การประมวลผล Re-ID และ global assignment อยู่ downstream จาก acquisition/tracking; capture worker ของแต่ละกล้องไม่รอ synchronization จากกล้องอื่น

## โครงสร้างโมดูลหลัก

```text
backend/
├── main.py                 # FastAPI production application และ pipeline integration
├── reid/
│   ├── config.py           # Runtime config และ OSNet contract
│   ├── crop.py             # Original/improved person crop
│   ├── extractor.py        # Checkpoint validation และ feature extraction
│   ├── gallery.py          # Quality-aware embedding gallery
│   └── similarity.py       # L2 normalization และ cosine similarity
├── identity/
│   ├── assignment.py       # Assignment helpers
│   ├── handoff.py          # Cross-camera handoff helpers
│   ├── manager.py          # Global identity policy และ batch matching
│   ├── pending.py          # Pending/ambiguous evidence
│   ├── persistence.py      # Identity persistence integration
│   ├── presence.py         # Camera presence rules
│   └── state_machine.py    # Observation decision state machine
├── topology/
│   ├── config.py           # Schema, validation และ config loading
│   ├── gates.py            # Topology gate primitives
│   └── travel_time.py      # Travel-time normalization
├── diagnostics/
│   ├── quality.py          # Re-ID quality metadata
│   └── serialization.py    # Safe forensic/diagnostic serialization
├── camera_topology.json    # Runtime topology configuration
├── identity_store.py       # SQLite identity store
└── test_*.py               # Regression/unit tests
```

Frontend อยู่ใน `frontend/` และเป็น React 19 + Vite 6 ส่วน production container build เป็น static site ที่เสิร์ฟด้วย Nginx

## Requirements

- Python 3.10 (ตรงกับ backend Docker image)
- Node.js 18+ และ npm สำหรับ frontend
- FFmpeg/OpenCV-compatible camera หรือ video source ตามรูปแบบที่ระบบรองรับ
- Model files:
  - `backend/yolov8s.pt`
  - `weights/osnet_x1_0_peoplelocation_balanced_v2.pth`
- Docker Engine/Desktop และ Docker Compose v2 หากรันด้วย container

Python packages ระบุใน `backend/requirements.txt` ได้แก่ FastAPI, Uvicorn, Ultralytics, OpenCV headless, PyTorch/Torchvision, SciPy, NumPy, torchreid และแพ็กเกจประกอบอื่น ๆ

## Production Re-ID configuration

ค่าปริยายปัจจุบันคือ:

| รายการ | ค่า |
|---|---|
| Architecture | `osnet_x1_0` |
| Checkpoint | `osnet_x1_0_peoplelocation_balanced_v2.pth` |
| Embedding | 512-D, L2 normalized |
| Input | RGB `256x128`, ImageNet normalization |
| Crop | `improved` จาก detector/tracker bounding box พร้อม side/top/bottom margin |
| Device | `auto` |
| Threshold safety | `conservative` |

`conservative` เป็นค่าที่แนะนำสำหรับ production ปัจจุบัน: similarity-only shortcut จะถูกปิด และการ reuse Global ID ต้องผ่าน evidence, lifecycle, topology/presence, ambiguity และ assignment safeguards ที่เกี่ยวข้อง อย่าเปลี่ยนเป็น `validated` จนกว่าจะมี held-out validation report ที่รองรับ deployment นั้น

Checkpoint ถูกตรวจ architecture, preprocessing และ embedding dimension ขณะโหลด หากโหลดไม่ได้ runtime จะรายงาน error ผ่าน `/api/status`; ไม่ควรถือว่า fallback เป็น production Re-ID ที่สมบูรณ์

## รันด้วย local venv

ตัวอย่างสำหรับ PowerShell จาก repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

เตรียม model files ตาม path ด้านบน แล้วเริ่ม backend โดยให้ working directory เป็น `backend`:

```powershell
Set-Location backend
$env:REID_CHECKPOINT_PATH = (Resolve-Path ..\weights\osnet_x1_0_peoplelocation_balanced_v2.pth).Path
uvicorn main:app --host 0.0.0.0 --port 8899
```

เปิดอีก terminal เพื่อรัน frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

เข้าใช้งาน development UI ที่ URL ที่ Vite แสดง (ปกติ `http://localhost:5173`) โดย backend ต้องอยู่ที่ `http://localhost:8899` ตาม frontend implementation ปัจจุบัน

## รันด้วย Docker Compose

คัดลอก environment template และเตรียม weights ก่อน:

```powershell
Copy-Item .env.example .env
docker compose config
docker compose up --build
```

จากนั้นเปิด:

- Frontend: `http://localhost:3000`
- Backend/API: `http://localhost:8899`
- FastAPI docs: `http://localhost:8899/docs`

Compose mount `backend/static`, `backend/data`, `backend/yolov8s.pt` และ `${MODEL_WEIGHTS_DIR}` เข้า backend container โดย OSNet weights เป็น read-only ปัจจุบัน service backend ระบุ `gpus: all`; เครื่องที่ใช้ CUDA ต้องมี NVIDIA driver และ NVIDIA Container Toolkit ที่ Docker เข้าถึงได้

Docker camera device passthrough ขึ้นกับ host OS และ Docker runtime และไม่ได้ถูกกำหนดแบบตายตัวใน Compose โปรดตรวจว่า container มองเห็นอุปกรณ์จริงก่อนเพิ่ม numeric camera source

## CUDA / GPU

- `REID_DEVICE=auto` เลือก CUDA เมื่อ `torch.cuda.is_available()` เป็นจริง มิฉะนั้นใช้ CPU
- ระบุ `cpu`, `cuda` หรือ `cuda:<index>` ได้โดยตรง; ถ้าบังคับ CUDA แต่ runtime มองไม่เห็น GPU การเริ่ม Re-ID จะล้มเหลวและแสดง error ใน status
- YOLO เลือก CUDA/FP16 อัตโนมัติจาก PyTorch runtime เดียวกัน
- Local venv ที่ต้องการ GPU ต้องติดตั้ง PyTorch build ที่ตรงกับ NVIDIA driver/CUDA ของเครื่อง; `requirements.txt` ไม่ได้ pin CUDA build
- CPU ใช้งานได้ แต่ throughput จะขึ้นกับจำนวนกล้อง, จำนวนบุคคล และ hardware

## Environment variables สำคัญ

| Variable | Default | ความหมาย |
|---|---|---|
| `MODEL_WEIGHTS_DIR` | `./weights` | Host directory ที่ Compose mount ไป `/app/weights` |
| `REID_ENABLED` | `true` | เปิด/ปิด OSNet Re-ID |
| `REID_CHECKPOINT_PATH` | Balanced V2 path | Path ของ production OSNet checkpoint |
| `REID_DEVICE` | `auto` | `auto`, `cpu`, `cuda` หรือ `cuda:<index>` |
| `REID_CROP_MODE` | `improved` | Crop policy: `original` หรือ `improved` |
| `REID_THRESHOLD_SAFETY_MODE` | `conservative` | Threshold safety: `conservative` หรือ `validated` |
| `IDENTITY_DB_PATH` | `/app/data/identity_memory.sqlite3` ใน Compose | SQLite identity persistence path |
| `LIVE_CAMERA_RECONNECT_INTERVAL_SEC` | `1.0` | เวลารอก่อน reconnect live source |

ไฟล์ `.env` ถูกใช้โดย Docker Compose แต่ local Uvicorn จะไม่โหลด `.env` ให้อัตโนมัติ ให้ export ตัวแปรใน shell หรือใช้กลไก env loading ของ deployment

## ตรวจสุขภาพด้วย `/api/status`

```powershell
Invoke-RestMethod http://localhost:8899/api/status |
    ConvertTo-Json -Depth 10
```

จุดสำคัญที่ควรตรวจ:

- `reid.enabled` เป็น `true`
- `reid.model_architecture` เป็น `osnet_x1_0`
- `reid.checkpoint_loaded` เป็น `true`
- `reid.embedding_dimension` เป็น `512`
- `reid.crop_mode` เป็น `improved`
- `reid.threshold_safety_mode` เป็น `conservative`
- `reid.similarity_only_shortcut_enabled` เป็น `false`
- `reid.fallback_active` เป็น `false` และ `reid.error` เป็น `null`
- `global_assignment` แสดงสถานะ coordinator ตามปกติ
- `identity` แสดง lifecycle/persistence diagnostics
- สำหรับกล้องสด `cameras.<name>.live_worker.running` และ `capture_open` ควรเป็น `true` และ counters ควรเพิ่มขึ้น

สถานะ HTTP 200 เพียงอย่างเดียวไม่ยืนยันว่า checkpoint, กล้อง และ inference พร้อมใช้งาน ต้องตรวจค่าภายใน response ด้วย

## การใช้งาน

1. เปิด backend และ frontend
2. อัปโหลด floorplan
3. เพิ่ม live camera/network stream หรืออัปโหลด video ผ่าน UI
4. เปิด calibration ของแต่ละกล้องและกำหนดจุดสัมพันธ์กับ floorplan
5. ตรวจ live/video preview, Local Track ID, Global ID และ global map
6. กำหนด topology ผ่าน `GET/PUT /api/topology` เมื่อทราบเส้นทางจริงระหว่างกล้อง

`backend/camera_topology.json` เริ่มต้นด้วย `enforce: false` และไม่มี transitions ดังนั้น topology gate จะมีผลตามข้อกำหนดของสถานที่ต่อเมื่อบันทึก config ที่เหมาะสมแล้ว

## Testing และ validation

รัน test suite จาก repository root:

```powershell
$env:REID_ENABLED = "false"
$env:IDENTITY_DB_PATH = ":memory:"
python -m unittest discover -s backend -p "test_*.py"
```

รัน frontend checks:

```powershell
Set-Location frontend
npm run lint
npm run build
```

ชุดทดสอบครอบคลุม OSNet contract/checkpoint integration, crop policy, threshold safety, per-camera tracker, tracklet/gallery quality, timestamps/video offsets, topology/travel time, global batch assignment, ambiguity rejection, identity lifecycle/persistence และ live-camera regression บางชุดต้องใช้ checkpoint หรือ artifact เฉพาะ และการตรวจกล้องจริง, uploaded-video playback, calibration, Docker device access และ cross-camera handoff ต้องทำ integration validation บน deployment hardware เพิ่มเติม

อย่าสรุปว่า Re-ID แม่นยำขึ้นจากการผ่าน unit tests เพียงอย่างเดียว ต้องวัดกับ held-out data และ scenario จริงของไซต์

## ข้อจำกัดที่ทราบ

- คนที่ถูกบัง, crop ไม่ครบตัว, motion blur, ขนาดเล็ก หรือแสง/มุมกล้องต่างกันมาก ทำให้ embedding และการเชื่อม Global ID ไม่น่าเชื่อถือขึ้น
- การซ้อนทับและ occlusion อาจทำให้ tracker สลับ Local ID; ระบบมี hold/freeze, quality gates และ anti-ID-swap logic แต่ไม่สามารถกำจัดความผิดพลาดได้ทั้งหมด
- Cross-camera blind zone ไม่มีภาพต่อเนื่องให้ตรวจสอบ จึงต้องพึ่ง appearance, เวลาเดินทาง และ topology; config ที่ไม่ตรงกับสถานที่อาจ reject handoff ที่ถูกต้องหรือเปิดทางให้ match ผิด
- บุคคลแต่งกายคล้ายกันอาจทำให้ top candidates ใกล้กัน ระบบจึง defer/reject กรณี ambiguous แทนการบังคับ merge ซึ่งอาจทำให้เกิด Global ID ใหม่ชั่วคราว
- Topology เริ่มต้นไม่ enforce และต้อง calibrate ตาม layout, overlap และ travel time จริง
- SQLite ช่วยกู้ identity memory หลัง restart แต่ไม่ทำให้ evidence ที่คุณภาพต่ำกลายเป็น match ที่เชื่อถือได้
- ประสิทธิภาพแบบเรียลไทม์ขึ้นกับ GPU/CPU, resolution, FPS, จำนวนกล้อง และจำนวนบุคคลพร้อมกัน
