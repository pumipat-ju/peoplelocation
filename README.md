# Multi-Camera People Location

ระบบตรวจจับ ติดตาม และเชื่อมโยงบุคคลข้ามกล้อง โดยแสดงตำแหน่งบน floorplan แบบเรียลไทม์

> Production backend อยู่ที่ `backend/main.py` เท่านั้น ไม่ใช่ `main.py` ที่ repository root

## ความสามารถ

- ตรวจจับบุคคลด้วย YOLOv8s

- ติดตามภายในแต่ละกล้องด้วย BoT-SORT โดย Local Track ID แยกขอบเขตต่อกล้อง

- ใช้ OSNet `osnet_x1_0` สร้าง appearance embedding 512 มิติเพื่อ Re-ID ข้ามกล้อง

- จับคู่ Global ID แบบ one-to-one ด้วย Hungarian assignment พร้อม quality, topology, presence, ambiguity และ anti-ID-swap gates

- เก็บ Global ID memory และสถานะ identity ใน SQLite

- รองรับ RTSP/HTTP/RTMP stream, local webcam index และ uploaded video

- แสดง annotated video (bounding box, Local Track ID, Global ID) และตำแหน่งบน floorplan

- Calibrate กล้องกับ floorplan ด้วยจุดคู่สัมพันธ์ 4 จุด

Global ID คือการเชื่อมโยง appearance ในวิดีโอ ไม่ใช่การยืนยันชื่อหรืออัตลักษณ์จริงของบุคคล

## ภาพรวมระบบ

```text

Live camera / RTSP / uploaded video

                 |

                 v

          YOLO person detection

                 |

                 v

     Per-camera BoT-SORT local tracking

                 |

                 v

          OSNet Re-ID embedding

                 |

                 v

 Global assignment coordinator + Hungarian matching

                 |

       +---------+----------+

       |                    |

       v                    v

SQLite identity memory   Annotated JPEG preview

       |                    |

       +--------> Homography / floorplan map

```

สำหรับ live camera ตัว capture worker จะรับภาพลง latest-frame slot แล้วส่งต่อให้ processing worker; เมื่อ inference ช้ากว่ากล้อง จะทิ้งเฟรมเก่าแทนการสะสมคิวไม่จำกัด

## โครงสร้าง repository

```text

peoplelocation/

├── backend/

│   ├── main.py                    # FastAPI production backend

│   ├── Dockerfile

│   ├── requirements.txt

│   ├── reid_config.py             # OSNet architecture/preprocessing contract

│   ├── identity_store.py          # SQLite persistence

│   ├── camera_topology.json       # Cross-camera transition rules

│   ├── osnet_smoke_test.py

│   ├── test_*.py                  # Device-free regression tests

│   ├── static/                    # Floorplan และ uploaded videos at runtime

│   └── data/                      # SQLite database at runtime

├── frontend/

│   ├── src/                       # React UI

│   ├── Dockerfile

│   └── package.json

├── weights/

│   └── osnet_x1_0_market1501.pth  # Production OSNet checkpoint

├── docker-compose.yml

├── .env.example

└── README.md

```

`backend/static/`, `backend/data/`, videos, floorplans, SQLite files และ credentials ไม่ควรนำขึ้น Git

## สิ่งที่ต้องมี

### Docker deployment

- Docker Desktop หรือ Docker Engine ที่รัน Linux containers ได้

- Docker Compose v2 (`docker compose`)

- Port `3000` และ `8899` ต้องว่าง

- Internet ในช่วง build ครั้งแรก เพื่อดาวน์โหลด Python dependencies และติดตั้ง `torchreid` จาก GitHub

- ไฟล์โมเดลตามหัวข้อถัดไป

- หากใช้ local webcam: Docker daemon ต้องเข้าถึง `/dev/video*` ของ host ได้

Compose ปัจจุบันเปิด frontend ที่ `3000` และ backend ที่ `8899` และ bind-mount runtime data กลับมาที่ `backend/static` กับ `backend/data` บน host

### Development แบบไม่ใช้ Docker

- Python 3.10+

- Node.js 18+ และ npm

- กล้องหรือ RTSP/video files สำหรับทดสอบจริง

## เตรียม model weights

### 1. OSNet Re-ID checkpoint

Backend ต้องใช้ไฟล์ชื่อนี้:

```text

weights/osnet_x1_0_market1501.pth

```

Checkpoint นี้ถูก track อยู่ใน repository เวอร์ชันปัจจุบันแล้ว แต่สามารถดาวน์โหลดใหม่ได้จากลิงก์นี้หากไฟล์หายหรือใช้ source bundle ที่ไม่มี binary:

<https://drive.usercontent.google.com/download?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA&export=download&authuser=0>

PowerShell:

```powershell

New-Item -ItemType Directory -Force weights | Out-Null

$osnetUrl = 'https://drive.usercontent.google.com/download?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA&export=download&authuser=0'

Invoke-WebRequest -Uri $osnetUrl -OutFile weights\osnet_x1_0_market1501.pth

```

Docker จะ mount directory `weights/` ไปที่ `/app/weights` แบบ read-only และใช้ checkpoint ที่ `/app/weights/osnet_x1_0_market1501.pth`

### 2. YOLOv8s detection weight

Compose mount ไฟล์นี้เข้าฝั่ง backend:

```text

backend/yolov8s.pt

```

ไฟล์ YOLO weight ไม่ได้ track ใน Git จึงต้องวางไว้ที่ path นี้ก่อน build/run Docker หากยังไม่มี ให้ดาวน์โหลด `yolov8s.pt` จาก [Ultralytics YOLO documentation](https://docs.ultralytics.com/models/yolov8/) แล้วบันทึกด้วยชื่อนี้

## เริ่มใช้งานด้วย Docker

### 1. สร้างไฟล์ environment (แนะนำ)

```powershell

Copy-Item .env.example .env

```

ค่าหลักใน `.env`:

| Variable | ค่าเริ่มต้น | ความหมาย |

|---|---|---|

| `MODEL_WEIGHTS_DIR` | `./weights` | directory ของ OSNet checkpoint บน host |

| `IDENTITY_DB_PATH` | `/app/data/identity_memory.sqlite3` | SQLite identity store ใน container |

| `REID_ENABLED` | `true` | เปิด OSNet Re-ID |

| `REID_CHECKPOINT_PATH` | `/app/weights/osnet_x1_0_market1501.pth` | checkpoint path ใน container |

| `REID_DEVICE` | `auto` | `auto`, `cpu`, `cuda` หรือ `cuda:<index>` |

| `REID_THRESHOLD_SAFETY_MODE` | `conservative` | ใช้ `validated` เฉพาะเมื่อมี validation report รองรับ |

| `LIVE_CAMERA_RECONNECT_INTERVAL_SEC` | `1.0` | ระยะเวลารอก่อน reconnect source ที่อ่านไม่ได้ |

### 2. Build และเริ่มระบบ

```powershell

docker compose up -d --build

docker compose ps

docker compose logs -f backend

```

เปิดใช้งาน:

- Web UI: <http://localhost:3000>

- Backend API: <http://localhost:8899>

- OpenAPI: <http://localhost:8899/docs>

ตรวจ runtime status:

```powershell

Invoke-RestMethod http://localhost:8899/api/status |

    ConvertTo-Json -Depth 8

```

### 3. หยุดระบบ

```powershell

docker compose down

```

คำสั่งนี้ไม่ลบข้อมูลใน bind mounts `backend/static/` และ `backend/data/`

## ใช้ local webcam ใน Docker (Linux / WSL)

`docker-compose.yml` ปัจจุบัน map `/dev/video0` และ `/dev/video1` เข้า backend container สำหรับกล้อง UVC/V4L2 บน Linux หรือ WSL ที่ attach อุปกรณ์แล้ว

ก่อน start ให้ตรวจสอบ device บน host:

```bash

ls -l /dev/video*

```

หากมีเพียง `/dev/video0` ให้ลบบรรทัด `/dev/video1:/dev/video1` ออกจาก `devices:` ใน `docker-compose.yml` ก่อน start; ทุก device ที่ map ต้องมีอยู่จริงบน host

หลัง start ตรวจว่ามองเห็นใน container:

```powershell

docker compose exec backend sh -lc 'ls -l /dev/video*'

```

เพิ่มกล้องผ่าน UI โดยใส่ Camera Name และ source เป็น `0` (หรือ `1`) ระบบจะแปลง string ตัวเลขเป็น OpenCV device index

สำหรับ **Linux local camera index เท่านั้น** backend จะเปิดด้วย V4L2 และตั้งค่า MJPG, `640x480`, `30 FPS` ก่อนอ่านเฟรมแรก เพื่อรองรับอุปกรณ์ที่ `cv2.VideoCapture(0)` เปิดได้แต่ `read()` ไม่ได้ ส่วน Windows native, RTSP/HTTP/RTMP และ uploaded video ใช้เส้นทางเดิม

ตรวจค่าจาก `/api/status` ที่ `cameras.<name>.live_worker`:

- `running: true`, `capture_open: true`

- `captured_frames` และ `processed_frames` เพิ่มขึ้น

- หากล้มเหลว `last_error` จะบอก `backend`, `device/index`, `fourcc`, `width`, `height`, `fps`, `opened` และ `read`

> Docker Desktop บน Windows ไม่ส่งผ่าน USB webcam ไปยัง Linux container ให้อัตโนมัติ ต้อง attach webcam เข้า WSL2 (เช่นผ่าน USB/IP) ให้เห็นเป็น `/dev/video*` ก่อน หรือรัน backend แบบ native Windows / ใช้ RTSP stream แทน

## วิธีใช้ผ่าน Web UI

1. เปิด <http://localhost:3000>

2. Upload floorplan image

3. เพิ่ม source อย่างใดอย่างหนึ่ง

   - local webcam: `0`, `1`, ...

   - RTSP: `rtsp://...`

   - HTTP/HTTPS/RTMP stream

   - uploaded video

4. รอ preview ที่มี bounding boxes, Local Track IDs และ Global IDs

5. กดปุ่ม calibration ของกล้อง แล้วเลือก 4 จุดในภาพกล้องกับ 4 จุดที่ตรงกันบน floorplan ตามลำดับเดียวกัน จากนั้น Save

6. ดูตำแหน่งที่ map ได้บน Live Tracking Map

### Uploaded video

- รองรับ `.mp4`, `.avi`, `.mov`, `.mkv`

- จำกัดขนาดไฟล์ละ 500 MB

- หาก upload หลายไฟล์ ระบบตั้งชื่อเป็น `<prefix>_1`, `<prefix>_2`, ...

- ตั้ง `Offset (s)` ต่อไฟล์ได้ เพื่อจัด timeline ข้ามกล้อง

- ใช้ Play/Pause Selected หรือ Play/Pause All เพื่อควบคุมการเล่น

## ตรวจสอบ OSNet Re-ID

หลังระบบเริ่มทำงาน ค่าใน `/api/status` → `reid` ที่ถือว่า OSNet production พร้อมใช้งานควรมีอย่างน้อย:

```json

{

  "enabled": true,

  "model_architecture": "osnet_x1_0",

  "checkpoint_loaded": true,

  "fallback_active": false,

  "embedding_dimension": 512,

  "error": null

}

```

รัน smoke test ใน backend container:

```powershell

docker compose exec backend python osnet_smoke_test.py

```

Smoke test จะ fail หาก OSNet ปิดอยู่, checkpoint ไม่พบหรือไม่ compatible, fallback ถูกเปิด หรือสร้าง embedding ที่ไม่ถูกต้องไม่ได้

### CPU และ CUDA

`REID_DEVICE=auto` จะเลือก `cuda` เฉพาะเมื่อ `torch.cuda.is_available()` เป็นจริง มิฉะนั้นจะใช้ CPU โดยอัตโนมัติ

Dockerfile/Compose ชุดปัจจุบันไม่ได้เปิด NVIDIA runtime หรือบังคับติดตั้ง CUDA-enabled PyTorch ดังนั้น deployment Docker มาตรฐานควรคาดหวัง CPU ก่อน การบังคับ `REID_DEVICE=cuda` โดยที่ container ไม่เห็น GPU จะทำให้ OSNet initialization ล้มเหลว

หากต้องการ CUDA ต้องจัดเตรียมทั้ง NVIDIA driver, NVIDIA Container Toolkit, GPU exposure ให้ container และ PyTorch build ที่รองรับ CUDA ก่อน แล้วตรวจ `reid.device`, `checkpoint_loaded` และ `fallback_active` จาก `/api/status` หลัง deploy

## API ที่ใช้งานบ่อย

OpenAPI ที่ <http://localhost:8899/docs> คือสัญญา API ที่ครบถ้วนที่สุด

| Method | Endpoint | ใช้ทำอะไร |

|---|---|---|

| `GET` | `/api/status` | สถานะกล้อง, workers, tracker, Re-ID, identity และ global assignment |

| `POST` | `/api/upload_floorplan` | อัปโหลด floorplan image |

| `POST` | `/api/add_camera` | เพิ่ม numeric device index หรือ stream URL |

| `DELETE` | `/api/delete_camera/{cam_name}` | ลบกล้อง/source และหยุด worker ที่เกี่ยวข้อง |

| `POST` | `/api/upload_video` | อัปโหลด video source พร้อม offset |

| `POST` | `/api/video_playback` | play/pause uploaded videos |

| `GET` | `/api/video_feed/{cam_name}` | annotated MJPEG preview |

| `GET` | `/api/global_map_feed` | floorplan MJPEG preview |

| `GET` | `/api/capture_frame/{cam_name}` | ภาพปัจจุบันสำหรับ calibration |

| `POST` | `/api/save_calibration/{cam_name}` | บันทึก 4-point homography |

| `POST` | `/api/reset_tracker/{cam_name}` | reset tracker เฉพาะกล้อง โดยไม่ลบ Global ID memory |

| `GET` / `PUT` | `/api/topology` | อ่าน/ปรับ topology transition rules |

## รันแบบไม่ใช้ Docker

### Backend

```powershell

Set-Location backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel Cython

pip install -r requirements.txt

pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git

python main.py

```

วาง OSNet checkpoint ที่ `../weights/osnet_x1_0_market1501.pth` เมื่ออยู่ใน directory `backend/` และวาง YOLO weight ที่ `backend/yolov8s.pt`

### Frontend

เปิด terminal ใหม่จาก repository root:

```powershell

Set-Location frontend

npm ci

npm run dev

```

Vite development UI ปกติเปิดที่ <http://localhost:5173> และเรียก backend ที่ `http://localhost:8899`

## การทดสอบก่อน commit

จาก repository root:

```powershell

python -m py_compile backend/main.py

python -m unittest backend.test_live_camera backend.test_deadline_regression_guards backend.test_timestamp_offsets

Set-Location frontend

npm ci

npm test

npm run build

```

การทดสอบเหล่านี้ไม่แทนการตรวจจริงกับ Docker device passthrough, live preview, calibration, uploaded playback และ physical handoff ระหว่างสองกล้อง

## แก้ปัญหาที่พบบ่อย

### `checkpoint_loaded=false` หรือ `fallback_active=true`

1. ตรวจชื่อและ path ของ checkpoint

2. ตรวจ `.env` ว่า `REID_ENABLED=true` และ `REID_CHECKPOINT_PATH=/app/weights/osnet_x1_0_market1501.pth`

3. ตรวจ mount ด้วย `docker compose exec backend ls -l /app/weights`

4. ดู `reid.error` ใน `/api/status` และ backend logs

5. หลังแก้ไฟล์หรือ environment ให้ rebuild/recreate backend:

```powershell

docker compose up -d --build backend

docker compose logs --tail 100 backend

```

### Local webcam preview ว่าง

1. ยืนยันว่า host เห็น `/dev/video0`

2. ยืนยันว่า `devices:` map เฉพาะ device ที่มีจริง

3. ตรวจ `/api/status` → `live_worker.last_error`

4. ตรวจว่า source ที่เพิ่มเป็นเลข index เช่น `0` ไม่ใช่ file path

5. ดู backend logs:

```powershell

docker compose logs --tail 100 backend

```

### UI ติดต่อ backend ไม่ได้

ตรวจ container และ port:

```powershell

docker compose ps

Invoke-RestMethod http://localhost:8899/api/status

```

## ข้อมูลที่ไม่ควรเผยแพร่

- RTSP URL ที่มี username/password, `.env` และ tokens

- floorplans, uploaded videos และภาพบุคคลจากระบบจริง

- SQLite identity database และไฟล์ `-wal` / `-shm`

- datasets และ training artifacts ที่ไม่ได้รับอนุญาตให้เผยแพร่

Windows + Docker Desktop + WSL2: ใช้ USB Webcam แบบละเอียด

บน Windows, cv2.VideoCapture(0) แบบ native ใช้ DirectShow/Media Foundation แต่ backend ใน Docker เป็น Linux container จึงต้องเห็น webcam เป็น /dev/video* ก่อน จึงจะใช้ local camera index ได้

เส้นทางที่ต้องผ่านคือ:

Windows USB Webcam
        |
        v
usbipd-win
        |
        v
WSL2 / Docker Desktop VM
        |
        v
uvcvideo + V4L2
        |
        v
/dev/video0
        |
        v
Docker backend container
        |
        v
OpenCV CAP_V4L2

1. ติดตั้ง usbipd-win

เปิด PowerShell แบบ Administrator:

winget install --interactive --exact dorssel.usbipd-win

หลังติดตั้ง หาก usbipd ยังไม่ถูกพบใน shell เดิม ให้ปิด PowerShell แล้วเปิดใหม่ หรือเรียก executable โดยตรง:

& "C:\Program Files\usbipd-win\usbipd.exe" list

2. หา BUSID ของ webcam

& "C:\Program Files\usbipd-win\usbipd.exe" list

ตัวอย่าง:

BUSID  VID:PID    DEVICE                        STATE
1-6    1bcf:28c4  USB Video Device, USB Camera  Not shared

ให้ใช้ BUSID ของ webcam จริงจากเครื่อง ไม่ควรคัดลอกเลข BUSID จากตัวอย่าง

3. Share webcam ให้ WSL

PowerShell แบบ Administrator:

& "C:\Program Files\usbipd-win\usbipd.exe" bind --busid <BUSID>

ตรวจอีกครั้ง:

& "C:\Program Files\usbipd-win\usbipd.exe" list

สถานะควรเป็น Shared

4. Attach webcam เข้า WSL2

ต้องมี WSL2 distribution กำลังรันอยู่ เช่น docker-desktop หรือ Ubuntu

ตรวจรายการ:

wsl -l -v

จากนั้น attach:

& "C:\Program Files\usbipd-win\usbipd.exe" attach --wsl --busid <BUSID>

ตรวจสถานะ:

& "C:\Program Files\usbipd-win\usbipd.exe" list

สถานะ webcam ควรเป็น Attached

หลัง wsl --shutdown, reboot, detach หรือถอด-เสียบ webcam ใหม่ อาจต้องรัน attach --wsl อีกครั้ง

5. โหลด UVC/V4L2 driver ใน WSL

เข้า WSL distribution ที่ Docker Desktop ใช้:

wsl -d docker-desktop

ตรวจ USB:

lsusb

ถ้าเห็น webcam แต่ยังไม่มี /dev/video* ให้โหลด UVC driver:

modprobe uvcvideo

ตรวจ module:

lsmod | grep -E "uvcvideo|videodev"

ตรวจ video device:

ls -l /dev/video*

ปกติ webcam หนึ่งตัวอาจสร้างมากกว่าหนึ่ง node เช่น:

/dev/video0
/dev/video1

ไม่ได้หมายความว่าทุก node จะเป็น capture endpoint ที่ OpenCV เปิดอ่านภาพได้

หากต้องการตรวจ kernel log:

dmesg | tail -n 50

ข้อความลักษณะนี้หมายความว่า UVC device ถูกตรวจพบแล้ว:

Found UVC device USB Camera
registered new interface driver uvcvideo

6. Map /dev/video* เข้า backend container

docker-compose.yml:

services:
  backend:
    devices:
      - /dev/video0:/dev/video0
      - /dev/video1:/dev/video1

หาก host มีเพียง /dev/video0 ให้ map เฉพาะ node ที่มีจริง

ตรวจ Compose:

docker compose config

จากนั้น recreate backend:

docker compose up -d --force-recreate backend

ถ้ามีการแก้ production code หรือ Docker image ให้ใช้:

docker compose up -d --build --force-recreate backend

ตรวจ device mapping ที่ Docker รับจริง:

docker inspect peoplelocation-backend --format '{{json .HostConfig.Devices}}'

ตรวจ /dev ภายใน container:

docker compose exec backend ls -l /dev

หรือถ้าต้องการใช้ wildcard ให้รันผ่าน shell ใน container:

docker compose exec backend sh -lc "ls -l /dev/video*"

การรัน docker compose exec backend ls -l /dev/video* ตรง ๆ จาก PowerShell อาจไม่ expand * ตามที่คาด จึงควรใช้ sh -lc

7. ทดสอบ OpenCV ใน container ก่อนใช้ผ่าน UI

ทดสอบ V4L2 + MJPG โดยตรง:

docker compose exec backend python -c "import cv2,time; c=cv2.VideoCapture(0,cv2.CAP_V4L2); c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')); c.set(cv2.CAP_PROP_FRAME_WIDTH,640); c.set(cv2.CAP_PROP_FRAME_HEIGHT,480); c.set(cv2.CAP_PROP_FPS,30); print('opened=',c.isOpened()); print('backend=',c.getBackendName() if c.isOpened() else None); [(time.sleep(.2),print('read',i,c.read()[0])) for i in range(5)]; c.release()"

ผลที่พร้อมใช้งานควรมีลักษณะ:

opened= True
backend= V4L2
read 0 True
read 1 True
read 2 True
read 3 True
read 4 True

ถ้า cv2.VideoCapture(0) ให้ opened=True แต่ read=False ขณะที่คำสั่ง V4L2 ด้านบนอ่านได้ แสดงว่า device passthrough ใช้งานได้แล้ว แต่ default OpenCV capture configuration ไม่เหมาะกับ webcam/USBIP path นี้

Production backend จึงควรใช้ V4L2 + MJPG + 640x480 + 30 FPS เฉพาะ numeric local webcam บน Linux โดยไม่เปลี่ยน Windows native, network streams หรือ uploaded-video path

8. เพิ่ม webcam ผ่าน PeopleLocation UI

เมื่อ /dev/video0 อ่านได้จาก container แล้ว ให้เพิ่มกล้องด้วย:

Camera Name: <ชื่อกล้อง>
Source: 0

จากนั้นตรวจ:

calibration modal ต้องมีภาพจาก webcam

live preview ต้องมีภาพ

bounding box / Local Track ID / Global ID แสดงตาม inference result

/api/status → cameras.<name>.live_worker ควรมี running=true และ capture_open=true

captured_frames และ processed_frames ควรเพิ่มขึ้น

ดู backend log:

docker compose logs backend --tail 100

หาก log มี global assignment ต่อเนื่อง เช่น observation จากชื่อกล้องที่เพิ่ม แสดงว่าภาพเดินทางผ่าน capture → detection/tracking → Re-ID → global assignment แล้ว

9. Troubleshooting matrix

อาการ

จุดที่ควรตรวจ

usbipd command not found

เปิด PowerShell ใหม่ หรือเรียก C:\Program Files\usbipd-win\usbipd.exe โดยตรง

usbipd list เห็นกล้องแต่ Not shared

รัน usbipd bind --busid <BUSID>

สถานะ Shared แต่ WSL ยังไม่เห็น

รัน usbipd attach --wsl --busid <BUSID>

lsusb เห็น webcam แต่ไม่มี /dev/video*

รัน modprobe uvcvideo และตรวจ dmesg

WSL มี /dev/video0 แต่ container ไม่มี

ตรวจ devices: ใน Compose, docker inspect ...HostConfig.Devices, แล้ว recreate container

container มี /dev/video0, opened=False

ตรวจว่าใช้ V4L2 endpoint ถูกตัว และลอง node อื่นเฉพาะที่มีจริง

opened=True, read=False

ทดสอบ CAP_V4L2 + MJPG + 640x480 + 30 FPS

/dev/video1 เปิดไม่ได้

อาจเป็น metadata/non-capture endpoint; ใช้ node ที่ read=True

calibration/live preview ว่าง แต่ OpenCV test อ่านได้

ตรวจ production live-camera open path และ /api/status → live_worker.last_error

หลัง reboot กล้องหาย

attach USB เข้า WSL ใหม่ และตรวจ uvcvideo//dev/video*

Deployment checklist ก่อนใช้งานจริง

ก่อน demo หรือ deploy on-premise ให้ตรวจตามลำดับนี้:

[ ] docker compose config ผ่าน
[ ] backend/static และ backend/data mount ถูกต้อง
[ ] yolov8s.pt อยู่ที่ backend/yolov8s.pt
[ ] OSNet checkpoint อยู่ที่ weights/osnet_x1_0_market1501.pth
[ ] /api/status -> checkpoint_loaded=true
[ ] /api/status -> fallback_active=false
[ ] webcam/RTSP/video source อ่านได้
[ ] calibration preview ใช้งานได้
[ ] live preview แสดง bbox / LID / GID
[ ] global map feed ใช้งานได้
[ ] SQLite persistence path อยู่ใน /app/data
[ ] uploaded-video playback และ offset ใช้งานได้
[ ] cross-camera handoff ผ่านการทดสอบกับข้อมูลจริง

สำหรับ Windows + Docker Desktop + USB webcam ให้เพิ่ม:

[ ] usbipd list -> webcam = Attached
[ ] WSL lsusb -> เห็น webcam
[ ] WSL ls -l /dev/video* -> มี video node
[ ] Docker inspect -> HostConfig.Devices มี video node
[ ] container -> /dev/video0 มีอยู่
[ ] OpenCV V4L2 smoke test -> read=True

หมายเหตุเรื่องประสิทธิภาพ

ระบบสามารถทำงานบน CPU ได้ แต่ YOLO/BoT-SORT และ OSNet เป็นงาน inference ที่มีต้นทุนสูง โดยเฉพาะ OSNet เมื่อมีหลายคนในเฟรม

REID_DEVICE=auto จะใช้ CUDA เมื่อ PyTorch ภายใน runtime รองรับ CUDA และ container มองเห็น NVIDIA GPU มิฉะนั้นจะ fallback ไป CPU ตาม configuration ที่รองรับ

การเปิด GPU acceleration เป็นงาน deployment/runtime แยกจาก webcam passthrough:

Webcam path: USB -> WSL -> /dev/video* -> Docker -> OpenCV
GPU path: NVIDIA driver -> Docker GPU exposure -> CUDA-enabled PyTorch

การแก้ GPU ไม่ควรเปลี่ยน camera capture, calibration, uploaded-video playback หรือ frame-cache architecture

Quick health check

หลัง start ระบบ สามารถใช้ชุดคำสั่งสั้น ๆ นี้ตรวจสุขภาพ deployment ได้:

docker compose ps

docker compose logs backend --tail 50

Invoke-RestMethod http://localhost:8899/api/status |
    ConvertTo-Json -Depth 8

หากใช้ local webcam บน Windows + Docker Desktop:

& "C:\Program Files\usbipd-win\usbipd.exe" list

docker inspect peoplelocation-backend --format '{{json .HostConfig.Devices}}'

docker compose exec backend sh -lc "ls -l /dev/video*"

