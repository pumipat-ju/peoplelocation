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
`- osnet_x0_25_market1501.pt
```

ชื่อไฟล์ต้องตรงกับ `docker-compose.yml`:

| Model | Host path | Container path |
|---|---|---|
| YOLOv8s | `weights/yolov8s.pt` | `/app/yolov8s.pt` |
| OSNet x0.25 | `weights/osnet_x0_25_market1501.pt` | `/app/osnet_x0_25_market1501.pt` |

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

สำหรับ manual mode ให้วาง `yolov8s.pt` และ `osnet_x0_25_market1501.pt` ใน `backend/` เพราะ path ของ model เป็น relative กับ working directory

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
weights/osnet_x0_25_market1501.pt
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
