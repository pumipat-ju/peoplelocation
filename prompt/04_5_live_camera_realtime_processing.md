# Prompt 04.5 — เพิ่ม Live Camera / Webcam Realtime Processing Loop

## ปัญหา

ระบบปัจจุบันสามารถรับค่า source เช่น `0` สำหรับ webcam index ได้ แต่ฝั่ง `backend/main.py` ยังไม่มี live-camera worker ที่อ่าน frame ต่อเนื่องแล้วส่งเข้า `process_camera_frame()` แบบเดียวกับ uploaded video

ผลคือ live source อาจเปิด capture ได้หรืออ่านภาพได้บางจังหวะ แต่ยังไม่มี realtime tracking pipeline ที่ทำงานต่อเนื่อง จึงยังไม่สามารถใช้ webcam / USB camera / RTSP stream เพื่อทดสอบ YOLO + BoT-SORT + Global Re-ID แบบ realtime ได้จริง

งานนี้ต้องแก้เฉพาะเส้นทาง **live camera processing** โดยห้ามทำลาย uploaded-video flow, video synchronization, per-camera tracker isolation และ Global ID memory ที่มีอยู่แล้ว

---

## เป้าหมาย

ทำให้ระบบรองรับ live camera แบบ realtime โดยสามารถ:

- ใส่ `0`, `1`, `2`, ... เป็น webcam/camera index
- รองรับ URL stream เช่น RTSP ถ้าโครงสร้างปัจจุบันรองรับ
- อ่าน frame ต่อเนื่อง
- ส่งทุก frame ที่เลือกประมวลผลเข้า `process_camera_frame(camera_name, frame, ...)`
- ใช้ per-camera YOLO / BoT-SORT context ที่ Prompt 04 ทำไว้แล้ว
- หยุด/restart/reconnect source ได้อย่างปลอดภัย
- ไม่ block API server
- ไม่สร้าง worker ซ้ำสำหรับกล้องเดียวกัน
- ไม่ล้าง Global ID memory เพียงเพราะ live capture reconnect

---

## ข้อกำหนดสำคัญ

### 1. อ่าน architecture ปัจจุบันก่อนแก้

ตรวจ flow ของ:

- camera/source registration
- uploaded-video worker
- `process_camera_frame()`
- per-camera tracker context
- camera cleanup/remove
- `/api/status`
- video reset/restart logic
- frame/result cache ที่ frontend ใช้อ่านภาพ

ห้ามสร้าง parallel pipeline ใหม่ที่ bypass `process_camera_frame()` หากไม่จำเป็น

Live camera ควร reuse processing path เดียวกับระบบปัจจุบันให้มากที่สุด

### 2. Parse live source ให้ถูกต้อง

รองรับอย่างน้อย:

```text
0
1
2
```

โดย string ที่เป็น integer ล้วนต้องถูกแปลงเป็น:

```python
cv2.VideoCapture(0)
cv2.VideoCapture(1)
```

ไม่ใช่:

```python
cv2.VideoCapture("0")
```

ซึ่งอาจถูกตีความเป็นชื่อไฟล์

สำหรับ source ที่ไม่ใช่ integer เช่น:

```text
rtsp://...
http://...
```

ให้ส่งเป็น string ตามเดิม

สร้าง helper ที่ชัดเจน เช่น `parse_video_source(value)` พร้อม test

### 3. สร้าง Live Camera Worker ต่อกล้อง

แต่ละ live camera ต้องมี worker/context แยกกัน เช่นแนวคิด:

```text
LiveCameraWorker
- camera_name
- source
- capture
- thread
- stop_event
- running
- reconnect_count
- frame_index
- last_frame_time
- last_error
```

ไม่จำเป็นต้องใช้ชื่อนี้ หาก repository มี abstraction ที่เหมาะสมกว่าอยู่แล้ว

Worker ต้อง:

1. เปิด `cv2.VideoCapture`
2. ตรวจ `isOpened()`
3. อ่าน frame ต่อเนื่อง
4. เพิ่ม frame index ต่อกล้อง
5. ส่ง frame เข้า processing path ปัจจุบัน
6. update latest-frame/result cache ที่ frontend ใช้อยู่
7. handle read failure
8. reconnect ตาม policy
9. หยุดอย่าง clean เมื่อ camera ถูก remove หรือ server shutdown

### 4. ห้าม Block FastAPI / API Thread

ห้ามเขียนลักษณะ:

```python
while True:
    cap.read()
```

ตรง request handler

การเพิ่ม live camera ต้อง start background worker/thread/task แล้ว return response ได้ทันที

ต้องตรวจ thread safety กับ data structures ที่มีอยู่แล้ว

### 5. ใช้ Per-camera Tracker จาก Prompt 04

ห้ามสร้าง tracker กลางกลับมาอีก

ทุก frame จาก live camera ต้องเข้า:

```text
camera A
→ tracker context A
→ process_camera_frame()
```

และ:

```text
camera B
→ tracker context B
→ process_camera_frame()
```

ต้องยังคง guarantee:

- YOLO predictor / BoT-SORT persistent state แยกต่อกล้อง
- tracker lock แยกต่อกล้อง
- local track IDs มี scope เฉพาะ camera
- Global ID memory อยู่เหนือ local tracker

### 6. Realtime Frame Scheduling / Backpressure

Live source ไม่ควรสร้าง queue ยาวจน latency เพิ่มเรื่อย ๆ

สำหรับ realtime camera ให้ prefer **latest-frame semantics** มากกว่าการพยายามประมวลผลทุก frame ถ้า inference ช้ากว่า camera FPS

ตัวอย่าง:

```text
camera = 30 FPS
processing = 12 FPS
```

ระบบควรทิ้ง frame เก่าที่ค้างและประมวลผล frame ล่าสุด แทนการสะสม queue จนภาพ delay หลายวินาที

ห้ามใช้ unbounded queue

ถ้า architecture ปัจจุบันมี latest-frame cache อยู่แล้ว ให้ reuse

เพิ่ม diagnostic เช่น:

- captured frames
- processed frames
- dropped/skipped frames
- effective processing FPS

### 7. Timestamp สำหรับ Live Camera

Live frame ต้องมี timestamp ที่เหมาะกับ realtime processing

ห้ามใช้ uploaded-video offset logic แบบเดียวกันโดยอัตโนมัติถ้า semantics ไม่ตรงกัน

สำหรับ live camera ให้ใช้ monotonic/realtime event timestamp ที่ระบบ cross-camera สามารถเปรียบเทียบได้

เก็บอย่างน้อย:

- frame capture/event time
- processing time ถ้าจำเป็นสำหรับ diagnostic

ห้ามให้ processing latency ถูกตีความเป็น travel time

อย่า refactor timestamp architecture ใหญ่เกิน scope; ให้เชื่อมกับ contract ปัจจุบันแบบ minimal

### 8. Reconnect Handling

หาก webcam/RTSP อ่าน frame ไม่ได้:

- อย่า crash server
- บันทึก last error
- release capture ที่เสีย
- retry/reconnect แบบมี delay
- ห้าม busy loop 100% CPU
- จำกัดหรือ configure reconnect interval

เมื่อ reconnect:

- Local tracker ของกล้องนั้นสามารถ reset ได้ถ้าจำเป็น
- ต้องใช้ existing per-camera reset mechanism
- ห้าม reset tracker ของกล้องอื่น
- ห้ามล้าง Global ID memory ทั้งระบบ

ถ้า source ถูกถอดออกอย่างถาวร worker ต้องสามารถถูก stop ได้

### 9. Start / Stop / Remove Lifecycle

ตรวจ endpoint/API ปัจจุบันและ integrate ให้เหมาะสม

ต้องไม่เกิด:

```text
Camera A → worker #1
เพิ่ม Camera A ซ้ำ → worker #2
```

โดยไม่มี cleanup

เมื่อ add/start camera:

- validate source
- create/start worker หนึ่งตัว

เมื่อ remove camera:

- set stop event
- join thread แบบ timeout
- release `VideoCapture`
- cleanup live worker state
- cleanup camera-local tracker state ตาม policy เดิม

เมื่อ server shutdown:

- stop workers ทุกตัว
- release camera handles

ห้ามทิ้ง zombie threads หรือ camera device lock หลัง shutdown

### 10. `/api/status` Diagnostics

เพิ่มข้อมูลใต้ camera แต่ละตัว เช่น:

```json
{
  "source_type": "live",
  "source": 0,
  "live_worker": {
    "running": true,
    "capture_open": true,
    "frame_index": 1234,
    "captured_frames": 1234,
    "processed_frames": 520,
    "dropped_frames": 714,
    "last_frame_age_ms": 25,
    "processing_fps": 11.8,
    "reconnect_count": 0,
    "last_error": null
  }
}
```

ชื่อ field ปรับตาม architecture จริงได้

อย่า expose credential จาก RTSP URL ใน status/log ถ้า URL มี username/password ให้ mask secret ก่อนแสดง

### 11. Webcam `0` Acceptance Test

เพิ่ม deterministic/unit tests และถ้าทำได้ให้มี manual smoke test

Unit tests อย่างน้อย:

1. `"0"` → integer `0`
2. `"1"` → integer `1`
3. RTSP URL → string เดิม
4. worker ไม่ถูกสร้างซ้ำสำหรับ camera เดิม
5. remove camera ทำให้ worker stop
6. read failure ไม่ crash server
7. reconnect reset เฉพาะ tracker ของกล้องนั้น
8. worker A/B ไม่ share state
9. latest-frame/backpressure ไม่สร้าง unbounded queue

Mock `cv2.VideoCapture` ใน automated test; ห้าม require webcam จริงสำหรับ unit suite

### 12. Manual Smoke Test

หลัง implementation ให้เขียนขั้นตอนทดสอบ manual แต่ **ห้าม restart/deploy เองหาก `AGENTS.md` ห้าม**

ตัวอย่างสิ่งที่ผู้ใช้ควรทำภายหลัง:

1. เปิด backend
2. เพิ่ม camera source `0`
3. ตรวจ:

```powershell
Invoke-RestMethod http://localhost:8899/api/status |
    ConvertTo-Json -Depth 8
```

4. ต้องเห็น:
   - source type = live
   - worker running
   - capture open
   - frame index เพิ่มต่อเนื่อง
   - processed frames เพิ่ม
   - tracker instance ของ camera ถูกสร้าง
5. เปิด frontend/live preview
6. เดินผ่านกล้องและตรวจ:
   - bounding box update realtime
   - Local ID ถูก track ต่อเนื่อง
   - Global ID ถูกแสดง
7. ถ้ามี webcam 2 ตัว:
   - ใช้ source `0` และ `1`
   - ตรวจ tracker instance ต้องคนละตัว
   - ทั้งสอง worker ต้องทำงานพร้อมกัน

### 13. อย่าแก้สิ่งเหล่านี้ใน Prompt นี้

ห้ามขยาย scope ไปทำ:

- fine-tune OSNet
- เปลี่ยน Re-ID threshold
- Prompt 05 tracklet quality gallery
- Prompt 07 entry/exit topology
- Prompt 08 global Hungarian redesign
- SQLite identity schema redesign
- frontend redesign ใหญ่
- root `main.py`
- deployment/restart หาก `AGENTS.md` ห้าม

ถ้าพบ bug ที่ block live camera จริง ให้แก้เฉพาะส่วน minimal และรายงานแยก

---

## Acceptance Criteria

ถือว่างานผ่านเมื่อ:

- ใส่ source `"0"` แล้ว backend เปิด `cv2.VideoCapture(0)` จริง
- live camera มี background processing loop
- frame ถูกส่งเข้า `process_camera_frame()` ต่อเนื่อง
- API server ไม่ถูก block
- frontend/result cache ได้ frame ใหม่ต่อเนื่อง
- Local tracking ใช้ per-camera tracker จาก Prompt 04
- camera A/B ไม่ share tracker state
- realtime processing ไม่สะสม unbounded frame queue
- camera remove/shutdown release capture และ worker ได้
- read failure/reconnect ไม่ทำให้ backend crash
- `/api/status` บอกสถานะ live camera ได้
- automated tests ผ่าน
- ไม่มี regression ใน uploaded-video flow ที่เกี่ยวข้อง

---

## กติกาการทำงานร่วมกัน

- อ่าน `AGENTS.md` และคำสั่งของ repository ก่อนทุกอย่าง
- อ่านโค้ดปัจจุบันก่อนแก้ ห้ามเดา architecture
- แก้เฉพาะ `backend/main.py` และไฟล์ helper/test/config ที่จำเป็น
- ห้ามแก้ root `main.py`
- ห้าม deploy/restart หาก project instructions ห้าม
- อย่ารายงานว่า webcam จริงผ่าน ถ้าทดสอบด้วย mock เท่านั้น
- อย่ารายงานว่า cross-camera accuracy ดีขึ้นจากงานนี้ เพราะ prompt นี้แก้ live input pipeline ไม่ใช่ Re-ID accuracy
- รัน relevant tests, syntax checks และ `git diff --check`
- หาก test เก่าที่ไม่เกี่ยวข้อง fail จาก API drift ให้รายงานแยก ห้ามไล่ refactor นอก scope

เมื่อจบงาน ให้สรุป:

1. Root cause ที่พบจริง
2. ไฟล์ที่แก้
3. Live-camera architecture ที่ implement
4. วิธี parse source `0` / `1` / RTSP
5. Worker lifecycle และ reconnect behavior
6. Backpressure/frame-dropping strategy
7. ผล automated tests
8. Manual test ที่ยังต้องทำกับ webcam จริง
9. `/api/status` fields ที่ใช้ตรวจ runtime
10. ความเสี่ยง/สิ่งที่ยังไม่ได้แก้
