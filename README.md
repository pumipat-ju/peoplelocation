# AI CCTV Monitoring & Multi-Camera Tracking System

ระบบจัดการและติดตามบุคคลจากกล้องวงจรปิด (Person Tracking & Re-Identification) ข้ามกล้องแบบเรียลไทม์ พร้อมการแสดงผลบนแผนที่ (Bird's Eye View/Floorplan) พัฒนาด้วย **YOLOv8**, **FastAPI** (Backend) และ **Vite + React** (Frontend)

## 🌟 ความสามารถหลัก (Features)

- **Multi-Camera Person Re-Identification (Re-ID):** 
  ติดตามการเคลื่อนที่ของบุคคลข้ามกล้องได้ โดยระบบจะให้ Global ID เดียวกันสำหรับบุคคลเดียวกัน แม้จะเดินหลุดกรอบจากกล้องหนึ่งไปโผล่อีกกล้องหนึ่งก็ตาม
- **รองรับ OSNet และ Lightweight Feature Extraction:** 
  สามารถใช้ `torchreid` เพื่อดึงคุณลักษณะที่แม่นยำสูง (OSNet) ได้ หรือหากไม่ได้ติดตั้งไว้ ระบบจะปรับไปใช้งาน Lightweight Feature Extractor (HSV Color + Shape) ให้โดยอัตโนมัติ
- **Floorplan & Homography Mapping:** 
  สามารถอัปโหลดรูปแผนผังอาคาร (Floorplan) เข้าสู่ระบบ และทำการสอบเทียบ (Calibration) มุมมองภาพจากกล้องแต่ละตัว ให้สอดคล้องกับพิกัดบนแผนผัง (Bird's Eye View)
- **Occlusion Handling:** 
  ระบบมีกลไกป้องกันปัญหาคนเดินบังกัน (Occlusion) ทำให้ ID ไม่เกิดการสลับกันมั่วในขณะที่คนเดินสวนกันหน้ากล้อง
- **รองรับวิดีโออัปโหลดและสตรีมมิ่ง:** 
  สามารถเพิ่มกล้อง Webcam (โดยใส่เลข `0`), IP Camera URL หรือทำการอัปโหลดไฟล์วิดีโอได้
- **Decoupled Modern Web Interface:** 
  หน้าจอ UI พัฒนาด้วย React แยกส่วนจาก Backend อย่างชัดเจน โหลดข้อมูลแบบไม่หน่วง สวยงามด้วยธีม Dark Mode และ Glassmorphism

## 🛠️ เทคโนโลยีที่ใช้
- **Frontend:** React, Vite, CSS (Vanilla with Modern Design), Lucide-React
- **Backend/API:** FastAPI, Uvicorn
- **Object Detection:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV, NumPy
- **Matching Algorithm:** Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`)
- **Deep Re-ID:** Torchreid (OSNet)
- **Infrastructure:** Docker & Docker Compose

---

## 🚀 วิธีติดตั้งและรันระบบ (ด้วย Docker - แนะนำ) 🐳

วิธีที่ง่ายที่สุดในการรันระบบคือการใช้ **Docker** ซึ่งจะจัดการเรื่อง Environment ให้ทั้งหมด:

1. เปิด Terminal ในโฟลเดอร์โปรเจกต์ (`C:\PeopleLocation`)
2. พิมพ์คำสั่ง:
   ```bash
   docker-compose up --build -d
   ```
3. เข้าใช้งานระบบ:
   - **หน้าเว็บ UI (Frontend):** เข้าไปที่ `http://localhost:3000`
   - **API (Backend):** ทำงานอยู่ที่ `http://localhost:8899`

---

## 💻 วิธีติดตั้งและรันระบบ (แบบ Manual / Dev Mode)

หากคุณต้องการรันเพื่อแก้ไขโค้ด (Development) สามารถแยกส่วนรันดังนี้:

### ส่วนหลังบ้าน (Backend)
1. เปิด Terminal ใหม่เข้าไปที่โฟลเดอร์ `backend`:
   ```bash
   cd backend
   ```
2. ใช้งาน Virtual Environment (ที่มีอยู่แล้ว):
   ```bash
   .\venv\Scripts\activate
   ```
3. ติดตั้ง Dependencies (หากรันครั้งแรก):
   ```bash
   pip install -r requirements.txt
   ```
4. รัน API Server:
   ```bash
   python main.py
   ```
   *(Backend จะทำงานที่พอร์ต **8899** ตามที่ตั้งไว้ในโค้ดใหม่)*

### ส่วนหน้าบ้าน (Frontend)
1. เปิด Terminal อีกหน้าต่างเข้าไปที่โฟลเดอร์ `frontend`:
   ```bash
   cd frontend
   ```
2. ติดตั้ง Dependencies (ทำเฉพาะครั้งแรก):
   ```bash
   npm install
   ```
3. รัน Development Server:
   ```bash
   npm run dev
   ```
4. เข้าใช้งานตาม URL ที่ระบบแจ้ง (ปกติคือ `http://localhost:5173`)

**หมายเหตุ:** หากรันแบบ Manual อย่าลืมตรวจสอบว่าไฟล์โมเดล `yolov8s.pt` อยู่ในโฟลเดอร์ `backend/` หรือโฟลเดอร์หลักเพื่อให้โค้ดเรียกใช้งานได้ครับ (ถ้าไม่มี ระบบจะดาวน์โหลดให้อัตโนมัติเมื่อรันครั้งแรก)

---

## ▶️ การใช้งานเบื้องต้น (Usage)

เมื่อเปิดหน้าเว็บขึ้นมาเรียบร้อยแล้ว:
1. **อัปโหลดแผนผัง (Global Map):** เลือกภาพแผนผังที่แถบเมนูด้านซ้ายและกดอัปโหลด
2. **เพิ่มกล้อง (Add Camera):** กรอกชื่อและ URL ของกล้อง (หรืออัปโหลดวิดีโอหากต้องการจำลอง)
3. ระบบจะเริ่มดึงภาพและส่งไปประมวลผลที่ Backend ทันที สามารถจัดการกล้องได้จากการ์ดที่ปรากฏขึ้นมาทางขวามือ