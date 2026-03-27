# AI CCTV Monitoring & Multi-Camera Tracking System

ระบบจัดการและติดตามบุคคลจากกล้องวงจรปิด (Person Tracking & Re-Identification) ข้ามกล้องแบบเรียลไทม์ พร้อมการแสดงผลบนแผนที่ (Bird's Eye View/Floorplan) พัฒนาด้วย **YOLOv8** และ **FastAPI**

## 🌟 ความสามารถหลัก (Features) เซตล่าสุดใน `main_ReID.py`

- **Multi-Camera Person Re-Identification (Re-ID):** 
  ติดตามการเคลื่อนที่ของบุคคลข้ามกล้องได้ โดยระบบจะให้ Global ID เดียวกันสำหรับบุคคลเดียวกัน แม้จะเดินหลุดกรอบจากกล้องหนึ่งไปโผล่อีกกล้องหนึ่งก็ตาม
- **รองรับ OSNet และ Lightweight Feature Extraction:** 
  สามารถใช้ `torchreid` เพื่อดึงคุณลักษณะที่แม่นยำสูง (OSNet) ได้ หรือหากไม่ได้ติดตั้งไว้ ระบบจะปรับไปใช้งาน Lightweight Feature Extractor (HSV Color + Shape) ให้โดยอัตโนมัติ
- **Floorplan & Homography Mapping:** 
  สามารถอัปโหลดรูปแผนผังอาคาร (Floorplan) เข้าสู่ระบบ และทำการสอบเทียบ (Calibration) มุมมองภาพจากกล้องแต่ละตัว ให้สอดคล้องกับพิกัดบนแผนผัง (Bird's Eye View) ทำให้สามารถเห็นตำแหน่งคนบนแผนที่ได้แบบเรียลไทม์
- **Occlusion Handling:** 
  ระบบมีกลไกป้องกันปัญหาคนเดินบังกัน (Occlusion) ทำให้ ID ไม่เกิดการสลับกันมั่วในขณะที่คนเดินสวนกันหน้ากล้อง
- **รองรับวิดีโออัปโหลดและสตรีมมิ่ง:** 
  สามารถเพิ่มกล้อง Webcam (โดยใส่เลข `0`), IP Camera URL หรือทำการอัปโหลดไฟล์วิดีโอ (MP4, AVI, MKV) เพื่อจำลองการทำงานของกล้องวงจรปิดได้ผ่านหน้าเว็บโดยตรง
- **Web Interface:** 
  จัดการทั้งหมดได้ง่ายผ่านเว็บเบราว์เซอร์

## 🛠️ ไฮไลท์เทคโนโลยี
- **Backend/API:** `FastAPI`, `Uvicorn`, `Jinja2`
- **Object Detection:** `YOLOv8` (Ultralytics)
- **Computer Vision:** `OpenCV`, `NumPy`
- **Matching Algorithm:** Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`)
- **Deep Re-ID:** `Torchreid` (OSNet)

---

## 🚀 วิธีติดตั้ง (Installation)

1. Clone โปรเจกต์นี้ลงเครื่องของคุณ

2. สร้าง Virtual Environment (แนะนำ):
   ```bash
   python -m venv venv
   # เข้าใช้งาน venv (สำหรับ Windows)
   venv\Scripts\activate
   # เข้าใช้งาน venv (สำหรับ Mac / Linux)
   source venv/bin/activate
   ```

3. ติดตั้ง Dependencies พื้นฐาน:
   ```bash
   pip install -r requirements.txt
   ```

4. **[ทางเลือกเสริม]** ติดตั้ง `torchreid` เพื่อการจำแนกบุคคลที่ดีขึ้นด้วยโมเดล OSNet:
   กรุณาอ้างอิงวิธีการติดตั้งบน GitHub ของ [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) (หากข้ามขั้นตอนนี้ไประบบจะใช้ Fallback แบบ Lightweight แทน)

---

## ▶️ การใช้งาน (Usage)

1. รันเซิร์ฟเวอร์ด้วยเวอร์ชันล่าสุด (`main_ReID.py`):
   ```bash
   python main_ReID.py
   ```
   *(หรือสามารถรันด้วย uvicorn ได้โดยตรง: `uvicorn main_ReID:app --host 0.0.0.0 --port 8000`)*

2. เปิดเว็บเบราว์เซอร์และเข้าไปที่:
   ```text
   http://localhost:8000
   ```

3. **ขั้นตอนการเริ่มต้นบนหน้าเว็บ:**
   - **Step 1:** อัปโหลดไฟล์แผนผัง (Floorplan) ของพื้นที่
   - **Step 2:** กดช่อง **"เพิ่มกล้อง"** (ถ้าเป็น Webcam ใส่ `0` หรือถ้ามีไฟล์วิดีโอให้ใช้เมนู "อัปโหลดวิดีโอ" แทน)
   - **Step 3:** ทำการเลือกจุด 4 จุดบนกล้อง (หน้าต่างแสดงภาพ) และ 4 จุดที่อ้างอิงตรงกันบนหน้า Floorplan แล้วกดบันทึกการ Calibration
   - **Step 4:** ระบบจะเริ่ม Tracking ตรวจจับบุคคล ดึงลักษณะ (Feature Extraction) และพลอตตำแหน่งลงบนแผนผังให้โดยอัตโนมัติ