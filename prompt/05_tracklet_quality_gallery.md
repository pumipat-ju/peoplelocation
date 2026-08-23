# Prompt 05 — เปลี่ยนจาก Frame-based Re-ID เป็น Tracklet + Quality Gallery

## ปัญหา

การตัดสิน identity จาก embedding ของภาพล่าสุดหรือ update gallery บ่อยเกินไปทำให้ภาพเบลอ ถูกบัง หลุดขอบ หรือคนซ้อนกันสามารถทำให้ Global ID ผิดและทำให้ gallery ถูก poison ต่อเนื่อง

## เป้าหมาย

สร้างตัวแทน appearance จาก **หลายภาพคุณภาพดีใน tracklet** และควบคุมการ update gallery อย่าง conservative

## งานที่ต้องทำ

1. หา flow ปัจจุบันที่สร้าง crop → embedding → update identity gallery
2. เพิ่ม Tracklet Builder ต่อ `(camera_id, local_track_id)`
3. เก็บ metadata ของ detection ใน tracklet เช่น:
   - timestamp/frame index
   - bbox
   - detector confidence
   - occlusion/overlap
   - border clipping
   - crop size
   - blur/quality score
4. เพิ่ม Quality Filter ก่อนสร้างหรือก่อนนำ embedding เข้า gallery
5. ห้าม gallery update จาก:
   - detection confidence ต่ำ
   - crop เล็กเกิน
   - blur มาก
   - ถูกบัง/overlap สูง
   - bbox หลุดขอบมาก
6. สะสม embeddings หลายเฟรมและสร้าง tracklet prototype ที่ robust เช่น normalized mean/medoid หรือวิธีที่เหมาะกับโค้ดปัจจุบัน
7. ป้องกัน near-duplicate embeddings กิน gallery จนหมด
8. จำกัด gallery size และเก็บตัวอย่างที่ diverse
9. รองรับ gallery แยกตาม camera และถ้าโค้ดรองรับให้เก็บ view/pose bucket โดยไม่ over-engineer
10. Re-ID decision ควรถูก trigger ตอนมี track ใหม่/tracklet mature/track หาย/ข้ามกล้อง มากกว่าตัดสิน identity ใหม่ทุกเฟรม
11. เพิ่ม diagnostics:
   - accepted/rejected gallery update
   - rejection reason
   - tracklet sample count
   - prototype quality
12. เพิ่ม tests สำหรับ gallery poisoning และ quality gate

## Acceptance criteria

- ภาพคุณภาพต่ำไม่สามารถ update identity gallery
- tracklet ที่มีหลายภาพสร้าง prototype ได้
- Global ID ไม่ถูก re-decide แบบอิสระทุกเฟรม
- gallery มีขนาดจำกัดและไม่เต็มไปด้วยภาพซ้ำ
- มี test ที่พิสูจน์ว่าภาพ occluded/blurred ถูก reject

## กติกาการทำงานร่วมกัน

- อ่านโค้ดปัจจุบันและ flow ที่เกี่ยวข้องให้ครบก่อนแก้ ห้ามเดาโครงสร้าง repository จาก prompt อย่างเดียว
- แก้เฉพาะปัญหาในไฟล์ prompt นี้ก่อน อย่า refactor ใหญ่หรือพ่วง feature จาก prompt อื่นโดยไม่จำเป็น
- รักษา API/behavior เดิมที่ไม่เกี่ยวข้องให้มากที่สุด
- ถ้า path หรือชื่อ class/function ใน repository จริงต่างจากที่ระบุ ให้ยึดโค้ดจริงเป็นหลักและอธิบาย mapping ที่พบ
- เพิ่ม logging/diagnostic ที่จำเป็นต่อการพิสูจน์ว่าแก้ปัญหาได้ แต่หลีกเลี่ยง log spam รายเฟรม
- เพิ่มหรือปรับ test สำหรับ behavior ที่แก้ และรัน test ที่เกี่ยวข้องจริง
- ห้ามรายงานว่า test ผ่านถ้ายังไม่ได้รัน
- ห้ามลด threshold หรือผ่อน gate แบบสุ่มเพื่อทำให้ตัวเลขดูดี
- ให้ความสำคัญกับการลด **false merge** มากกว่า false split เพราะการรวมคนสองคนเป็น Global ID เดียวทำให้ identity memory เสียต่อเนื่อง
- เมื่อจบงาน ให้สรุป:
  1. root cause ที่พบจริง
  2. ไฟล์ที่แก้
  3. สิ่งที่เปลี่ยน
  4. test/คำสั่งที่รันและผล
  5. diagnostic ที่ควรตรวจตอนรันจริง
  6. ความเสี่ยงหรือสิ่งที่ยังไม่ได้แก้
