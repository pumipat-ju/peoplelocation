# Prompt 07 — เพิ่ม Entry/Exit Zone, Camera Topology และ Travel-time Gate

## ปัญหา

Appearance similarity อย่างเดียวทำให้คนเสื้อคล้ายกันถูก match ผิด ระบบต้องรู้ว่าคนออกจากกล้องไหน ตรง zone ใด สามารถไปโผล่กล้องไหน ตรง zone ใด และต้องใช้เวลาประมาณเท่าไร

## เป้าหมาย

สร้าง candidate gating เชิงพื้นที่และเวลา ก่อนส่งเข้า Re-ID matching เพื่อ **ตัด candidate ที่เป็นไปไม่ได้ออกตั้งแต่ต้น**

## งานที่ต้องทำ

1. อ่าน calibration/floorplan/homography และ camera configuration ปัจจุบัน
2. ออกแบบ representation สำหรับ:
   - entry zones
   - exit zones
   - overlap zones
   - camera-to-camera topology
   - allowed source zone → destination zone
   - min/max travel time
3. เก็บ config แบบ persist ได้และ version/validate ได้
4. ถ้ามี calibration UI ให้เพิ่มวิธีกำหนด zone โดยไม่ทำลาย calibration เดิม
5. เมื่อ tracklet ออกจากกล้อง ให้บันทึก:
   - exit camera
   - exit zone
   - exit timestamp
   - floorplan position ถ้ามี
6. เมื่อ tracklet ใหม่เข้าอีกกล้อง ให้หา candidate เฉพาะ transition ที่ topology อนุญาต
7. Hard reject เมื่อ:
   - camera pair ไปถึงกันไม่ได้
   - zone transition ไม่อนุญาต
   - travel time ต่ำกว่าหรือสูงกว่า window ที่กำหนด
8. รองรับ camera overlap แยกจาก non-overlap transition
9. ห้ามใช้ appearance score เพื่อ override impossible transition
10. เพิ่ม endpoint/config diagnostics ที่ดู transition rules ได้
11. เพิ่ม tests เช่น:
   - Cam1 Exit-Door → Cam2 Entry-Left, 2–12s = allowed
   - same route ที่ 0.5s = rejected
   - camera pair ไม่มี edge = rejected

## Acceptance criteria

- candidate ที่ผิดเส้นทางไม่ถูกส่งเข้าการ matching ปกติ
- travel time gate ใช้ canonical timestamp
- overlap camera ถูกกำหนด explicit
- config persistence/validation ใช้งานได้
- tests ครอบคลุม allowed และ impossible transitions

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
