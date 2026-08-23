# Prompt 10 — เพิ่ม Identity State Machine: PROVISIONAL / ACTIVE / DORMANT / EXPIRED

## ปัญหา

การ commit Global ID เร็วเกินไปจากเฟรมแรกทำให้ identity ผิดได้ง่าย และ lifecycle ระหว่าง “เพิ่งพบ”, “กำลังเห็น”, “หายไป”, “หมดอายุ” ยังไม่ชัดเจนพอสำหรับ cross-camera Re-ID

## เป้าหมาย

ทำ state machine ที่ explicit และ deterministic เพื่อให้ระบบรอหลักฐานก่อนยืนยัน identity และจัดการการหาย/กลับเข้ามาได้ถูกต้อง

## งานที่ต้องทำ

1. สำรวจ lifecycle ของ Global ID ปัจจุบัน
2. นิยาม state อย่างน้อย:
   - `PROVISIONAL`
   - `ACTIVE`
   - `DORMANT`
   - `EXPIRED`
3. ระบุ transition conditions อย่างชัดเจน เช่น:
   - new tracklet → PROVISIONAL
   - evidence/quality เพียงพอ → ACTIVE
   - track หาย/exit confirmed → DORMANT
   - TTL หมด → EXPIRED
   - cross-camera recovery → ACTIVE
4. ห้าม commit permanent gallery จาก PROVISIONAL ที่ evidence ยังไม่พอ
5. กำหนด behavior เมื่อ local track ID เปลี่ยนในกล้องเดิมแต่บุคคลน่าจะเป็นคนเดิม
6. กำหนด behavior สำหรับ overlapping cameras ที่ GID เดียวอาจ ACTIVE มากกว่าหนึ่งกล้องได้เฉพาะ rule ที่อนุญาต
7. แยก state transition logic ออกจาก ad-hoc if statements ให้ test ได้
8. บันทึก reason/timestamp ทุก transition
9. แสดง state counts และ recent transitions ใน diagnostics
10. เพิ่ม tests สำหรับ lifecycle หลัก รวมทั้ง:
   - new → provisional → active
   - active → dormant → active cross-camera
   - dormant → expired
   - ambiguous candidate คง provisional ไม่ force active

## Acceptance criteria

- state transitions เป็น deterministic และมีเหตุผลบันทึก
- provisional identity ไม่ poison permanent gallery
- dormant recovery รักษา GID เดิมเมื่อ evidence ผ่าน
- expired identity ไม่ถูก reuse โดย logic ปกติโดยไม่มี policy รองรับ
- lifecycle tests ผ่าน

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
