# Prompt 02 — ทำให้ OSNet Architecture ตรงกันทั้ง Train / Eval / Production

## ปัญหา

ปัจจุบันมี architecture ไม่ตรงกัน: backend เคยระบุ `osnet_x0_25` ขณะที่ training pipeline และ baseline ใช้ `osnet_x1_0` การ train ด้วย architecture หนึ่งแล้ว production instantiate อีก architecture ทำให้ checkpoint โหลดไม่ได้หรือได้ผลที่ไม่ตรงกับ evaluation

## เป้าหมาย

ให้มี **single source of truth** สำหรับ OSNet architecture และให้ train, evaluation, threshold analysis และ production ใช้ architecture/checkpoint metadata ที่สอดคล้องกัน

## งานที่ต้องทำ

1. สำรวจทุกจุดที่กำหนดชื่อ OSNet architecture ใน repository
2. สรุปว่าไฟล์ใดใช้ `osnet_x0_25`, `osnet_x1_0` หรือชื่ออื่น
3. เลือก architecture production เดียวโดยยึดของที่ training/evaluation ใช้งานจริงในโปรเจกต์ปัจจุบัน
4. ย้ายค่าที่ซ้ำกันไปอยู่ config/constant/metadata ที่เหมาะสม แทนการ hard-code หลายที่
5. ตอน save checkpoint ให้บันทึก metadata อย่างน้อย:
   - architecture
   - embedding dimension
   - training dataset/version ถ้ามี
   - epoch/best metric
6. ตอน load checkpoint ใน production/evaluation ให้ตรวจ metadata กับ architecture ที่กำลัง instantiate
7. ถ้า architecture mismatch ให้ fail ด้วยข้อความชัดเจน แทนการ discard keys จำนวนมากแล้วรันต่อโดยไม่รู้ตัว
8. ตรวจ training script, evaluation script, threshold analysis และ backend ให้ใช้ preprocessing/normalization ที่สอดคล้องกันด้วย
9. เพิ่ม test สำหรับ:
   - checkpoint architecture ตรง → load ได้
   - architecture ไม่ตรง → ถูก reject พร้อม error ที่เข้าใจได้
10. อัปเดต status/log ให้แสดง architecture ที่ runtime ใช้จริง

## Acceptance criteria

- ไม่มี config ที่ขัดกันระหว่าง train/eval/production
- checkpoint ที่สร้างจาก architecture ที่เลือกสามารถโหลดใน production ได้ตรง ๆ
- architecture mismatch ถูกตรวจพบก่อน inference
- preprocessing หลักของ train/eval/production สอดคล้องกัน
- มีหลักฐานจาก test หรือ smoke test

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
