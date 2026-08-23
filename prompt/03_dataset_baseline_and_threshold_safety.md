# Prompt 03 — แก้ปัญหา Dataset น้อยและ Threshold ที่ยังเชื่อถือไม่ได้

## ปัญหา

ข้อมูลสถานที่จริงปัจจุบันมีเพียงประมาณ 3 identities และ diagnostic baseline เคยได้ Rank-1 = 0 อีกทั้ง similarity เฉลี่ยของคนเดียวกันประมาณ `0.5247` แต่คนละคนประมาณ `0.5433` ซึ่งหมายความว่า feature distribution ชุดนี้ยังไม่เหมาะกับการตั้ง threshold แบบง่าย ๆ

## เป้าหมาย

ทำให้ระบบ **ไม่ใช้ threshold ที่ไม่มีหลักฐานรองรับ** และสร้าง pipeline สำหรับวัด baseline/เลือก threshold จาก validation data อย่าง reproducible พร้อมเตือนเมื่อ dataset เล็กเกินไป

## งานที่ต้องทำ

1. อ่าน dataset preparation, evaluation และ threshold analysis pipeline ปัจจุบัน
2. ตรวจ split ว่า train/validation/test แยก identity ไม่รั่วกัน
3. เพิ่ม validation ของ dataset:
   - จำนวน identities
   - จำนวน cameras
   - จำนวนภาพต่อ identity
   - identities ที่มี cross-camera positives
   - query ที่ไม่มี positive ใน gallery
4. ถ้าชุด validation เล็กเกินไป ให้แสดง warning ชัดเจนและห้ามสรุป threshold ว่า production-ready
5. ทำ threshold analysis จาก validation data โดยรายงานอย่างน้อย:
   - same-person distribution
   - different-person distribution
   - false accept/false merge rate
   - false reject/false split rate
   - precision/recall หรือ ROC/PR ที่เหมาะสม
6. แยก threshold สำหรับ use case ที่ต่างกันถ้าจำเป็น เช่น active match vs dormant recovery แต่ห้ามสร้างค่าจากการเดา
7. เพิ่ม conservative default behavior เมื่อ model/dataset ยังไม่น่าเชื่อถือ:
   - reject ambiguous match
   - ไม่ commit จาก similarity เพียงอย่างเดียว
8. ทำ output artifact เช่น JSON report ที่มี dataset stats + metrics + threshold candidates + warning
9. เพิ่มคำสั่ง reproducible สำหรับรัน baseline ก่อน/หลัง fine-tune
10. อย่าพยายาม “แก้ Rank-1” ด้วยการใช้ test set ปรับ threshold

## Acceptance criteria

- pipeline ตรวจ data leakage และ dataset insufficiency ได้
- threshold ทุกค่าที่เสนอมีที่มาจาก validation metrics
- report แสดง clearly ว่าชุดข้อมูลปัจจุบันเพียงพอหรือยัง
- ระบบไม่บังคับ match เมื่อ similarity distribution แยกคนไม่ได้
- สามารถรัน baseline ซ้ำได้ด้วยคำสั่งเดียวหรือขั้นตอนที่ชัดเจน

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
