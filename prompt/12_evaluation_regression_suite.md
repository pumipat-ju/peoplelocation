# Prompt 12 — สร้าง Evaluation/Regression Suite สำหรับ Cross-camera Global ID

## ปัญหา

การดูเฉพาะ Re-ID Rank-1/mAP ไม่พอสำหรับระบบ Global ID จริง เพราะยังต้องวัด false merge, false split, ID switch และ handoff ระหว่างกล้อง รวมถึง scenario ยาก เช่น คนเสื้อคล้ายกัน เดินสวน ถูกบัง และกล้อง overlap

## เป้าหมาย

สร้าง evaluation pipeline ที่วัดทั้ง Re-ID model และ end-to-end identity association แบบ reproducible เพื่อใช้เทียบก่อน/หลังการแก้แต่ละ prompt

## งานที่ต้องทำ

1. สำรวจ test/evaluation tools ปัจจุบันและ reuse format ที่มีอยู่
2. นิยาม metrics อย่างน้อย:
   - Cross-camera handoff accuracy
   - False merge rate
   - False split / fragmentation rate
   - ID switches
   - IDF1
   - HOTA หรือ association metric ที่ library/project รองรับ
   - Re-ID Rank-1
   - Re-ID mAP
3. รายงานผลแยกตาม:
   - camera pair
   - entry/exit transition
   - scenario type ถ้ามี label
4. เตรียม test scenarios ที่รองรับอย่างน้อย:
   - คนเสื้อคล้ายกัน
   - เดินสวนกัน
   - occlusion
   - คนออกแล้วกลับเข้า
   - overlapping cameras
   - lighting ต่างกัน
5. เพิ่ม event-level trace เพื่อระบุว่า ID switch/false merge เกิดตรง decision ไหน
6. ทำ report machine-readable (เช่น JSON) และ human-readable summary
7. เก็บ config/model/checkpoint/threshold version ใน report ทุกครั้ง
8. ทำ regression thresholds แบบ conservative:
   - ห้าม accept PR ที่ false merge แย่ลงมากแม้ Rank-1 ดีขึ้น
9. ถ้ามี unit/integration tests อยู่แล้ว ให้เพิ่ม synthetic deterministic cases สำหรับ assignment logic โดยไม่ต้องพึ่ง GPU
10. สร้าง command/documented workflow สำหรับ:
   - baseline ก่อนแก้
   - evaluation หลังแก้
   - compare reports

## Acceptance criteria

- วัด end-to-end Global ID ได้ ไม่ใช่เฉพาะ embedding retrieval
- report เปรียบเทียบสอง run ได้
- false merge เป็น first-class metric
- มี deterministic regression cases สำหรับ matching logic
- output ระบุ model/config/version ที่ใช้ครบพอ reproduce ได้

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
