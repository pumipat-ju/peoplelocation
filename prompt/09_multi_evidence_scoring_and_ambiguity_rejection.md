# Prompt 09 — Matching หลายหลักฐาน + Top-1/Top-2 Margin + Reject Option

## ปัญหา

Similarity จาก OSNet เพียงค่าเดียวไม่เพียงพอ โดยเฉพาะเมื่อคนแต่งตัวคล้ายกัน ระบบต้องรวมหลายหลักฐานและต้องสามารถตอบว่า “ยังไม่มั่นใจ” แทนการเลือก candidate ที่คะแนนสูงสุดเสมอ

## เป้าหมาย

ทำ scoring แบบ multi-evidence ที่ conservative และแยก **hard gates** ออกจาก **soft scores** ชัดเจน พร้อม ambiguity rejection

## งานที่ต้องทำ

1. หา scoring logic ปัจจุบันทั้งหมด
2. แยก hard gate ก่อน scoring เช่น:
   - impossible topology
   - impossible travel time
   - incompatible simultaneous presence
3. สำหรับ candidate ที่ผ่าน hard gate คำนวณ component score อย่างน้อยจากข้อมูลที่ระบบมีจริง:
   - appearance
   - entry/exit compatibility
   - travel-time compatibility
   - floorplan/location compatibility
   - track/tracklet quality
4. Normalize score components ให้อยู่ในสเกลที่เปรียบเทียบ/ถ่วงน้ำหนักได้
5. ห้ามให้ low-quality appearance มีน้ำหนักเท่ากับ high-quality tracklet
6. เพิ่ม acceptance threshold
7. เพิ่ม top-1 vs top-2 ambiguity margin
8. ถ้า:
   - top-1 ต่ำกว่า acceptance
   - margin ต่ำเกินไป
   - evidence ขัดแย้งกัน
   ให้ reject/defer ไม่ force match
9. แยก threshold/weights เป็น config และบันทึกเหตุผลตอน commit/reject
10. diagnostics ต่อ decision ต้องมี:
   - candidate GIDs
   - component scores
   - total score
   - top-1/top-2 margin
   - gate failures
   - assignment source
11. เพิ่ม tests สำหรับคน appearance คล้ายกันแต่เส้นทางต่างกัน และกรณีคะแนน top-1/top-2 ใกล้กัน

## Acceptance criteria

- similarity อย่างเดียวไม่สามารถทำให้ impossible match ผ่านได้
- ambiguity มี explicit reject/defer behavior
- decision สามารถ audit ย้อนหลังได้จาก component scores/reasons
- threshold/weight เป็น config ไม่กระจาย hard-code
- tests ครอบคลุม false-merge-prone scenarios

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
