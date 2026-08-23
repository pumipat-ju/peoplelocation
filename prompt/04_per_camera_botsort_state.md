# Prompt 04 — แยก BoT-SORT State ต่อกล้อง

## ปัญหา

ปัจจุบันมีความเสี่ยงจากการเรียก `model.track(..., persist=True)` ผ่าน model/tracker กลางร่วมกันหลายกล้อง ทำให้ state ของ local tracker อาจปะปนข้าม camera และ Local Track ID มีโอกาสได้รับผลจาก frame ของกล้องอื่น

## เป้าหมาย

ให้แต่ละกล้องมี tracker state ของตัวเองอย่างเด็ดขาด ขณะที่ detector weights สามารถแชร์ได้ถ้าปลอดภัยและเหมาะสม

## งานที่ต้องทำ

1. Trace video processing path ทุกกล้องจนถึง `model.track(... persist=True)`
2. ระบุว่า object ใดเก็บ BoT-SORT persistent state จริง
3. แก้ architecture ให้มี tracker context/instance แยกต่อ `camera_id`
4. ห้ามให้ frame ของ Camera A update state ของ Camera B
5. รองรับ lifecycle:
   - create tracker เมื่อเพิ่มกล้อง/source
   - reset tracker เมื่อ source ถูก reset/restarted ตามสมควร
   - cleanup เมื่อ camera ถูก remove
6. ตรวจ concurrency/thread safety ถ้าหลายกล้องประมวลผลพร้อมกัน
7. แยกความหมาย:
   - `local_track_id` unique เฉพาะใน camera
   - `global_id` จัดการโดย Re-ID layer
8. ห้ามใช้ local track id ข้ามกล้องเป็น identity evidence โดยตรง
9. เพิ่ม diagnostics ใน `/api/status` หรือ debug endpoint:
   - camera name/id
   - tracker instance/state identifier
   - active local tracks
10. เพิ่ม regression test ที่ feed sequence สองกล้องสลับกัน และพิสูจน์ว่า tracker state ไม่ contaminate กัน

## Acceptance criteria

- แต่ละ camera มี independent tracker state
- local track IDs ของกล้องหนึ่งไม่เกิดจาก history ของอีกกล้อง
- restart/reset camera หนึ่งไม่ reset tracker ของกล้องอื่น
- concurrency ไม่ทำให้ state race
- regression test ครอบคลุม multi-camera isolation

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
