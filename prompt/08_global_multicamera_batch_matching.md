# Prompt 08 — เปลี่ยน Hungarian Matching จาก Per-camera เป็น Global Multi-camera Batch

## ปัญหา

ปัจจุบันแม้ใช้ Hungarian assignment แต่การ assign ยังมีลักษณะทำเป็น batch ภายในกล้องเดียว ทำให้ไม่บังคับ one-to-one identity assignment ในระดับหลายกล้องจริง และกล้องต่าง ๆ อาจแย่ง GID เดียวกันโดยไม่ได้ตัดสินพร้อมกัน

## เป้าหมาย

รวบรวม candidate tracklets จากหลายกล้องใน time window เดียวกัน แล้วทำ assignment แบบ global one-to-one พร้อม hard constraints

## งานที่ต้องทำ

1. Trace current assignment pipeline และตำแหน่งที่ Hungarian ถูกเรียก
2. แยกขั้นตอนให้ชัด:
   - collect mature/new tracklets
   - build candidate GIDs
   - apply hard gates
   - build global score/cost matrix
   - Hungarian assignment
   - reject invalid/ambiguous pairs
   - commit assignments
3. สร้าง batching/rendezvous window ที่เล็กพอสำหรับ realtime แต่เปิดโอกาสให้หลายกล้องเข้ามาตัดสินพร้อมกัน
4. ห้าม block video processing โดยไม่จำเป็น; ถ้าใช้ queue/coordinator ต้องมี bounded behavior
5. enforce one-to-one:
   - tracklet หนึ่งได้ GID เดียว
   - GID หนึ่งไม่ถูก assign ให้ incompatible simultaneous tracklets
6. อนุญาต same GID พร้อมกันเฉพาะกรณีกล้อง overlap ที่ rule อนุญาต
7. unmatched tracklet ต้องมีทางไป PROVISIONAL/new identity ไม่ถูก force ให้ match
8. unmatched identity ต้องคง state ได้ตาม lifecycle
9. log candidate matrix/selected assignment แบบ debug ที่เปิดปิดได้
10. เพิ่ม deterministic test อย่างน้อย 2–3 cameras ที่ per-camera greedy/Hungarian จะให้ผลผิด แต่ global batch ให้ assignment ถูก

## Acceptance criteria

- assignment decision พิจารณาหลายกล้องใน window เดียวกัน
- one-to-one constraint ถูก enforce ระดับ global
- impossible simultaneous presence ถูก reject
- unmatched tracklet ไม่ถูกบังคับ match
- regression test แสดงความต่างจาก per-camera assignment ได้

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
