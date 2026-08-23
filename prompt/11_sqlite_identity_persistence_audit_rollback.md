# Prompt 11 — ทำ SQLite Identity Memory ให้ Persist, Audit และกู้คืนได้

## ปัญหา

Global Identity Memory ต้องอยู่รอดหลัง restart และต้องตรวจย้อนหลังได้ว่า match/reject เพราะอะไร รวมถึงต้องลดความเสียหายจาก gallery poisoning เมื่อเกิด false match

## เป้าหมาย

ทำ persistence layer ที่เก็บ identity lifecycle และ evidence สำคัญอย่างปลอดภัย โดย runtime state กับ SQLite สอดคล้องกัน

## งานที่ต้องทำ

1. สำรวจ SQLite schema/store ปัจจุบันก่อนแก้
2. ออกแบบ migration แบบ backward-compatible ถ้าต้องเพิ่ม column/table
3. persist อย่างน้อย:
   - global_id
   - state
   - timestamps
   - per-camera gallery/prototype metadata
   - tracklet summary
   - latest floorplan position
   - entry/exit zone
   - camera history
   - candidate/decision scores
   - match/reject reason
4. แยก identity snapshot/current state ออกจาก match event/audit log ตามสมควร
5. ตอน startup:
   - load valid identities
   - restore GID counter/sequence อย่างปลอดภัย
   - handle corrupted/incompatible record แบบไม่ crash ทั้งระบบ
6. ระบุ transaction boundaries เพื่อไม่ให้ RAM state กับ DB แตกต่างกันเมื่อ process ล้ม
7. ป้องกัน duplicate GID หลัง restart
8. เพิ่ม cleanup/TTL สำหรับ EXPIRED data โดยไม่ลบ audit ที่ต้องการเก็บทันที
9. เพิ่ม mechanism สำหรับ rollback/quarantine ของ gallery update ที่มาจาก match ที่ถูกตรวจว่าผิด โดยออกแบบให้ minimal และ audit ได้
10. ห้าม pickle object ที่ผูกกับ code version แบบเปราะบางถ้ามี format ที่ stable กว่า
11. เพิ่ม tests:
   - save → restart simulation → restore
   - GID sequence ไม่ชน
   - dormant identity restore ได้
   - failed transaction ไม่ทิ้ง half-committed identity

## Acceptance criteria

- restart แล้ว identity memory ที่ควรอยู่ยังกลับมา
- ไม่มี duplicate GID จาก persistence
- audit log อธิบาย decision สำคัญได้
- schema upgrade ปลอดภัยกับ DB เดิม
- persistence tests ผ่าน

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
