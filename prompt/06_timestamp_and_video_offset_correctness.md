# Prompt 06 — ทำให้ Timestamp และ Video Offset ใช้ได้ถูกต้องกับ Cross-camera Re-ID

## ปัญหา

Cross-camera matching ที่ใช้ travel time จะเชื่อถือไม่ได้ถ้า timestamp ของแต่ละ source ไม่ได้อยู่บน timeline เดียวกัน หรือ video offset ถูกใช้ไม่สอดคล้องกัน

## เป้าหมาย

กำหนด canonical timestamp สำหรับทุก detection/tracklet ให้เปรียบเทียบข้ามกล้องได้ และทำให้ offset ของวิดีโอมี semantics เดียวตลอดระบบ

## งานที่ต้องทำ

1. Trace เวลาตั้งแต่ video/live source → frame → detection → tracklet → identity matching
2. ระบุ timestamp แต่ละชนิดที่มีอยู่และหน่วยที่ใช้
3. สร้าง canonical event time สำหรับ cross-camera logic
4. สำหรับ uploaded videos ให้รวม:
   - frame timestamp / FPS-derived time
   - playback clock
   - per-camera offset
   อย่างชัดเจนและไม่ apply offset ซ้ำ
5. สำหรับ live camera ให้ใช้ clock ที่เหมาะสมและหลีกเลี่ยงการใช้ processing completion time เป็น event timeโดยไม่จำเป็น
6. Normalize หน่วยเวลาให้ชัดเจน เช่น seconds หรือ milliseconds แบบเดียว
7. ป้องกัน negative/ย้อนเวลา event จาก seek, restart หรือ reconnect
8. ระบุ behavior เมื่อ:
   - video เริ่มไม่พร้อมกัน
   - source lag
   - source EOF
   - camera reconnect/reset
9. เพิ่ม diagnostics ต่อ camera:
   - source time
   - canonical time
   - configured offset
   - drift/lag ถ้ามี
10. เพิ่ม deterministic test สองกล้องที่กำหนด offset แล้วตรวจ travel-time difference

## Acceptance criteria

- detection/tracklet จากต่างกล้องเปรียบเทียบเวลาได้บน timeline เดียว
- offset ถูก apply ครั้งเดียวและตรวจสอบได้
- travel-time gate ใช้ event time ไม่ใช่ processing latency
- test สามารถพิสูจน์ expected delta ระหว่างกล้องได้

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
