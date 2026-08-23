# Prompt 01 — ทำให้ OSNet Production โหลด checkpoint จริง

## ปัญหา

ระบบ production มีการตั้งค่า OSNet แต่จากข้อมูลปัจจุบัน checkpoint `osnet_x0_25_market1501.pt` ไม่พบ ทำให้ระบบ fallback ไปใช้ feature สี/รูปร่างแทน ส่งผลให้การแยกคนที่แต่งตัวคล้ายกันอ่อนลงมาก จุดตั้งค่าที่เคยพบอยู่บริเวณ `backend/main.py` แถว configuration ของ Re-ID model

## เป้าหมาย

ทำให้ production โหลด OSNet checkpoint ที่ตั้งใจใช้ได้จริงทั้งตอนรัน local และใน Docker/Deployment และทำให้ผู้ดูระบบตรวจได้ชัดเจนว่า ณ runtime ใช้โมเดลอะไรอยู่ ไม่ใช่ fallback แบบเงียบ ๆ

## งานที่ต้องทำ

1. Trace initialization path ของ Re-ID/OSNet ตั้งแต่ config → model creation → checkpoint loading → inference
2. ตรวจ path resolution ให้ทำงานถูกทั้ง:
   - Windows/local
   - Docker/container
   - working directory ที่ต่างกัน
3. ห้าม hard-code absolute path เฉพาะเครื่องเดียว
4. รองรับ checkpoint path ผ่าน config/environment ที่ชัดเจน
5. ตรวจว่า checkpoint file ถูก copy/mount เข้า container จริงใน deployment configuration
6. Validate checkpoint ก่อนเริ่ม inference:
   - file exists
   - architecture ตรงกับ model ที่ instantiate
   - state dict โหลดได้
   - embedding inference ทำงาน
7. ถ้าโหลดไม่ได้:
   - log error ที่เห็นสาเหตุจริง
   - แสดงสถานะ degraded/fallback ชัดเจน
   - อย่าทำให้ผู้ใช้เข้าใจผิดว่า OSNet ทำงานอยู่
8. เพิ่มข้อมูลใน `/api/status` อย่างน้อย:
   - Re-ID enabled/disabled
   - model architecture
   - checkpoint path/name
   - checkpoint loaded = true/false
   - device
   - fallback active = true/false
   - embedding dimension ถ้าทราบ
9. เพิ่ม startup log เพียงครั้งเดียวที่สรุป Re-ID runtime configuration
10. เพิ่ม test หรือ smoke test ที่พิสูจน์ว่า checkpoint ถูก load และสามารถสร้าง embedding จาก crop ตัวอย่างได้

## Acceptance criteria

- เมื่อ checkpoint ถูกต้อง ระบบต้องไม่ fallback
- `/api/status` ต้องบอกได้ทันทีว่า OSNet โหลดสำเร็จหรือไม่
- container ต้องมองเห็น checkpoint จาก path ที่ config ระบุ
- ถ้า checkpoint หาย ต้องเห็นสาเหตุชัดเจนและสถานะ fallback ต้องถูกต้อง
- test/smoke test ต้องตรวจได้ว่า embedding ถูกสร้างด้วย OSNet จริง

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
