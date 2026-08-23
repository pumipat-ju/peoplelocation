───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• เป้าหมายควรเป็น: “Global ID แบบไม่รู้ชื่อบุคคล” ที่จำจากรูปลักษณ์และเส้นทางในช่วงเวลาหนึ่ง ไม่ควรใช้ similarity อย่างเดียว เพราะคนแต่งตัวคล้ายกันจะถูกรวม ID ง่าย แผนสุดท้ายจะให้ appearance เป็นเพียงหนึ่งหลักฐาน
  แล้วบังคับผ่านเวลา, ประตูกล้อง, ตำแหน่ง และเงื่อนไขห้ามคนเดียวกันอยู่สองที่พร้อมกัน

• “จำคน” ด้วย Global ID แบบหลายหลักฐาน ไม่ใช้ความคล้ายของภาพเพียงอย่างเดียว และควรตัดสินจากชุดภาพของคนหนึ่งคนหรือ tracklet แทนการตัดสินใหม่ทุกเฟรม

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  ## ภาพรวมระบบที่เสนอ

  Camera A / B / C / etc.
        ↓
  YOLO Person Detection
        ↓
  BoT-SORT แยกต่อกล้อง
        ↓
  Tracklet Builder
  รวมภาพคนเดียวกันหลายเฟรม
        ↓
  Quality Filter
  ตัดภาพเบลอ / ถูกบัง / หลุดขอบ
        ↓
  OSNet Embedding Gallery
        ↓
  Candidate Gate
  เวลา + Entry/Exit zone + Floorplan + Camera topology
        ↓
  Global Hungarian Matching
  พร้อม Reject กรณีไม่มั่นใจ
        ↓
  Global Identity Memory
  ACTIVE / DORMANT / PROVISIONAL
        ↓
  SQLite Persistence

  ## สิ่งที่พบในระบบปัจจุบัน

  ระบบมีพื้นฐานที่ดีอยู่แล้ว เช่น YOLO, BoT-SORT, OSNet interface, gallery, Hungarian assignment, occlusion handling และ homography

  แต่มีประเด็นสำคัญที่ต้องแก้ก่อนจูนความแม่น:

  1. OSNet production ยังไม่ได้ทำงานจริง

  จาก log ของ container พบว่าไม่มีไฟล์ osnet_x0_25_market1501.pt จึง fallback ไปใช้เพียงสีและรูปร่าง ระบบจึงแยกคนเสื้อคล้ายกันได้ยากมาก

  2. สถาปัตยกรรมโมเดลยังไม่ตรงกัน

  Backend ระบุ osnet_x0_25 แต่ training pipeline และ baseline ใช้ osnet_x1_0 ต้องเลือกให้เหลือแบบเดียวทั้ง train, evaluation และ production

  3. ข้อมูลสถานที่จริงยังน้อยเกินไป

  ปัจจุบันมีเพียง 3 คนใน reid_dataset/dataset_stats.json:7 และผล diagnostic มี Rank-1 เท่ากับ 0 ใน models/osnet/custom_diagnostic_baseline.json:2

  ที่สำคัญคือ similarity เฉลี่ย:

  - คนเดียวกัน: 0.5247
  - คนละคน: 0.5433

  แปลว่า feature ชุดนี้ยังใช้ threshold อย่างเดียวไม่ได้ เพราะคนละคนกลับคล้ายกันมากกว่าโดยเฉลี่ย

  4. BoT-SORT มีความเสี่ยงใช้ state ร่วมข้ามกล้อง

  ปัจจุบันเรียก model.track(... persist=True) จาก model กลางที่ backend/main.py:3741 ควรมี tracker instance แยกสำหรับแต่ละกล้อง ไม่เช่นนั้น local track state อาจปะปนกัน

  5. Global matching ปัจจุบันทำทีละกล้อง

  แม้จะใช้ Hungarian แต่เป็นการ assign ภายใน batch ของกล้องหนึ่งกล้อง ควรรวม tracklet จากทุกกล้องในช่วงเวลาเดียวกันก่อนตัดสิน เพื่อให้ one-to-one assignment เป็นระดับ global จริง

  ## แนวคิดหลักในการจดจำบุคคล

  ### 1. แยก Local ID และ Global ID อย่างชัดเจน

  - Local Track ID มาจาก BoT-SORT และใช้เฉพาะภายในกล้อง
  - Global ID เป็นตัวแทนบุคคลในระบบทั้งหมด
  - การเปลี่ยน Local ID ไม่ควรทำให้ Global ID เปลี่ยนทันที
  - Re-ID ควรทำเมื่อเกิด track ใหม่, track หาย หรือข้ามกล้อง ไม่ต้องตัดสินใหม่ทุกเฟรม

  ### 2. สร้างตัวแทนจากหลายภาพ

  แทนที่จะใช้ embedding จากภาพล่าสุดเพียงภาพเดียว:

  - เก็บเฉพาะ crop ที่คมและเห็นตัวค่อนข้างเต็ม
  - ไม่เก็บภาพระหว่างถูกบังหรือเดินซ้อนกัน
  - เก็บภาพด้านหน้า ด้านหลัง และด้านข้าง
  - รวม embedding หลายเฟรมเป็น tracklet prototype
  - เก็บ gallery แยกตามกล้อง เพราะสีและแสงแต่ละกล้องไม่เหมือนกัน

  สิ่งนี้จะลดปัญหาภาพไม่ดีเพียงเฟรมเดียวทำให้ ID ผิดและทำให้ gallery เสียตามไปด้วย

  ### 3. ใช้ Entry/Exit zone และเส้นทางกล้อง

  กำหนดว่า:

  - คนออกจากจุดไหนของ Camera A
  - สามารถไปปรากฏที่จุดไหนของ Camera B ได้
  - ใช้เวลาเดินต่ำสุดและสูงสุดเท่าไร
  - กล้องคู่ใดมีพื้นที่ทับซ้อนกัน

  ตัวอย่าง:

  Cam1 Exit-Door → Cam2 Entry-Left
  เวลาที่เป็นไปได้: 2–12 วินาที

  ถ้าคนเพิ่งออก Cam1 แล้วปรากฏ Cam2 ภายในเวลาที่สมเหตุสมผล คะแนนควรสูงขึ้นมาก แต่ถ้าปรากฏในกล้องที่ไปไม่ถึงภายในเวลานั้น ให้ตัด candidate ออกทันที

  ### 4. Matching แบบหลายหลักฐาน

  คะแนนอาจประกอบด้วย:

  Match Score =
    Appearance
  + Entry/Exit compatibility
  + Travel-time compatibility
  + Floorplan distance
  + Track quality

  และมี hard rules ก่อนคำนวณ:

  - ห้าม GID เดียวอยู่สองกล้องที่ไปถึงกันไม่ได้ในเวลาเดียวกัน
  - อนุญาต GID เดียวปรากฏพร้อมกันเฉพาะกล้องที่มีภาพซ้อนกัน
  - ถ้าอันดับหนึ่งกับอันดับสองมีคะแนนใกล้กันมาก ให้ยังไม่ match
  - ถ้าความมั่นใจต่ำ ให้สร้าง ID ชั่วคราวแทนการเดาสุ่ม

  หลักการสำคัญคือ false merge หรือเอาคนสองคนมารวมเป็น GID เดียว ร้ายแรงกว่าการสร้าง ID ใหม่เกินมาหนึ่งตัว ดังนั้นระบบควรตัดสินแบบ conservative

  ### 5. เพิ่มสถานะ PROVISIONAL

  นอกจาก ACTIVE และ DORMANT ควรมี:

  - PROVISIONAL: พบคนใหม่แต่ยังมีภาพไม่พอ
  - ACTIVE: มีหลักฐานเพียงพอและกำลังอยู่ในระบบ
  - DORMANT: ออกจากกล้องแล้ว รอจับคู่เมื่อเข้ากล้องอื่น
  - EXPIRED: หมดช่วงเวลาที่ควรนำกลับมาจับคู่

  เมื่อพบคนในกล้องใหม่ ระบบอาจรอภาพคุณภาพดีหลายเฟรมก่อนยืนยัน GID ทำให้ไม่ต้อง commit จากเฟรมแรก

  ### 6. SQLite Identity Memory

  ควรบันทึกอย่างน้อย:

  - Global ID และสถานะ
  - embedding gallery แยกตามกล้อง
  - tracklet และช่วงเวลา
  - ตำแหน่งล่าสุดบน floorplan
  - entry/exit zone
  - ประวัติกล้อง
  - คะแนน candidate ทั้งหมดตอนตัดสิน
  - เหตุผลที่ match หรือ reject

  ข้อมูลเหล่านี้ช่วยให้ restart โปรแกรมแล้ว memory ไม่หาย และช่วยตรวจย้อนหลังว่า ID สลับเพราะอะไร

  ## แผนพัฒนาเป็นเฟส

  ### Phase 1 — ทำ foundation ให้ถูกต้อง

  - ทำให้ OSNet checkpoint โหลดใน Docker ได้จริง
  - เลือก architecture เดียวกันทั้ง train และ production
  - แสดง model name, checkpoint, device และสถานะ fallback ใน /api/status
  - แยก BoT-SORT instance ต่อกล้อง
  - ตรวจ timestamp และ video offset ให้ตรงกัน

  ผลที่ต้องได้: รู้แน่นอนว่า production ใช้โมเดลอะไร และ local tracking ไม่ปะปนกัน

  ### Phase 2 — Tracklet และ Quality Gallery

  - รวม detection หลายเฟรมเป็น tracklet
  - เพิ่ม blur, crop size, occlusion และ border quality checks
  - ไม่ update identity gallery จากภาพคุณภาพต่ำ
  - สร้าง embedding prototype จากหลายภาพ
  - แยก gallery ตามมุมและตามกล้อง

  ผลที่ต้องได้: appearance ของแต่ละคนเสถียรกว่าการใช้ภาพล่าสุด

  ### Phase 3 — Cross-camera Intelligence

  - เพิ่ม entry/exit zones บน calibration UI
  - กำหนด camera topology
  - เรียนรู้หรือกำหนด travel-time window
  - สร้าง candidate เฉพาะเส้นทางที่เป็นไปได้
  - ทำ Hungarian assignment ระดับหลายกล้อง
  - เพิ่ม top-1/top-2 margin และ reject option

  ผลที่ต้องได้: คนข้ามกล้องได้ GID เดิมโดยไม่อาศัยเสื้อผ้าเพียงอย่างเดียว

  ### Phase 4 — ACTIVE/DORMANT Persistence

  - ทำ state machine ที่ชัดเจน
  - เชื่อม SQLite กับ lifecycle
  - กู้ memory หลัง restart
  - เก็บ match event เพื่อ debug
  - ป้องกัน gallery poisoning และรองรับ rollback เมื่อ match ผิด

  ### Phase 5 — เก็บข้อมูลและ Fine-tune

  - เก็บข้อมูลจากทุกกล้อง หลายช่วงเวลาและหลายสภาพแสง
  - ใช้คนจำนวนหลายสิบคนขึ้นไป ไม่ใช่เพียง 3 คน
  - แบ่ง train/validation/test แบบคนไม่ซ้ำกัน
  - เริ่มจาก public pretrained checkpoint
  - fine-tune กับสถานที่จริง
  - หา threshold จาก validation data ไม่กำหนดจากการคาดเดา

  ### Phase 6 — Evaluation

  ควรวัดอย่างน้อย:

  - Cross-camera handoff accuracy
  - False merge rate
  - False split/fragmentation rate
  - ID switches
  - IDF1
  - HOTA/Association Accuracy
  - Re-ID Rank-1 และ mAP
  - ผลแยกตามคู่กล้องและแต่ละ entry/exit zone

  ชุดทดสอบควรมีคนเสื้อคล้ายกัน, เดินสวนกัน, ถูกบัง, ออกแล้วกลับเข้า, อยู่ในสองกล้องที่ภาพทับกัน และเปลี่ยนสภาพแสง

  ## ลำดับที่ผมแนะนำให้ลงมือจริง

  1. ทำ OSNet production ให้โหลดได้จริง
  2. แยก tracker ต่อกล้อง
  3. เพิ่ม tracklet quality gallery
  4. เพิ่ม entry/exit zone และ camera topology
  5. ทำ global batch matching พร้อม ambiguity rejection
  6. เชื่อม ACTIVE/DORMANT กับ SQLite
  7. เก็บข้อมูลจริงและวัด baseline
  8. ค่อย fine-tune และปรับ threshold

  ขอบเขตของแผนนี้คือจำคนภายในหนึ่ง session หรือช่วงที่ยังแต่งตัวแบบเดิม หากต้องการจำคนข้ามวันหรือหลังเปลี่ยนเสื้อผ้า จะต้องเพิ่มข้อมูลประเภทอื่น เช่น face recognition หรือระบบลงทะเบียนบุคคล
  ซึ่งเป็นอีกระดับหนึ่งและต้องพิจารณาเรื่องความยินยอมและความเป็นส่วนตัวด้วยครับ