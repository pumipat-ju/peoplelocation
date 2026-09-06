# Prompt 06 — Production Integration Gate

## เป้าหมาย
ตัดสิน GO/NO-GO สำหรับ fine-tuned checkpoint และ integrate เฉพาะ Re-ID model path ถ้าผล test ดีจริง

## ข้อห้ามสำคัญ

ระบบ camera/video/calibration ถือเป็น frozen subsystem

ห้ามแก้:
- camera discovery / device access
- Docker camera passthrough
- live camera capture
- uploaded video input
- capture / processing workers
- reconnect logic
- video decoding / playback
- live preview / frame cache
- calibration flow / modal
- homography input plumbing
- camera generation / worker generation logic

ห้าม revert งานเดิมของผู้ใช้ และก่อนแก้ให้ตรวจ `git status` เสมอ
ถ้างานจำเป็นต้องแตะ frozen subsystem ให้หยุดและรายงาน conflict แทน


## ก่อนรัน
- Prompt 05 ต้องเสร็จ
- held-out metrics ต้องดีขึ้นอย่างมีนัยสำคัญ
- เก็บ baseline checkpoint สำหรับ rollback
- ตรวจ `git status`
- ยืนยัน frozen subsystem ยังไม่ถูกแตะ

ถ้าผล fine-tuned ไม่ดีขึ้นชัดเจน ให้ NO-GO และห้าม integrate

## งาน

### Phase A — Gate
สรุป:
- baseline metrics
- fine-tuned metrics
- delta
- risks
- dataset limitations
- crop limitations

คืนสถานะ:
- `GO_FOR_GUARDED_INTEGRATION`
หรือ
- `NO_GO`

### Phase B — ทำเฉพาะถ้า GO
แก้เฉพาะ Re-ID model-loading/config path เท่าที่จำเป็น เช่น:
- checkpoint path/config
- model version logging
- safe checkpoint validation
- Re-ID smoke tests

ห้ามแก้:
- camera
- video
- worker
- preview
- calibration
- topology
- tracker
- handoff logic

### Validation
- syntax/tests
- model load
- embedding dim 512
- finite embedding
- unit norm
- offline verification
- non-camera Re-ID regression tests

## Output
- `PRODUCTION_INTEGRATION_GATE.md`
- exact files changed
- old/new checkpoint
- rollback instructions
- tests
- final GO/NO-GO
