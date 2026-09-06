# Prompt 03 — Prepare Fine-Tuning Dataset

## เป้าหมาย
สร้าง dataset สำหรับ fine-tune ที่ clean และไม่มี identity leakage

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
- Prompt 01 และ 02 ต้องเสร็จ
- เลือก offline crop policy จากผล measurement แล้ว
- freeze dataset version สำหรับ experiment นี้
- เก็บ baseline metrics ไว้

## งาน
1. สร้าง dataset จาก versioned manifest
2. ใช้ crop policy ที่เลือกจาก Prompt 02
3. แบ่ง train/val/test แบบ deterministic
4. ต้องเป็น identity-disjoint split:
   - คนที่อยู่ train ห้ามอยู่ val/test
5. ตรวจ cross-camera positives ต่อ identity
6. รายงาน images per identity / camera coverage / imbalance
7. เก็บ hard negatives ตามธรรมชาติ
8. ห้าม relabel อัตโนมัติ
9. สร้าง split manifests
10. ตรวจ:
   - identity leakage
   - duplicate sample IDs
   - unreadable images
   - empty crops
   - wrong GT mapping

ถ้า dataset เล็กเกินไปให้รายงานตรง ๆ ห้ามแก้ด้วย image-level random split

## Output
- `datasets/peoplelocation_reid_v1/`
- train/val/test
- split manifests
- `dataset_stats.json`
- `DATASET_PREPARATION_REPORT.md`

## สรุปกลับมา
- total identities/images
- train/val/test identities + images
- cross-camera positive counts
- rejected samples
- leakage result
- dataset พร้อม fine-tune หรือไม่

ห้ามเริ่ม training
