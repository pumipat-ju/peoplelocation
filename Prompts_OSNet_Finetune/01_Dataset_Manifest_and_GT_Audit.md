# Prompt 01 — Dataset Manifest + Cross-Camera GT Audit

## เป้าหมาย
สร้าง dataset manifest แบบ versioned และ audit GT identity ข้ามกล้องก่อนเริ่ม fine-tune

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
- ยืนยันว่า `labeled_data` และ GT files อยู่ครบ
- เก็บ baseline ไว้:
  - Cross-camera AUC = 0.519965
  - EER = 0.479167
  - Gap = 0.008373
- ห้ามเริ่ม fine-tune ใน prompt นี้

## งาน
1. ตรวจ schema ของ GT ทุก sequence/camera
2. ถ้าเป็น MOT format ให้ตีความ `[frame,id,x,y,width,height,...]`
   และแปลง bbox เป็น `(x,y,x+width,y+height)`
3. สร้าง versioned manifest โดยมีอย่างน้อย:
   - dataset_version
   - sequence
   - camera
   - source/frame
   - ground_truth_person_id
   - bbox_xywh
   - bbox_xyxy
   - image width/height
   - bbox area
   - aspect ratio
   - deterministic sample_id
4. ตรวจ duplicate / invalid bbox / missing frame
5. Audit identity ข้ามกล้อง:
   - แต่ละ ID อยู่กล้องไหน
   - จำนวน sample
   - first/last frame
   - ID ที่มีแค่กล้องเดียว
   - ID ที่น่าสงสัยว่า label ผิด
6. สร้าง representative crops สำหรับ review ข้ามกล้อง
7. ห้ามแก้ label อัตโนมัติ
8. เพิ่ม tests สำหรับ bbox conversion, deterministic manifest, duplicate sample IDs, valid bounds

## Output ที่ต้องมี
- `reid_dataset/manifest_v1.jsonl`
- `reid_dataset/manifest_v1_summary.json`
- `reid_dataset/gt_audit/cross_camera_identity_summary.csv`
- `reid_dataset/gt_audit/GT_AUDIT_REPORT.md`

## สรุปกลับมา
- files changed
- จำนวน samples / identities
- identities shared across cameras
- invalid/rejected bboxes
- suspected GT issues
- พร้อมสำหรับ training หรือไม่

หยุดหลัง Prompt 01
