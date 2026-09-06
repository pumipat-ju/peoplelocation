# Prompt 05 — Held-Out Evaluation

## เป้าหมาย
เทียบ baseline Market1501 กับ fine-tuned checkpoint บน held-out test set เดียวกัน

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
- best checkpoint ต้องเลือกจาก validation เท่านั้น
- test set ต้องไม่เคยใช้เลือก model
- baseline checkpoint ต้องยังอยู่
- ใช้ test samples / crop / preprocessing / pair rule เดียวกันทั้งสอง model

## งาน
ประเมิน:
1. baseline `osnet_x1_0_market1501.pth`
2. fine-tuned best checkpoint

Primary rule = CROSS-CAMERA verification

วัด:
- Same-ID mean/std/min/max
- Different-ID mean/std/min/max
- Gap
- ROC-AUC
- EER
- Youden threshold
- TPR/FPR
- hard positives / hard negatives

ถ้ามี evaluator ที่ถูกต้องให้เพิ่ม:
- Rank-1
- Rank-5
- mAP

สร้างตาราง:
| Metric | Baseline | Fine-tuned | Delta |

ห้ามสรุปว่าดีขึ้นเพียงเพราะ training loss ลดลง

## Output
- evaluation JSON/CSV
- `FINAL_MODEL_EVALUATION.md`

## สรุปกลับมา
ตอบให้ชัด:
1. AUC ดีขึ้นหรือไม่
2. EER ลดหรือไม่
3. Gap เพิ่มหรือไม่
4. false match ดีขึ้นหรือไม่
5. พร้อม guarded production integration หรือไม่

ห้ามแก้ production runtime
