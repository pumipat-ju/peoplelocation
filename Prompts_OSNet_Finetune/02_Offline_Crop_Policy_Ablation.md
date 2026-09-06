# Prompt 02 — Offline Crop Policy Ablation

## เป้าหมาย
วัดว่า crop geometry / margin / overhead viewpoint เป็นสาเหตุหลักของ Re-ID ที่แย่หรือไม่ โดยไม่แก้ production

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
- Prompt 01 ต้องผ่านแล้ว
- cross-camera GT ต้องไม่มี critical label issue ค้างอยู่
- ใช้ checkpoint เดิม `osnet_x1_0_market1501.pth`
- เก็บ production crop เดิมไว้ ห้ามแก้

## งาน
สร้าง offline ablation อย่างน้อย 4 แบบ:

A. Current production crop  
B. Raw GT bbox / no extra margin  
C. Preserve aspect ratio + padding ไปยัง 128x256 โดยไม่ stretch  
D. Quality-filtered subset เช่นขั้นต่ำของ area, aspect ratio, truncation

ทุก policy ต้องใช้:
- checkpoint เดียวกัน
- preprocessing เดียวกัน
- L2 normalization เดียวกัน
- cosine similarity เดียวกัน
- sample set เดียวกันเท่าที่เป็นไปได้

วัด:
- Same-ID mean/std/min/max
- Different-ID mean/std/min/max
- Gap
- ROC-AUC
- EER
- Youden threshold
- TPR/FPR
- hard positives / hard negatives

Breakdown:
- cam1 same-camera
- cam2 same-camera
- cam1↔cam2
- normal aspect
- suspicious aspect

## Output
- `reid_crop_ablation/`
- metrics JSON ต่อ policy
- comparison CSV
- `CROP_ABLATION_REPORT.md`

## สรุปกลับมา
ตาราง:
| Policy | Samples | AUC | EER | Same Mean | Diff Mean | Gap |

บอกด้วยว่า crop policy อย่างเดียวช่วยได้มากหรือไม่ และยังควร fine-tune หรือไม่

ห้ามแก้ production crop
หยุดหลัง Prompt 02
