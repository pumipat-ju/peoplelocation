# Prompt 04 — Controlled OSNet Fine-Tuning

## เป้าหมาย
Fine-tune OSNet บน PeopleLocation dataset แบบ reproducible และไม่ใช้ test set ในการเลือก model

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
- Prompt 03 ต้องผ่าน leakage checks
- เช็ค `torch.cuda.is_available()` และ device จริง
- ระบุ initial checkpoint ชัดเจน
- สร้าง experiment directory ใหม่แบบ versioned
- test split ต้องถูกล็อกและห้ามใช้เลือก checkpoint

## งาน
1. ตรวจ training code เดิมก่อนแก้
2. initialize จาก verified Market1501 checkpoint
3. log hyperparameters:
   - architecture
   - checkpoint
   - input size
   - optimizer
   - LR
   - weight decay
   - scheduler
   - batch size
   - epochs
   - losses
   - sampler
   - augmentation
   - random seed
4. ใช้ validation สำหรับ checkpoint selection / early stopping
5. ห้ามใช้ test metric เลือก checkpoint
6. save training log, best checkpoint, final checkpoint, config
7. smoke test checkpoint:
   - load ได้
   - output 512-D
   - finite
   - inference ใช้งานได้

## Output
- `reid_experiments/osnet_peoplelocation_v1/`
- `config.json`
- `training_log.csv`
- `best_checkpoint.pth`
- `final_checkpoint.pth`
- `TRAINING_REPORT.md`

## สรุปกลับมา
- device
- hyperparameters
- best epoch
- validation metrics
- overfitting signs
- checkpoint paths
- smoke-test result

ห้าม integrate production
