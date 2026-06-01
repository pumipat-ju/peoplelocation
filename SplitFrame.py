import cv2
import os

VIDEO_PATH = input("กรอก path ไฟล์วิดีโอ: ").strip().strip('"')
OUTPUT_DIR = input("กรอก path โฟลเดอร์ปลายทาง: ").strip().strip('"')

save_every = input("ต้องการเซฟทุกกี่เฟรม? (เช่น 10): ").strip()
SAVE_EVERY_N_FRAME = int(save_every) if save_every.isdigit() else 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("เปิดวิดีโอไม่สำเร็จ")
    raise SystemExit

frame_idx = 0
save_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % SAVE_EVERY_N_FRAME == 0:
        filename = os.path.join(OUTPUT_DIR, f"frame_{save_idx:06d}.jpg")
        cv2.imwrite(filename, frame)
        save_idx += 1

    frame_idx += 1

cap.release()
print(f"แตกเฟรมเสร็จ: {save_idx} รูป")
print(f"บันทึกไว้ที่: {OUTPUT_DIR}")