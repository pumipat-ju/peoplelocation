import cv2
import time
from ultralytics import YOLO

# =========================
# ตั้งค่า
# =========================
MODEL_PATH = "yolo11s.pt"   #yolov8s.pt, yolov8x.pt, yolo11s.pt, yolo11x.pt
CAMERA_SOURCE = 0           
CONF = 0.25
IMG_SIZE = 640

# =========================
# โหลดโมเดล
# =========================
print(f"กำลังโหลดโมเดล: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# =========================
# เปิดกล้อง
# =========================
cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

prev_time = time.time()

print("เริ่ม realtime detection... กด q เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        print("อ่านภาพจากกล้องไม่สำเร็จ")
        break

    # predict เฉพาะ person class (class 0)
    results = model.predict(
        source=frame,
        conf=CONF,
        imgsz=IMG_SIZE,
        classes=[0],
        verbose=False
    )

    annotated = frame.copy()
    person_count = 0

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and boxes.xyxy is not None:
            xyxy_list = boxes.xyxy.cpu().numpy()

            confs = None
            if boxes.conf is not None:
                confs = boxes.conf.cpu().numpy().tolist()

            for i, box in enumerate(xyxy_list):
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf_text = ""
                if confs is not None and i < len(confs):
                    conf_text = f"{confs[i]:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"Person {conf_text}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                person_count += 1

    # คำนวณ FPS
    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    # แสดงข้อมูลบนภาพ
    cv2.putText(
        annotated,
        f"Model: {MODEL_PATH}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )
    cv2.putText(
        annotated,
        f"People: {person_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2
    )

    cv2.imshow("YOLO Realtime Detection", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()