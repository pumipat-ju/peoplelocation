from ultralytics import YOLO
import pandas as pd

# =============================
# config
# =============================

DATASET = "coco8.yaml"   # dataset demo
MODELS = [
    "yolov8s.pt",
    "yolov8x.pt",
    "yolo11s.pt",
    "yolo11x.pt"
]

IMG_SIZE = 640


# =============================
# evaluate function
# =============================

def evaluate_model(model_name):

    print(f"\nกำลังทดสอบ {model_name}")

    model = YOLO(model_name)

    results = model.val(
        data=DATASET,
        imgsz=IMG_SIZE,
        verbose=False
    )

    metrics = {
        "model": model_name,
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map)
    }

    return metrics


# =============================
# main
# =============================

all_results = []

for m in MODELS:
    metrics = evaluate_model(m)
    all_results.append(metrics)

# convert to table
df = pd.DataFrame(all_results)

print("\n================ RESULT ================")
print(df)
print("========================================")

# save csv
df.to_csv("yolo_compare_results.csv", index=False)

print("\nบันทึกผลไว้ที่ yolo_compare_results.csv")


print("\nคำอธิบาย metric:")
print("mAP50     → ความแม่นยำเมื่อ IoU = 0.5")
print("mAP50-95  → ค่าเฉลี่ยหลาย threshold")
print("Precision → detect แล้วถูกกี่ %")
print("Recall    → ตรวจเจอครบกี่ %")