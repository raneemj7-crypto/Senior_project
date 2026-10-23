
from pathlib import Path
from ultralytics import YOLO
import cv2
import csv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/CAUCAFall_frames/train"
OUTPUT = ROOT / "runs/candidate_boxes"
OUTPUT.mkdir(parents=True, exist_ok=True)

# General person detector, used only for annotation
model = YOLO("yolo11n.pt")

classes = ("fallen", "sitting", "standing")
records = []

for class_name in classes:
    for image_path in sorted((SOURCE / class_name).glob("*.jpg")):
        result = model.predict(
            source=str(image_path),
            classes=[0],  # COCO person
            conf=0.15,
            imgsz=640,
            device="mps",
            verbose=False
        )[0]

        image = cv2.imread(str(image_path))
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

        cv2.imwrite(
            str(OUTPUT / image_path.name),
            image
        )

        records.append({
            "image": str(image_path),
            "label": class_name,
            "detections": len(boxes)
        })

with open(OUTPUT / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "label", "detections"]
    )
    writer.writeheader()
    writer.writerows(records)

print(f"Processed {len(records)} images")
print(f"Results saved to {OUTPUT}")
