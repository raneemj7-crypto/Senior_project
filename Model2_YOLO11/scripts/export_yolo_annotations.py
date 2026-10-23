
from pathlib import Path
from ultralytics import YOLO
import shutil
import csv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/CAUCAFall_frames/train"
OUTPUT = ROOT / "datasets/CAUCAFall_annotation_review"

CLASSES = {
    "fallen": 0,
    "sitting": 1,
    "standing": 2
}

model = YOLO("yolo11n.pt")

records = []

for class_name, class_id in CLASSES.items():
    for image_path in sorted((SOURCE / class_name).glob("*.jpg")):

        image_output = OUTPUT / "images" / "train"
        label_output = OUTPUT / "labels" / "train"

        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        destination = image_output / (
            f"{class_name}_{image_path.name}"
        )

        shutil.copy2(image_path, destination)

        result = model.predict(
            source=str(image_path),
            classes=[0],
            conf=0.15,
            imgsz=640,
            device="mps",
            verbose=False
        )[0]

        label_path = label_output / (
            destination.stem + ".txt"
        )

        boxes = result.boxes.xywhn.cpu().numpy()

        with open(label_path, "w") as f:
            for x, y, w, h in boxes:
                f.write(
                    f"{class_id} "
                    f"{x:.6f} {y:.6f} "
                    f"{w:.6f} {h:.6f}\n"
                )

        records.append({
            "image": destination.name,
            "class": class_name,
            "detections": len(boxes),
            "review_status": "unverified"
        })

with open(OUTPUT / "review_manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image", "class",
            "detections", "review_status"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

print("Candidate annotation export completed!")
print("Images:", len(records))
print("Output:", OUTPUT)
