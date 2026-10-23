
from pathlib import Path
from ultralytics import YOLO
import cv2
import random

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets/CAUCAFall_frames/train"
OUTPUT = ROOT / "runs/annotation_preview_conf010"
OUTPUT.mkdir(parents=True, exist_ok=True)

model = YOLO(
    str(ROOT / "models/yolo11_fall_pretrained.pt")
)

random.seed(42)

for class_name in ("fallen", "sitting", "standing"):
    folder = SOURCE / class_name
    images = sorted(folder.glob("*.jpg"))

    if not images:
        print(f"No images in {folder}")
        continue

    samples = random.sample(images, min(5, len(images)))

    for image_path in samples:
        results = model.predict(
            source=str(image_path),
            conf=0.10,
            imgsz=640,
            device="mps",
            verbose=False
        )

        annotated = results[0].plot()

        destination = (
            OUTPUT / f"{class_name}_{image_path.name}"
        )

        cv2.imwrite(str(destination), annotated)

        print(
            f"{class_name}: {image_path.name} "
            f"-> {len(results[0].boxes)} detections"
        )

print("\nPreviews saved to:", OUTPUT)
