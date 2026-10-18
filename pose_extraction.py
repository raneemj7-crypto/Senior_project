from pathlib import Path
from ultralytics import YOLO
import cv2
import csv

# Project folders
BASE = Path(__file__).resolve().parent
DATASET = BASE / "split_dataset" / "train" / "fall"
OUTPUT = BASE / "results" / "pose_extraction"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load pretrained YOLO pose model
model = YOLO(str(BASE / "yolov8n-pose.pt"))

# Select one fall sequence
sequences = sorted([p for p in DATASET.iterdir() if p.is_dir()])
assert sequences, "No fall sequences found!"

sequence = sequences[0]
frames = sorted(sequence.glob("*.png"))[:30]

print(f"Sequence: {sequence.name}")
print(f"Frames found: {len(frames)}")

assert len(frames) == 30, "Sequence has fewer than 30 frames!"

# Save extracted keypoints
csv_path = OUTPUT / "keypoints.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    header = ["frame", "person_confidence"]

    for i in range(17):
        header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_confidence"]

    writer.writerow(header)

    for index, frame_path in enumerate(frames):
        results = model.predict(
            source=str(frame_path),
            conf=0.25,
            verbose=False
        )

        result = results[0]
        image = result.plot()

        # Save visualization
        cv2.imwrite(
            str(OUTPUT / f"frame_{index:03d}.jpg"),
            image
        )

        if len(result.boxes) > 0 and result.keypoints is not None:
            # Select highest-confidence person
            person_idx = int(result.boxes.conf.argmax().item())

            person_conf = float(
                result.boxes.conf[person_idx].item()
            )

            keypoints = result.keypoints.data[person_idx].cpu().numpy()

            row = [frame_path.name, person_conf]

            for kp in keypoints:
                row.extend([
                    float(kp[0]),
                    float(kp[1]),
                    float(kp[2])
                ])

            writer.writerow(row)

        else:
            writer.writerow(
                [frame_path.name, 0.0] + [""] * 51
            )

        print(f"Processed {index + 1}/{len(frames)} frames")

print("\nDONE!")
print(f"Results saved to: {OUTPUT}")

