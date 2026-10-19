import os
import csv
import cv2
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

# Change this path to where your URFD split_dataset folder is located
from pathlib import Path

DATASET_ROOT = Path("/Users/reemahumayed/desktop/UDFD/split_dataset")

# We will save all extracted pose rows here
OUTPUT_CSV = "urfd_pose_data.csv"

# Use the same YOLOv8 Pose model as the baseline repo
MODEL_PATH = "yolov8n-pose.pt"

# YOLO confidence threshold
MIN_CONF_DETECTION = 0.5

# True labels from the dataset folders
LABEL_MAP = {
    "fall": 1,
    "non_fall": 0
}

# ============================================================
# LOAD YOLO POSE MODEL
# ============================================================

model = YOLO(MODEL_PATH)

print(f"Loaded pose model: {MODEL_PATH}")

# ============================================================
# PREPARE CSV
# ============================================================

csv_file = open(OUTPUT_CSV, "w", newline="")
csv_writer = csv.writer(csv_file)

header = [
    "split",
    "class_name",
    "true_label",
    "sequence_name",
    "frame_name",
    "frame_index"
]

# YOLOv8 Pose has 17 keypoints
# Save x, y, confidence for each one
for i in range(17):
    header += [
        f"kpt_{i}_x",
        f"kpt_{i}_y",
        f"kpt_{i}_conf"
    ]

csv_writer.writerow(header)

# ============================================================
# PROCESS DATASET
# ============================================================

for split in ["train", "val", "test"]:

    for class_name in ["fall", "non_fall"]:

        class_folder = DATASET_ROOT / split / class_name

        if not class_folder.exists():
            print(f"Skipping missing folder: {class_folder}")
            continue

        true_label = LABEL_MAP[class_name]

        # Each subfolder is one temporal sequence
        sequence_folders = sorted(
            [p for p in class_folder.iterdir() if p.is_dir()]
        )

        for sequence_folder in sequence_folders:

            sequence_name = sequence_folder.name

            print(f"Processing: {split}/{class_name}/{sequence_name}")

            # Read frames in order
            frame_files = sorted(
                list(sequence_folder.glob("*.png"))
                + list(sequence_folder.glob("*.jpg"))
                + list(sequence_folder.glob("*.jpeg"))
            )

            for frame_index, frame_path in enumerate(frame_files):

                frame = cv2.imread(str(frame_path))

                if frame is None:
                    print(f"Could not read: {frame_path}")
                    continue

                # Run YOLOv8-Pose
                result = model(
                    frame,
                    conf=MIN_CONF_DETECTION,
                    verbose=False
                )[0]

                # URFD is mainly single-person data.
                # For now, use the first detected person's pose.
                if (
                    result.keypoints is None
                    or len(result.keypoints) == 0
                ):
                    print(f"No pose detected: {frame_path.name}")
                    continue

                pose = result.keypoints[0]

                if (
                    pose.xy is None
                    or pose.conf is None
                    or len(pose.xy) == 0
                    or len(pose.conf) == 0
                ):
                    print(f"Incomplete pose: {frame_path.name}")
                    continue

                keypoints = pose.xy[0].cpu().numpy()
                confs = pose.conf[0].cpu().numpy()

                if len(keypoints) < 17:
                    print(f"Not enough keypoints: {frame_path.name}")
                    continue

                row = [
                    split,
                    class_name,
                    true_label,
                    sequence_name,
                    frame_path.name,
                    frame_index
                ]

                for i in range(17):
                    row += [
                        float(keypoints[i][0]),
                        float(keypoints[i][1]),
                        float(confs[i])
                    ]

                csv_writer.writerow(row)

# ============================================================
# CLEANUP
# ============================================================

csv_file.close()

print()
print("Finished extracting URFD pose keypoints.")
print(f"Saved to: {OUTPUT_CSV}")