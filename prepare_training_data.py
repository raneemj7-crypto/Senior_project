from pathlib import Path
from ultralytics import YOLO
import numpy as np

BASE = Path(__file__).resolve().parent
DATASET = BASE / "split_dataset" / "train"
OUTPUT = BASE / "results" / "training_data"

OUTPUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(BASE / "yolov8n-pose.pt"))

WINDOW = 30
STRIDE = 10

X = []
y = []
sequence_names = []
event_ids = []

# Approximate fall intervals verified visually
FALL_EVENTS = {
    "fall-01-cam0-rgb": (90, 125),
    "fall-01-cam1-rgb": (90, 125),
    "fall-03-cam0-rgb": (155, 190),
    "fall-03-cam1-rgb": (160, 195),
    "fall-04-cam0-rgb": (17, 55),
    "fall-04-cam1-rgb": (25, 60),
}


def extract_pose(frame):
    result = model.predict(
        str(frame),
        conf=0.25,
        verbose=False
    )[0]

    if len(result.boxes) > 0 and result.keypoints is not None:
        idx = int(result.boxes.conf.argmax().item())
        keypoints = result.keypoints.data[idx].cpu().numpy().copy()
    else:
        keypoints = np.zeros((17, 3), dtype=np.float32)

    height, width = result.orig_shape

    keypoints[:, 0] /= width
    keypoints[:, 1] /= height

    return keypoints


def add_windows(poses, sequence_name, event_id, fall_range=None):
    for start in range(0, len(poses) - WINDOW + 1, STRIDE):
        end = start + WINDOW

        if fall_range is None:
            label = 0
        else:
            fall_start, fall_end = fall_range

            overlap = max(
                0,
                min(end, fall_end) - max(start, fall_start)
            )

            # Require at least 15 frames of falling movement
            if overlap >= 15:
                label = 1

            # Windows entirely before the fall are non-falls
            elif end <= fall_start:
                label = 0

            # Skip uncertain and post-fall windows
            else:
                continue

        X.append(poses[start:end])
        y.append(label)
        sequence_names.append(f"{sequence_name}_{start}_{end}")
        event_ids.append(event_id)


# Process non-fall recordings
non_fall_folder = DATASET / "non_fall"

non_fall_sequences = sorted(
    p for p in non_fall_folder.iterdir() if p.is_dir()
)[:10]

for sequence in non_fall_sequences:
    print(f"Processing non-fall: {sequence.name}")

    frames = sorted(sequence.glob("*.png"))

    if len(frames) < WINDOW:
        continue

    poses = np.array(
        [extract_pose(frame) for frame in frames],
        dtype=np.float32
    )

    add_windows(
        poses,
        sequence.name,
        sequence.name
    )


# Process fall recordings
fall_folder = DATASET / "fall"

for sequence_name, fall_range in FALL_EVENTS.items():
    sequence = fall_folder / sequence_name

    if not sequence.exists():
        print(f"Missing: {sequence_name}")
        continue

    print(f"Processing fall: {sequence_name}")

    frames = sorted(sequence.glob("*.png"))

    if len(frames) < WINDOW:
        continue

    poses = np.array(
        [extract_pose(frame) for frame in frames],
        dtype=np.float32
    )

    # Both camera angles share the same event ID
    event_id = sequence_name.split("-cam")[0]

    add_windows(
        poses,
        sequence_name,
        event_id,
        fall_range
    )


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

np.save(OUTPUT / "X_train_sample.npy", X)
np.save(OUTPUT / "y_train_sample.npy", y)
np.save(OUTPUT / "sequence_names.npy", np.array(sequence_names))
np.save(OUTPUT / "event_ids.npy", np.array(event_ids))

print("\nDATASET PREPARATION COMPLETE!")
print("Input shape:", X.shape)
print("Fall windows:", int(sum(y == 1)))
print("Non-fall windows:", int(sum(y == 0)))
print("Unique recording/event IDs:", len(set(event_ids)))
print("Saved to:", OUTPUT)