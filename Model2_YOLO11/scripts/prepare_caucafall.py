
from pathlib import Path
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets/CAUCAFall_clean"
OUTPUT = ROOT / "datasets/CAUCAFall_frames"

LABEL_MAP = {
    "fallen": "fallen",
    "lying": "fallen",
    "sitting": "sitting",
    "standing": "standing",
}

# Extract approximately every 10th frame
FRAME_STEP = 10

segments = pd.read_csv(SOURCE / "labels/segments.csv")

counts = {}

for row in segments.itertuples(index=False):
    if row.label_name not in LABEL_MAP:
        continue

    video_path = SOURCE / row.file

    if not video_path.exists():
        print(f"Missing video: {video_path}")
        continue

    label = LABEL_MAP[row.label_name]
    output_dir = OUTPUT / row.split / label
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    start = int(row.start_frame)
    end = int(row.end_frame)

    # Avoid frames close to action transitions
    margin = min(5, max(0, (end - start) // 4))

    for frame_id in range(start + margin, end - margin, FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if not ok:
            continue

        filename = f"{row.video}_{frame_id:05d}.jpg"
        destination = output_dir / filename

        cv2.imwrite(str(destination), frame)

        key = (row.split, label)
        counts[key] = counts.get(key, 0) + 1

    cap.release()

print("\nExtraction completed!")
for key, count in sorted(counts.items()):
    print(key, count)

print("\nOutput:", OUTPUT)
