from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATASET = BASE / "split_dataset" / "train" / "fall"
OUTPUT = BASE / "results" / "fall_verification"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Approximate fall ranges identified from our previous screenshots
FALL_RANGES = {
    "fall-01-cam0-rgb": (80, 135),
    "fall-01-cam1-rgb": (75, 140),
    "fall-03-cam0-rgb": (145, 195),
    "fall-03-cam1-rgb": (145, 200),
    "fall-04-cam0-rgb": (5, 65),
    "fall-04-cam1-rgb": (15, 70),
}

for name, (start, end) in FALL_RANGES.items():
    sequence = DATASET / name

    if not sequence.exists():
        print(f"Missing: {name}")
        continue

    frames = sorted(sequence.glob("*.png"))

    if not frames:
        continue

    # Select 20 frames from the suspected fall interval
    indices = np.linspace(
        start,
        min(end, len(frames) - 1),
        20,
        dtype=int
    )

    thumbnails = []

    for index in indices:
        image = cv2.imread(str(frames[index]))

        if image is None:
            continue

        image = cv2.resize(image, (320, 240))

        cv2.putText(
            image,
            f"Frame {index}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        thumbnails.append(image)

    if len(thumbnails) != 20:
        print(f"Skipping incomplete sequence: {name}")
        continue

    rows = []

    for i in range(0, 20, 5):
        rows.append(np.hstack(thumbnails[i:i + 5]))

    contact_sheet = np.vstack(rows)

    cv2.imwrite(
        str(OUTPUT / f"{name}.jpg"),
        contact_sheet
    )

    print(f"Verified sheet created: {name}")

print("\nDONE!")
print("Saved to:", OUTPUT)