from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATASET = BASE / "split_dataset" / "train" / "fall"
OUTPUT = BASE / "results" / "fall_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Analyze the first 6 fall recordings
sequences = sorted(
    p for p in DATASET.iterdir() if p.is_dir()
)[:6]

for sequence in sequences:
    frames = sorted(sequence.glob("*.png"))

    if not frames:
        continue

    # Select 12 evenly spaced frames
    indices = np.linspace(
        0, len(frames) - 1, 12, dtype=int
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

    if len(thumbnails) != 12:
        print(f"Skipping incomplete sequence: {sequence.name}")
        continue

    rows = []

    for i in range(0, 12, 4):
        rows.append(
            np.hstack(thumbnails[i:i + 4])
        )

    contact_sheet = np.vstack(rows)

    cv2.imwrite(
        str(OUTPUT / f"{sequence.name}.jpg"),
        contact_sheet
    )

    print(f"Analyzed: {sequence.name}")

print("\nDONE!")
print("Contact sheets saved to:", OUTPUT)