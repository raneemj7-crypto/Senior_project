from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent

SEQUENCE = (
    BASE / "split_dataset" / "train" / "fall"
    / "fall-03-cam0-rgb"
)

frames = sorted(SEQUENCE.glob("*.png"))

assert frames, "No frames found!"

first = cv2.imread(str(frames[0]))
height, width = first.shape[:2]

output = BASE / "resulbuts" / "fall_sequence_preview.mp4"
output.parent.mkdir(parents=True, exist_ok=True)

video = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    15,
    (width, height)
)

for index, frame in enumerate(frames):
    image = cv2.imread(str(frame))

    cv2.putText(
        image,
        f"Frame: {index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    video.write(image)

video.release()

print("Video created!")
print("Total frames:", len(frames))
print("Saved to:", output)