import cv2
from pathlib import Path

source = Path("../test/fall/fall-06-cam1-rgb")
output = Path("../test/fall-06-test.mp4")

images = sorted(source.glob("*.png"))

if not images:
    raise RuntimeError(f"No PNG images found in {source}")

first = cv2.imread(str(images[0]))
if first is None:
    raise RuntimeError("Could not read first image")

height, width = first.shape[:2]

# Temporary playback rate; confirm the dataset's original FPS.
fps = 15

writer = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

for image in images:
    frame = cv2.imread(str(image))
    if frame is not None:
        writer.write(frame)

writer.release()

print(f"Created: {output}")
print(f"Frames: {len(images)}")
print(f"Playback FPS: {fps}")
