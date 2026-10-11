
from pathlib import Path
import csv
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

FRAMES_DIR = (
    Path.home()
    / "Desktop/senior-project-storage/raw/mot20/train/MOT20-01/img1"
)

OUTPUT_CSV = "yolo26_mot20_tracking.csv"
OUTPUT_VIDEO = "yolo26_mot20_tracking.mp4"

frames = sorted(FRAMES_DIR.glob("*.jpg"))
assert frames, "MOT20 frames not found"

model = YOLO("yolo26m.pt")
tracker = sv.ByteTrack()

first = cv2.imread(str(frames[0]))
height, width = first.shape[:2]

# Visualization output only
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    25,
    (width, height),
)
assert writer.isOpened(), "Cannot create output video"

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

start = time.perf_counter()

with open(OUTPUT_CSV, "w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([
        "frame", "track_id",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "width", "height", "confidence"
    ])

    for frame_number, path in enumerate(frames, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read {path}")

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            classes=[0],
            device="cpu",
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Update tracker on every frame, even if empty.
        detections = tracker.update_with_detections(detections)

        for box, conf, track_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.tracker_id,
        ):
            if track_id is None:
                continue

            x1, y1, x2, y2 = map(float, box)
            csv_writer.writerow([
                frame_number, int(track_id),
                x1, y1, x2, y2,
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
                float(conf),
            ])

        labels = [
            f"ID {track_id}"
            for track_id in detections.tracker_id
        ]

        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
        )
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )
        writer.write(annotated)

        if frame_number % 50 == 0:
            print(f"Processed frame: {frame_number}")

writer.release()

print(f"\nTracking completed in {time.perf_counter()-start:.1f}s")
print("Video:", OUTPUT_VIDEO)
print("CSV:", OUTPUT_CSV)
