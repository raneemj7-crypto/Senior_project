
from pathlib import Path
import csv
import time

import cv2
import numpy as np
import torch
import supervision as sv

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

ROOT = Path.home() / (
    "Desktop/senior-project-storage/raw/mot20/"
    "train/MOT20-01"
)
FRAMES_DIR = ROOT / "img1"
OUTPUT_DIR = Path("confidence_experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

WEIGHTS = "weights/crowdhuman_yolov5m.pt"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
THRESHOLDS = [CONF_THRES]

device = torch.device("cpu")
frames = sorted(FRAMES_DIR.glob("*.jpg"))
assert frames, "MOT20-01 frames not found"

model = attempt_load(WEIGHTS, map_location=device)
model.eval()
stride = int(model.stride.max())
names = model.module.names if hasattr(model, "module") else model.names
person_id = list(names).index("person") if not isinstance(names, dict) else next(
    key for key, value in names.items() if value == "person"
)

trackers = {}
files = {}
writers = {}

for threshold in THRESHOLDS:
    trackers[threshold] = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=25,
        minimum_consecutive_frames=1,
    )

    filename = OUTPUT_DIR / f"tracking_conf_{threshold:.2f}.csv"
    f = open(filename, "w", newline="")
    files[threshold] = f
    writer = csv.writer(f)
    writers[threshold] = writer
    writer.writerow([
        "frame", "track_id", "x1", "y1", "x2", "y2",
        "center_x", "center_y", "width", "height",
        "confidence"
    ])

start = time.perf_counter()

try:
    for frame_no, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read {path}")

        img = letterbox(frame, IMG_SIZE, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            prediction = model(img, augment=False)[0]
            predictions = non_max_suppression(
                prediction,
                conf_thres=CONF_THRES,
                iou_thres=IOU_THRES,
                classes=[person_id],
                agnostic=False,
            )

        boxes, scores = [], []

        for det in predictions:
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], frame.shape
                ).round()

                for *xyxy, conf, cls in det:
                    boxes.append([float(x) for x in xyxy])
                    scores.append(float(conf))

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32)

        for threshold, tracker in trackers.items():
            keep = scores >= threshold

            if keep.any():
                detections = sv.Detections(
                    xyxy=boxes[keep],
                    confidence=scores[keep],
                    class_id=np.zeros(int(keep.sum()), dtype=int),
                )
            else:
                detections = sv.Detections.empty()

            tracked = tracker.update_with_detections(detections)

            for box, confidence, track_id in zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.tracker_id,
            ):
                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(float, box)
                writers[threshold].writerow([
                    frame_no, int(track_id),
                    x1, y1, x2, y2,
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1, y2 - y1,
                    float(confidence),
                ])

        if frame_no % 50 == 0:
            print(f"Processed {frame_no}/{len(frames)} frames", flush=True)
finally:
    for f in files.values():
        f.close()

elapsed = time.perf_counter() - start
print(f"\nCompleted {len(frames)} frames in {elapsed:.1f}s")
print("Saved results in:", OUTPUT_DIR)
