
from pathlib import Path
import csv
import time
import configparser

import cv2
import numpy as np
import torch
import supervision as sv

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

# ---------------------------------------
# SETTINGS
# ---------------------------------------
ROOT = Path.home() / (
    "Desktop/senior-project-storage/raw/mot20/"
    "train/MOT20-02"
)
FRAMES_DIR = ROOT / "img1"
WEIGHTS = "weights/crowdhuman_yolov5m.pt"
OUTPUT_DIR = Path("validation_mot20_02")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = torch.device("cpu")

frames = sorted(FRAMES_DIR.glob("*.jpg"))
assert frames, "MOT20-02 frames not found"

cfg = configparser.ConfigParser()
cfg.read(ROOT / "seqinfo.ini")
fps = cfg.getint("Sequence", "frameRate", fallback=25)

# ---------------------------------------
# LOAD YOLOv5 ONCE
# ---------------------------------------
model = attempt_load(WEIGHTS, map_location=DEVICE)
model.eval()
stride = int(model.stride.max())

names = (
    model.module.names
    if hasattr(model, "module")
    else model.names
)

person_id = next(
    i for i, name in enumerate(names)
    if name == "person"
)

# ---------------------------------------
# CREATE BOTH TRACKERS
# ---------------------------------------
configs = {
    "A": 30,
    "B": 60,
}

trackers = {}
files = {}
writers = {}

for name, buffer_size in configs.items():
    trackers[name] = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=buffer_size,
        minimum_matching_threshold=0.8,
        frame_rate=fps,
        minimum_consecutive_frames=1,
    )

    file = open(
        OUTPUT_DIR / f"tracking_{name}.csv",
        "w",
        newline=""
    )
    files[name] = file
    writers[name] = csv.writer(file)
    writers[name].writerow([
        "frame", "track_id",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "width", "height", "confidence"
    ])

# ---------------------------------------
# DETECTION + TRACKING
# ---------------------------------------
start = time.perf_counter()

try:
    for frame_number, image_path in enumerate(frames, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read: {image_path}")

        img = letterbox(
            frame, IMG_SIZE, stride=stride
        )[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(DEVICE)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(
                pred,
                CONF_THRES,
                IOU_THRES,
                classes=[person_id],
                agnostic=False
            )

        boxes = []
        confidences = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:],
                    det[:, :4],
                    frame.shape
                ).round()

                for *xyxy, conf, cls in det:
                    boxes.append([
                        float(x) for x in xyxy
                    ])
                    confidences.append(float(conf))

        if boxes:
            detections = sv.Detections(
                xyxy=np.asarray(boxes, dtype=np.float32),
                confidence=np.asarray(
                    confidences, dtype=np.float32
                ),
                class_id=np.zeros(
                    len(boxes), dtype=int
                ),
            )
        else:
            detections = sv.Detections.empty()

        # Same detections for both trackers
        for name, tracker in trackers.items():
            tracked = tracker.update_with_detections(
                detections
            )

            for box, conf, tid in zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.tracker_id,
            ):
                if tid is None:
                    continue

                x1, y1, x2, y2 = map(float, box)

                writers[name].writerow([
                    frame_number,
                    int(tid),
                    x1, y1, x2, y2,
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1,
                    float(conf),
                ])

        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number}/{len(frames)} frames",
                flush=True
            )
finally:
    for file in files.values():
        file.close()

elapsed = time.perf_counter() - start

print("\nValidation tracking completed!")
print(f"Total frames: {len(frames)}")
print(f"Total time: {elapsed:.1f} seconds")
print(f"Average FPS: {len(frames)/elapsed:.2f}")
print("Results saved to:", OUTPUT_DIR)
