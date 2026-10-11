"""Compare ByteTrack matching thresholds using one YOLOv5 pass per frame."""

from contextlib import ExitStack
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

PROJECT_DIR = Path(__file__).resolve().parent
ROOT = Path.home() / "Desktop/senior-project-storage/raw/mot20/train/MOT20-01"
FRAMES_DIR = ROOT / "img1"
OUTPUT_DIR = PROJECT_DIR / "matching_experiments"
WEIGHTS = PROJECT_DIR / "weights/crowdhuman_yolov5m.pt"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
MATCHING_THRESHOLDS = (0.7, 0.8, 0.9)
CSV_COLUMNS = [
    "frame", "track_id", "x1", "y1", "x2", "y2",
    "center_x", "center_y", "width", "height", "confidence",
]


def main():
    output_paths = {
        threshold: OUTPUT_DIR / f"tracking_match_{threshold:.1f}.csv"
        for threshold in MATCHING_THRESHOLDS
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite experimental outputs: "
            + ", ".join(str(path) for path in existing)
        )

    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"MOT20-01 JPEG frames not found: {FRAMES_DIR}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS}")

    trackers = {
        threshold: sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=threshold,
            frame_rate=25,
            minimum_consecutive_frames=1,
        )
        for threshold in MATCHING_THRESHOLDS
    }

    device = torch.device("cpu")
    model = attempt_load(str(WEIGHTS), map_location=device)
    model.eval()
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, "module") else model.names
    person_id = list(names).index("person") if not isinstance(names, dict) else next(
        key for key, value in names.items() if value == "person"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    start = time.perf_counter()
    with ExitStack() as stack:
        writers = {}
        for threshold, path in output_paths.items():
            # Exclusive creation also protects against files created after preflight.
            handle = stack.enter_context(path.open("x", newline=""))
            writers[threshold] = csv.writer(handle)
            writers[threshold].writerow(CSV_COLUMNS)

        for frame_no, path in enumerate(frames, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not read {path}")

            img = letterbox(frame, IMG_SIZE, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
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
                # Identical values, independent arrays: trackers may mutate detections.
                if len(scores):
                    detections = sv.Detections(
                        xyxy=boxes.copy(),
                        confidence=scores.copy(),
                        class_id=np.full(len(scores), person_id, dtype=int),
                    )
                else:
                    detections = sv.Detections.empty()

                # Advance every tracker even when no people were detected.
                tracked = tracker.update_with_detections(detections)
                for box, confidence, track_id in zip(
                    tracked.xyxy, tracked.confidence, tracked.tracker_id
                ):
                    if track_id is None:
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    writers[threshold].writerow([
                        frame_no, int(track_id), x1, y1, x2, y2,
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        x2 - x1, y2 - y1, float(confidence),
                    ])

            if frame_no % 50 == 0:
                print(f"Processed {frame_no}/{len(frames)} frames", flush=True)

    elapsed = time.perf_counter() - start
    print(f"\nCompleted {len(frames)} frames in {elapsed:.1f}s")
    print("Saved results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
