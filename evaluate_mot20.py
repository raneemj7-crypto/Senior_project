
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

# Paths
ROOT = Path.home() / "Desktop/senior-project-storage/raw/mot20/train/MOT20-01"
GT_FILE = ROOT / "gt/gt.txt"
PRED_DIR = Path("runs/yolo26/mot20_01/labels")

WIDTH, HEIGHT = 1920, 1080
IOU_THRESHOLD = 0.5

def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)

    return intersection / np.maximum(
        area_a[:, None] + area_b[None, :] - intersection, 1e-9
    )

# Load labeled pedestrian ground truth
gt_by_frame = defaultdict(list)
ignored_by_frame = defaultdict(list)

with GT_FILE.open() as f:
    for line in f:
        values = line.strip().split(",")
        if len(values) < 8:
            continue

        frame = int(values[0])
        x, y, w, h = map(float, values[2:6])
        mark = int(float(values[6]))
        cls = int(float(values[7]))

        box = [x, y, x + w, y + h]

        if mark == 1 and cls == 1:
            gt_by_frame[frame].append(box)
        else:
            ignored_by_frame[frame].append(box)

# Load YOLO person predictions
pred_by_frame = defaultdict(list)

for file in PRED_DIR.glob("*.txt"):
    frame = int(file.stem)

    for line in file.read_text().splitlines():
        values = line.split()

        if not values or int(values[0]) != 0:
            continue

        cx, cy, w, h = map(float, values[1:5])
        pred_by_frame[frame].append([
            (cx - w / 2) * WIDTH,
            (cy - h / 2) * HEIGHT,
            (cx + w / 2) * WIDTH,
            (cy + h / 2) * HEIGHT,
        ])

TP = FP = FN = 0

for frame in range(1, 430):
    gt = gt_by_frame[frame]
    preds = pred_by_frame[frame]

    matches = []

    if gt and preds:
        ious = iou_matrix(gt, preds)
        rows, cols = linear_sum_assignment(-ious)
        matches = [
            (r, c) for r, c in zip(rows, cols)
            if ious[r, c] >= IOU_THRESHOLD
        ]

    matched_preds = {c for _, c in matches}

    TP += len(matches)
    FN += len(gt) - len(matches)

    # Exclude unmatched predictions significantly
    # overlapping ignored annotations
    for idx, pred in enumerate(preds):
        if idx in matched_preds:
            continue

        ignored = ignored_by_frame[frame]

        if ignored:
            overlaps = iou_matrix([pred], ignored)[0]
            if np.max(overlaps) >= 0.5:
                continue

        FP += 1

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("\nYOLOv5 CrowdHuman — MOT20-01")
print("=" * 38)
print("True positives: ", TP)
print("False positives:", FP)
print("False negatives:", FN)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print("=" * 38)
print("Diagnostic evaluation at IoU 0.50")
print("Not official MOTChallenge metrics or AP50")
