
from pathlib import Path
import numpy as np
import pandas as pd
import motmetrics as mm

# =====================================
# CONFIGURATION
# =====================================

GT_PATH = (
    Path.home()
    / "Desktop/senior-project-storage/raw/mot20"
    / "train/MOT20-01/gt/gt.txt"
)

TRACKERS = {
    "YOLOv5_Baseline_A": "yolov5_mot20_tracking.csv",
    "YOLOv5_Tuned_B": "yolov5_mot20_tuned.csv",
    "YOLOv5_Tuned_C": "yolov5_mot20_C.csv",
}

NUM_FRAMES = 429
IOU_THRESHOLD = 0.5

# =====================================
# LOAD GROUND TRUTH
# =====================================

columns = [
    "frame", "id", "x", "y", "w", "h",
    "mark", "class_id", "visibility"
]

gt = pd.read_csv(GT_PATH, header=None, names=columns)

# Evaluate marked pedestrian annotations
gt = gt[
    (gt["mark"] == 1)
    & (gt["class_id"] == 1)
].copy()

print("Ground-truth observations:", len(gt))
print("Ground-truth identities:", gt["id"].nunique())

# =====================================
# EVALUATE TRACKER
# =====================================

def evaluate_tracker(csv_path):
    pred = pd.read_csv(csv_path)

    acc = mm.MOTAccumulator(auto_id=False)

    for frame in range(1, NUM_FRAMES + 1):
        g = gt[gt["frame"] == frame]
        p = pred[pred["frame"] == frame]

        gt_ids = g["id"].astype(int).tolist()
        pred_ids = p["track_id"].astype(int).tolist()

        gt_boxes = g[["x", "y", "w", "h"]].to_numpy()

        pred_boxes = p[
            ["x1", "y1", "width", "height"]
        ].to_numpy().astype(float)

        # Convert x1,y1,width,height to MOT xywh format
        distances = mm.distances.iou_matrix(
            gt_boxes,
            pred_boxes,
            max_iou=IOU_THRESHOLD,
        )

        acc.update(
            gt_ids,
            pred_ids,
            distances,
            frameid=frame,
        )

    return acc

# =====================================
# COMPUTE METRICS
# =====================================

accumulators = []
names = []

for name, filename in TRACKERS.items():
    print(f"Evaluating {name} ...")
    accumulators.append(evaluate_tracker(filename))
    names.append(name)

mh = mm.metrics.create()

metrics = [
    "idf1",
    "idp",
    "idr",
    "num_switches",
    "num_fragmentations",
    "mota",
    "motp",
    "precision",
    "recall",
    "num_matches",
    "num_misses",
    "num_false_positives",
]

summary = mh.compute_many(
    accumulators,
    names=names,
    metrics=metrics,
    generate_overall=False,
)

print("\nTRACKING COMPARISON — MOT20-01")
print("=" * 65)

print(mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names,
))

summary.to_csv("tracking_metrics_ABC.csv")
print("\nSaved: tracking_metrics_comparison.csv")
