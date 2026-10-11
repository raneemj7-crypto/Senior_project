"""Diagnostic MOT20-01 comparison, not official MOTChallenge scores.

Uses the reference evaluator's marked-pedestrian filtering and frame-wise
IoU association. MOT20 ignored-region handling is not fully implemented.
"""

from pathlib import Path

import pandas as pd
import motmetrics as mm


PROJECT_DIR = Path(__file__).resolve().parent
GT_PATH = (
    Path.home()
    / "Desktop/senior-project-storage/raw/mot20/train/MOT20-01/gt/gt.txt"
)
OUTPUT_DIR = PROJECT_DIR / "matching_experiments"
RESULTS_PATH = OUTPUT_DIR / "metrics_matching.csv"
TRACKERS = {
    f"ByteTrack_match_{threshold:.1f}": OUTPUT_DIR / f"tracking_match_{threshold:.1f}.csv"
    for threshold in (0.7, 0.8, 0.9)
}
NUM_FRAMES = 429
IOU_THRESHOLD = 0.5
DIAGNOSTIC_NOTICE = (
    "Diagnostic evaluation, not official MOTChallenge scores: "
    "MOT20 ignored-region handling is not fully implemented."
)
GT_COLUMNS = [
    "frame", "id", "x", "y", "w", "h", "mark", "class_id", "visibility",
]
METRICS = [
    "idf1", "idp", "idr", "num_switches", "num_fragmentations",
    "mota", "precision", "recall", "num_misses", "num_false_positives",
]


def evaluate_tracker(csv_path, gt):
    """Apply the same association method to every frame, including empty ones."""
    pred = pd.read_csv(csv_path)
    acc = mm.MOTAccumulator(auto_id=False)

    for frame in range(1, NUM_FRAMES + 1):
        g = gt[gt["frame"] == frame]
        p = pred[pred["frame"] == frame]
        gt_ids = g["id"].astype(int).tolist()
        pred_ids = p["track_id"].astype(int).tolist()
        gt_boxes = g[["x", "y", "w", "h"]].to_numpy()
        pred_boxes = p[["x1", "y1", "width", "height"]].to_numpy().astype(float)

        # motmetrics gates the distance (1 - IoU), so IoU must be >= 0.5.
        distances = mm.distances.iou_matrix(
            gt_boxes, pred_boxes, max_iou=1.0 - IOU_THRESHOLD,
        )
        acc.update(gt_ids, pred_ids, distances, frameid=frame)

    return acc


def main():
    print(DIAGNOSTIC_NOTICE)
    if RESULTS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {RESULTS_PATH}")
    for path in [GT_PATH, *TRACKERS.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input not found: {path}")

    gt = pd.read_csv(GT_PATH, header=None, names=GT_COLUMNS)
    # Same filtering as evaluate_tracking_mot20.py, shared by all configurations.
    gt = gt[(gt["mark"] == 1) & (gt["class_id"] == 1)].copy()
    print(f"Evaluating frames 1–{NUM_FRAMES}; minimum IoU: {IOU_THRESHOLD}")

    accumulators = []
    for name, filename in TRACKERS.items():
        print(f"Evaluating {name} ...")
        accumulators.append(evaluate_tracker(filename, gt))

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        names=list(TRACKERS),
        metrics=METRICS,
        generate_overall=False,
    )
    print("\nDIAGNOSTIC TRACKING COMPARISON — MOT20-01")
    print(DIAGNOSTIC_NOTICE)
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={
            "idf1": "IDF1", "idp": "ID precision", "idr": "ID recall",
            "num_switches": "ID switches", "num_fragmentations": "Fragmentations",
            "mota": "MOTA", "precision": "Tracking precision",
            "recall": "Tracking recall", "num_misses": "False negatives",
            "num_false_positives": "False positives",
        },
    ))

    # Keep the diagnostic qualification with the exported scores as well.
    summary.index.name = "configuration"
    summary["evaluation_note"] = DIAGNOSTIC_NOTICE
    summary["num_frames"] = NUM_FRAMES
    summary["iou_threshold"] = IOU_THRESHOLD
    summary.to_csv(RESULTS_PATH, mode="x")
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
