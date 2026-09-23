import csv
import json
import random
import time
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# YOLO Model + Resolution Comparison
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CrowdHuman_val"
    / "Images"
)

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "annotation_val.odgt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

# Same experimental settings for every model.
NUM_IMAGES = 100
RANDOM_SEED = 42

CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.50

# Models/resolutions to compare.
EXPERIMENTS = [
    {
        "name": "YOLO26n_640",
        "model": "yolo26n.pt",
        "imgsz": 640,
    },
    {
        "name": "YOLO26s_640",
        "model": "yolo26s.pt",
        "imgsz": 640,
    },
    {
        "name": "YOLO26n_960",
        "model": "yolo26n.pt",
        "imgsz": 960,
    },
]


def load_annotations():
    """Load CrowdHuman annotations."""

    annotations = {}

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            annotations[str(record["ID"])] = record

    return annotations


def get_ground_truth_boxes(record):
    """Extract full-body person boxes."""

    boxes = []

    for gtbox in record.get("gtboxes", []):

        if gtbox.get("tag") != "person":
            continue

        fbox = gtbox.get("fbox")

        if not fbox or len(fbox) != 4:
            continue

        x, y, width, height = map(
            float,
            fbox
        )

        if width <= 0 or height <= 0:
            continue

        boxes.append([
            x,
            y,
            x + width,
            y + height
        ])

    return boxes


def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    intersection = width * height

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def match_predictions(
    predictions,
    ground_truths
):
    """Greedy one-to-one matching using IoU."""

    possible_matches = []

    for pred_index, prediction in enumerate(
        predictions
    ):

        for gt_index, ground_truth in enumerate(
            ground_truths
        ):

            iou = calculate_iou(
                prediction,
                ground_truth
            )

            if iou >= IOU_THRESHOLD:

                possible_matches.append(
                    (
                        iou,
                        pred_index,
                        gt_index
                    )
                )

    possible_matches.sort(
        reverse=True
    )

    matched_predictions = set()
    matched_ground_truths = set()

    tp = 0

    for (
        iou,
        pred_index,
        gt_index
    ) in possible_matches:

        if pred_index in matched_predictions:
            continue

        if gt_index in matched_ground_truths:
            continue

        matched_predictions.add(
            pred_index
        )

        matched_ground_truths.add(
            gt_index
        )

        tp += 1

    fp = len(predictions) - tp
    fn = len(ground_truths) - tp

    return tp, fp, fn


def calculate_metrics(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return precision, recall, f1


def main():

    print("=" * 60)
    print("YOLO Model + Resolution Comparison")
    print("=" * 60)

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    annotations = load_annotations()

    print(
        f"\nLoaded {len(annotations)} annotation records."
    )

    # --------------------------------------------------------
    # Select same 100 images as previous experiment
    # --------------------------------------------------------

    image_files = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    random.seed(RANDOM_SEED)

    selected_images = random.sample(
        image_files,
        min(NUM_IMAGES, len(image_files))
    )

    print(
        f"Using {len(selected_images)} images."
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"IoU threshold: {IOU_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_table = []

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for experiment in EXPERIMENTS:

        name = experiment["name"]
        model_name = experiment["model"]
        image_size = experiment["imgsz"]

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {name}")
        print("=" * 60)

        print(
            f"Model: {model_name}"
        )

        print(
            f"Input size: {image_size}"
        )

        print("\nLoading model...")

        model = YOLO(model_name)

        print("Model loaded.")

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_ground_truth = 0
        total_predictions = 0

        inference_times = []

        for index, image_path in enumerate(
            selected_images,
            start=1
        ):

            record = annotations.get(
                image_path.stem
            )

            if record is None:
                continue

            ground_truths = get_ground_truth_boxes(
                record
            )

            start_time = time.perf_counter()

            results = model.predict(
                source=str(image_path),
                classes=[0],
                conf=CONFIDENCE_THRESHOLD,
                imgsz=image_size,
                verbose=False
            )

            elapsed = time.perf_counter() - start_time

            inference_times.append(
                elapsed
            )

            result = results[0]

            predictions = []

            if result.boxes is not None:

                predictions = (
                    result.boxes.xyxy.tolist()
                )

            tp, fp, fn = match_predictions(
                predictions,
                ground_truths
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            total_ground_truth += len(
                ground_truths
            )

            total_predictions += len(
                predictions
            )

            if index % 10 == 0:
                print(
                    f"Processed "
                    f"{index}/{len(selected_images)}"
                )

        precision, recall, f1 = calculate_metrics(
            total_tp,
            total_fp,
            total_fn
        )

        average_time = (
            sum(inference_times)
            / len(inference_times)
        )

        average_fps = (
            1 / average_time
            if average_time > 0
            else 0
        )

        row = {
            "experiment": name,
            "model": model_name,
            "image_size": image_size,
            "ground_truth": total_ground_truth,
            "predictions": total_predictions,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_inference_seconds": average_time,
            "approx_fps": average_fps,
        }

        results_table.append(row)

        print("\nRESULTS")

        print(
            f"Ground truth: {total_ground_truth}"
        )

        print(
            f"Predictions:  {total_predictions}"
        )

        print(
            f"TP: {total_tp}"
        )

        print(
            f"FP: {total_fp}"
        )

        print(
            f"FN: {total_fn}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1-score:  {f1:.4f}"
        )

        print(
            f"Avg inference: "
            f"{average_time:.4f} sec/image"
        )

        print(
            f"Approx FPS: "
            f"{average_fps:.2f}"
        )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    output_file = (
        RESULTS_DIR
        / "yolo_model_resolution_comparison.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = list(
            results_table[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results_table
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for row in results_table:

        print(
            f"\n{row['experiment']}"
        )

        print(
            f"  Precision: "
            f"{row['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{row['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{row['f1']:.4f}"
        )

        print(
            f"  FPS:       "
            f"{row['approx_fps']:.2f}"
        )

    print("\n" + "=" * 60)
    print("Experiment completed.")
    print(
        f"Results saved to:\n{output_file}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()