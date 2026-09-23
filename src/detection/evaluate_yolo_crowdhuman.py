import csv
import json
import random
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# YOLO vs CrowdHuman - Threshold Evaluation
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

MODEL_NAME = "yolo26n.pt"

# Run inference once at this threshold.
INFERENCE_CONFIDENCE = 0.05

# Thresholds we want to compare.
CONFIDENCE_THRESHOLDS = [0.10, 0.25, 0.40]

# IoU required to count a detection as a match.
IOU_THRESHOLD = 0.50

# Number of validation images.
NUM_IMAGES = 100

# Fixed seed makes the experiment reproducible.
RANDOM_SEED = 42


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

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def match_predictions(
    predictions,
    ground_truths
):
    """
    Greedy one-to-one matching using highest IoU.
    """

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

    true_positives = 0

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

        true_positives += 1

    false_positives = (
        len(predictions)
        - true_positives
    )

    false_negatives = (
        len(ground_truths)
        - true_positives
    )

    return (
        true_positives,
        false_positives,
        false_negatives
    )


def calculate_metrics(
    true_positives,
    false_positives,
    false_negatives
):

    precision_denominator = (
        true_positives
        + false_positives
    )

    recall_denominator = (
        true_positives
        + false_negatives
    )

    precision = (
        true_positives
        / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positives
        / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return precision, recall, f1


def main():

    print("=" * 60)
    print("YOLO vs CrowdHuman - Threshold Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    print("\nLoading CrowdHuman annotations...")

    annotations = load_annotations()

    print(
        f"Loaded {len(annotations)} annotation records."
    )

    # --------------------------------------------------------
    # Select reproducible random sample
    # --------------------------------------------------------

    image_files = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    random.seed(RANDOM_SEED)

    sample_size = min(
        NUM_IMAGES,
        len(image_files)
    )

    selected_images = random.sample(
        image_files,
        sample_size
    )

    print(
        f"\nSelected {len(selected_images)} "
        f"random validation images."
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading YOLO model...")

    model = YOLO(MODEL_NAME)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    all_predictions = []

    print(
        "\nRunning inference..."
    )

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        print(
            f"Processing {index}/{len(selected_images)}: "
            f"{image_path.name}"
        )

        results = model.predict(
            source=str(image_path),
            classes=[0],
            conf=INFERENCE_CONFIDENCE,
            verbose=False
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()

            for box, confidence in zip(
                boxes,
                confidences
            ):

                predictions.append(
                    {
                        "box": box,
                        "confidence": confidence
                    }
                )

        all_predictions.append(
            (
                image_path,
                predictions
            )
        )

    print("\nInference completed.")

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_table = []

    print("\n" + "=" * 60)
    print("THRESHOLD RESULTS")
    print("=" * 60)

    for confidence_threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        total_tp = 0
        total_fp = 0
        total_fn = 0

        total_ground_truth = 0
        total_predictions = 0

        for image_path, predictions in (
            all_predictions
        ):

            image_id = image_path.stem

            record = annotations.get(
                image_id
            )

            if record is None:
                continue

            ground_truths = get_ground_truth_boxes(
                record
            )

            filtered_predictions = [
                prediction["box"]
                for prediction in predictions
                if prediction["confidence"]
                >= confidence_threshold
            ]

            tp, fp, fn = match_predictions(
                filtered_predictions,
                ground_truths
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            total_ground_truth += len(
                ground_truths
            )

            total_predictions += len(
                filtered_predictions
            )

        precision, recall, f1 = calculate_metrics(
            total_tp,
            total_fp,
            total_fn
        )

        row = {
            "confidence_threshold":
                confidence_threshold,

            "ground_truth":
                total_ground_truth,

            "predictions":
                total_predictions,

            "true_positives":
                total_tp,

            "false_positives":
                total_fp,

            "false_negatives":
                total_fn,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1
        }

        results_table.append(row)

        print(
            f"\nConfidence threshold: "
            f"{confidence_threshold:.2f}"
        )

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

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        RESULTS_DIR
        / "yolo_threshold_evaluation.csv"
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

    print("\n" + "=" * 60)
    print("Experiment completed.")
    print(
        f"Results saved to:\n{output_file}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()