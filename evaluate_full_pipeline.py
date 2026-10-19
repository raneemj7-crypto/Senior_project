from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

POSE_MODEL_PATH = "yolov8n-pose.pt"
LSTM_MODEL_PATH = "best_fall_detection_lstm.pth"

# Same sequence length used during training
SEQUENCE_LENGTH = 30

# Same sliding-window step used when creating LSTM sequences
WINDOW_STEP = 10

# Untouched URFD test split
TEST_ROOT = Path(
    "/Users/reemahumayed/desktop/UDFD/split_dataset/test"
)

# True labels
LABEL_MAP = {
    "fall": 1,
    "non_fall": 0
}


# ============================================================
# DEFINE THE SAME LSTM ARCHITECTURE USED DURING TRAINING
# ============================================================

class FallDetectionLSTM(nn.Module):

    def __init__(
        self,
        input_size=51,
        hidden_size=64,
        num_layers=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        # Use the representation from the final frame
        final_output = lstm_output[:, -1, :]

        prediction = self.classifier(
            final_output
        )

        return prediction


# ============================================================
# LOAD YOLOV8-POSE
# ============================================================

pose_model = YOLO(
    POSE_MODEL_PATH
)

print(
    f"YOLOv8-Pose loaded successfully: "
    f"{POSE_MODEL_PATH}"
)


# ============================================================
# LOAD TRAINED LSTM
# ============================================================

lstm_model = FallDetectionLSTM()

lstm_model.load_state_dict(
    torch.load(
        LSTM_MODEL_PATH,
        map_location="cpu"
    )
)

lstm_model.eval()

print(
    f"LSTM loaded successfully: "
    f"{LSTM_MODEL_PATH}"
)

print()
print("Both models are ready.")


# ============================================================
# NORMALIZE ONE POSE FRAME
# ============================================================

def normalize_pose(keypoints, confs):
    """
    Same normalization used during LSTM training.

    Each pose contains:
    17 keypoints × (x, y, confidence)
    """

    x = keypoints[:, 0].astype(
        np.float32
    )

    y = keypoints[:, 1].astype(
        np.float32
    )

    conf = confs.astype(
        np.float32
    )

    # Only use sufficiently visible keypoints
    visible = conf > 0.3

    if np.sum(visible) >= 2:

        visible_x = x[visible]
        visible_y = y[visible]

        min_x = np.min(
            visible_x
        )

        max_x = np.max(
            visible_x
        )

        min_y = np.min(
            visible_y
        )

        max_y = np.max(
            visible_y
        )

        width = (
            max_x - min_x
        )

        height = (
            max_y - min_y
        )

        # Avoid division by zero
        if width < 1:
            width = 1

        if height < 1:
            height = 1

        normalized_x = (
            x - min_x
        ) / width

        normalized_y = (
            y - min_y
        ) / height

    else:

        # If there are not enough usable keypoints,
        # use zero coordinates for that frame.
        normalized_x = np.zeros(
            17,
            dtype=np.float32
        )

        normalized_y = np.zeros(
            17,
            dtype=np.float32
        )

    # Keep confidence scores unchanged
    pose = np.stack(
        [
            normalized_x,
            normalized_y,
            conf
        ],
        axis=1
    )

    return pose


# ============================================================
# EXTRACT POSES FROM ONE URFD SEQUENCE
# ============================================================

def extract_sequence_poses(sequence_folder):

    frame_files = sorted(
        list(
            sequence_folder.glob("*.png")
        )
        +
        list(
            sequence_folder.glob("*.jpg")
        )
        +
        list(
            sequence_folder.glob("*.jpeg")
        )
    )

    poses = []

    for frame_path in frame_files:

        frame = cv2.imread(
            str(frame_path)
        )

        if frame is None:
            continue

        result = pose_model(
            frame,
            conf=0.5,
            verbose=False
        )[0]

        # No detected pose
        if (
            result.keypoints is None
            or len(result.keypoints) == 0
        ):
            continue

        # URFD is mainly single-person data.
        # For now, use the first detected pose.
        pose = result.keypoints[0]

        if (
            pose.xy is None
            or pose.conf is None
            or len(pose.xy) == 0
            or len(pose.conf) == 0
        ):
            continue

        keypoints = (
            pose.xy[0]
            .cpu()
            .numpy()
        )

        confs = (
            pose.conf[0]
            .cpu()
            .numpy()
        )

        if len(keypoints) < 17:
            continue

        normalized_pose = normalize_pose(
            keypoints,
            confs
        )

        poses.append(
            normalized_pose
        )

    return np.array(
        poses,
        dtype=np.float32
    )


# ============================================================
# CREATE 30-FRAME WINDOWS
# ============================================================

def create_windows(poses):

    windows = []

    # The LSTM requires at least 30 usable pose frames
    if len(poses) < SEQUENCE_LENGTH:
        return windows

    for start in range(
        0,
        len(poses) - SEQUENCE_LENGTH + 1,
        WINDOW_STEP
    ):

        end = (
            start
            + SEQUENCE_LENGTH
        )

        window = poses[
            start:end
        ]

        windows.append(
            window
        )

    return windows


# ============================================================
# PREDICT ONE 30-FRAME WINDOW
# ============================================================

def predict_window(window):

    # Current shape:
    # (30, 17, 3)
    #
    # LSTM input shape:
    # (30, 51)

    window = window.reshape(
        SEQUENCE_LENGTH,
        -1
    )

    # Add batch dimension:
    # (1, 30, 51)
    tensor = torch.tensor(
        window,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        output = lstm_model(
            tensor
        )

        probability = torch.sigmoid(
            output
        ).item()

    return probability


# ============================================================
# COMBINE WINDOW PREDICTIONS INTO ONE SEQUENCE PREDICTION
# ============================================================

def predict_sequence(sequence_folder):

    poses = extract_sequence_poses(
        sequence_folder
    )

    windows = create_windows(
        poses
    )

    if len(windows) == 0:

        return (
            None,
            [],
            len(poses)
        )

    probabilities = []

    for window in windows:

        probability = predict_window(
            window
        )

        probabilities.append(
            probability
        )

    # ========================================================
    # SEQUENCE-LEVEL DECISION RULE
    # ========================================================
    #
    # A collapse may happen during only one part of the
    # sequence, so we use the highest fall probability
    # produced by any 30-frame window.
    #
    # If any window strongly indicates Fall,
    # the full sequence is considered to contain a Fall.

    sequence_probability = max(
        probabilities
    )

    sequence_prediction = (
        1
        if sequence_probability >= 0.5
        else 0
    )

    return (
        sequence_prediction,
        probabilities,
        len(poses)
    )


# ============================================================
# EVALUATE THE FULL TEST SET
# ============================================================

results = []

skipped_sequences = []


for class_name in [
    "fall",
    "non_fall"
]:

    class_folder = (
        TEST_ROOT
        / class_name
    )

    true_label = LABEL_MAP[
        class_name
    ]

    sequence_folders = sorted(
        [
            p
            for p
            in class_folder.iterdir()
            if p.is_dir()
        ]
    )

    for sequence_folder in sequence_folders:

        print()
        print(
            f"Processing: "
            f"{sequence_folder.name}"
        )

        (
            prediction,
            probabilities,
            pose_count
        ) = predict_sequence(
            sequence_folder
        )


        # ====================================================
        # HANDLE SEQUENCES WITH FEWER THAN 30 USABLE POSES
        # ====================================================

        if prediction is None:

            print(
                f"Skipped: only "
                f"{pose_count} usable pose frames"
            )

            skipped_sequences.append(
                {
                    "sequence_name":
                        sequence_folder.name,

                    "true_label":
                        true_label,

                    "usable_pose_frames":
                        pose_count
                }
            )

            continue


        max_probability = max(
            probabilities
        )

        predicted_name = (
            "Fall"
            if prediction == 1
            else "Non-fall"
        )

        true_name = (
            "Fall"
            if true_label == 1
            else "Non-fall"
        )


        print(
            f"Ground truth: "
            f"{true_name}"
        )

        print(
            f"Prediction: "
            f"{predicted_name}"
        )

        print(
            f"Max fall probability: "
            f"{max_probability:.4f}"
        )

        print(
            f"Usable pose frames: "
            f"{pose_count}"
        )

        print(
            f"30-frame windows: "
            f"{len(probabilities)}"
        )


        results.append(
            {
                "sequence_name":
                    sequence_folder.name,

                "true_label":
                    true_label,

                "prediction":
                    prediction,

                "max_probability":
                    max_probability,

                "usable_pose_frames":
                    pose_count,

                "window_count":
                    len(probabilities)
            }
        )


# ============================================================
# FULL EVALUATION SUMMARY
# ============================================================

print()
print("=" * 60)
print("FULL PIPELINE TEST SUMMARY")
print("=" * 60)


# Make sure at least one sequence was evaluated
if len(results) == 0:

    print(
        "No sequences could be evaluated."
    )

else:

    y_true = [
        result["true_label"]
        for result in results
    ]

    y_pred = [
        result["prediction"]
        for result in results
    ]


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    # Force confusion matrix to always use:
    #
    # 0 = Non-fall
    # 1 = Fall
    #
    # This ensures a 2×2 matrix even if one class
    # happens to have no prediction errors.

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Sequences evaluated: "
        f"{len(results)}"
    )

    print(
        f"Sequences skipped: "
        f"{len(skipped_sequences)}"
    )

    print()

    print(
        f"Accuracy:  "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"F1-score:  "
        f"{f1:.4f}"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("Confusion Matrix:")
    print()

    print(
        "                 Predicted"
    )

    print(
        "               Non-fall  Fall"
    )

    print(
        f"Actual Non-fall    "
        f"{tn:>3}      {fp:>3}"
    )

    print(
        f"Actual Fall        "
        f"{fn:>3}      {tp:>3}"
    )


    # ========================================================
    # ERROR COUNTS
    # ========================================================

    print()

    print(
        f"True Positives:  "
        f"{tp}"
    )

    print(
        f"True Negatives:  "
        f"{tn}"
    )

    print(
        f"False Positives: "
        f"{fp}"
    )

    print(
        f"False Negatives: "
        f"{fn}"
    )

    print()

    print(
        f"Missed falls: "
        f"{fn}"
    )

    print(
        f"False alarms: "
        f"{fp}"
    )


# ============================================================
# REPORT SKIPPED SEQUENCES
# ============================================================

if len(skipped_sequences) > 0:

    print()
    print("=" * 60)
    print("SKIPPED SEQUENCES")
    print("=" * 60)

    for sequence in skipped_sequences:

        label_name = (
            "Fall"
            if sequence["true_label"] == 1
            else "Non-fall"
        )

        print(
            f"{sequence['sequence_name']} | "
            f"Ground truth: {label_name} | "
            f"Usable poses: "
            f"{sequence['usable_pose_frames']}"
        )