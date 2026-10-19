import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

POSE_MODEL_PATH = "yolov8n-pose.pt"
LSTM_MODEL_PATH = "best_fall_detection_lstm.pth"

SEQUENCE_LENGTH = 30


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

        # Same LSTM structure used during training
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final binary classification layer
        self.classifier = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        # Process all frames in the sequence
        lstm_output, _ = self.lstm(x)

        # Use the final frame representation
        final_output = lstm_output[:, -1, :]

        # Produce one score for Fall / Non-fall
        prediction = self.classifier(final_output)

        return prediction


# ============================================================
# LOAD YOLOV8-POSE
# ============================================================

pose_model = YOLO(POSE_MODEL_PATH)

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

# Put model in evaluation mode
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
    Normalize one frame relative to the person's own skeleton.

    IMPORTANT:
    This matches the same normalization used when we created
    the LSTM training sequences.

    Each frame contains:
    17 keypoints × (x, y, confidence)
    """

    # Separate x, y, and confidence
    x = keypoints[:, 0].astype(np.float32)
    y = keypoints[:, 1].astype(np.float32)
    conf = confs.astype(np.float32)

    # Only use reasonably visible keypoints
    visible = conf > 0.3

    if np.sum(visible) >= 2:

        visible_x = x[visible]
        visible_y = y[visible]

        min_x = np.min(visible_x)
        max_x = np.max(visible_x)

        min_y = np.min(visible_y)
        max_y = np.max(visible_y)

        width = max_x - min_x
        height = max_y - min_y

        # Avoid division by zero
        if width < 1:
            width = 1

        if height < 1:
            height = 1

        # Normalize coordinates relative to the person's body
        normalized_x = (x - min_x) / width
        normalized_y = (y - min_y) / height

    else:

        # If not enough reliable keypoints are visible,
        # use zeros for the coordinates
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

    # Final shape:
    # (17 keypoints, 3 values)
    return pose


print("Normalization function ready.")