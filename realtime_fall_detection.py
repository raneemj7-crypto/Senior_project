from collections import deque

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

# Probability threshold from the LSTM
FALL_THRESHOLD = 0.5

# === MODIFIED FOR OUR PROJECT ===
# Instead of confirming a fall after only 3 consecutive
# predictions, keep a short prediction history and evaluate it.
DECISION_HISTORY = 15

# At least 75% of the recent predictions must be Fall
POSITIVE_RATIO_REQUIRED = 0.75

# The average probability must also be at least 0.60
AVERAGE_PROBABILITY_REQUIRED = 0.60


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
    Same normalization used during training.
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

        normalized_x = np.zeros(
            17,
            dtype=np.float32
        )

        normalized_y = np.zeros(
            17,
            dtype=np.float32
        )

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
# PREDICT ONE 30-FRAME SEQUENCE
# ============================================================

def predict_sequence(pose_buffer):

    sequence = np.array(
        pose_buffer,
        dtype=np.float32
    )

    # Shape:
    # (30, 17, 3)
    #
    # Convert to:
    # (30, 51)
    sequence = sequence.reshape(
        SEQUENCE_LENGTH,
        -1
    )

    # Add batch dimension:
    # (1, 30, 51)
    tensor = torch.tensor(
        sequence,
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
# REAL-TIME STATE
# ============================================================

# Keeps the most recent 30 valid pose frames
pose_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

# === MODIFIED FOR OUR PROJECT ===
# Stores recent LSTM probabilities
probability_history = deque(
    maxlen=DECISION_HISTORY
)

fall_confirmed = False

current_probability = 0.0

average_probability = 0.0

positive_ratio = 0.0


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():

    print("Error: Could not open webcam.")

    exit()

print()
print("Webcam opened successfully.")
print("Press ESC to stop.")
print("Press R to reset a confirmed fall.")


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Could not read webcam frame."
        )

        break


    # ========================================================
    # YOLOV8-POSE
    # ========================================================

    result = pose_model(
        frame,
        conf=0.5,
        verbose=False
    )[0]

    plotted_frame = result.plot()


    # ========================================================
    # EXTRACT FIRST DETECTED PERSON
    # ========================================================
    #
    # For this prototype we assume only one person
    # is being tested.
    #
    # Later ByteTrack can connect each Track ID
    # to its own pose buffer.

    if (
        result.keypoints is not None
        and len(result.keypoints) > 0
    ):

        pose = result.keypoints[0]

        if (
            pose.xy is not None
            and pose.conf is not None
            and len(pose.xy) > 0
            and len(pose.conf) > 0
        ):

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

            if len(keypoints) >= 17:

                normalized_pose = normalize_pose(
                    keypoints,
                    confs
                )

                pose_buffer.append(
                    normalized_pose
                )


    # ========================================================
    # RUN LSTM WHEN 30 POSES ARE AVAILABLE
    # ========================================================

    if len(pose_buffer) == SEQUENCE_LENGTH:

        current_probability = predict_sequence(
            pose_buffer
        )

        # === MODIFIED FOR OUR PROJECT ===
        probability_history.append(
            current_probability
        )


        # ====================================================
        # IMPROVED FALL CONFIRMATION LOGIC
        # ====================================================

        if len(probability_history) > 0:

            probabilities = np.array(
                probability_history
            )

            positive_ratio = np.mean(
                probabilities >= FALL_THRESHOLD
            )

            average_probability = np.mean(
                probabilities
            )


        # Only make a final decision once the
        # full decision history is available.
        if len(probability_history) == DECISION_HISTORY:

            if (
                positive_ratio >= POSITIVE_RATIO_REQUIRED
                and
                average_probability >= AVERAGE_PROBABILITY_REQUIRED
            ):

                fall_confirmed = True


    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    if fall_confirmed:

        status_text = "FALL DETECTED"

        status_color = (
            0,
            0,
            255
        )

    elif len(pose_buffer) < SEQUENCE_LENGTH:

        status_text = (
            f"Collecting pose history: "
            f"{len(pose_buffer)}/{SEQUENCE_LENGTH}"
        )

        status_color = (
            0,
            165,
            255
        )

    elif len(probability_history) < DECISION_HISTORY:

        status_text = (
            f"Evaluating: "
            f"{len(probability_history)}/{DECISION_HISTORY}"
        )

        status_color = (
            0,
            165,
            255
        )

    else:

        status_text = "NORMAL"

        status_color = (
            0,
            255,
            0
        )


    # ========================================================
    # DISPLAY TEXT
    # ========================================================

    cv2.putText(
        plotted_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3
    )


    cv2.putText(
        plotted_frame,
        f"Fall probability: "
        f"{current_probability:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )


    cv2.putText(
        plotted_frame,
        f"Pose buffer: "
        f"{len(pose_buffer)}/{SEQUENCE_LENGTH}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        plotted_frame,
        f"Avg probability: "
        f"{average_probability:.2f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        plotted_frame,
        f"Positive ratio: "
        f"{positive_ratio:.2f}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        plotted_frame,
        f"Decision history: "
        f"{len(probability_history)}/{DECISION_HISTORY}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Real-Time Fall Detection",
        plotted_frame
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # ESC = quit
    if key == 27:

        print(
            "ESC pressed. Exiting."
        )

        break


    # R = reset
    if key == ord("r"):

        fall_confirmed = False

        pose_buffer.clear()

        probability_history.clear()

        current_probability = 0.0

        average_probability = 0.0

        positive_ratio = 0.0

        print(
            "Fall state reset."
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()