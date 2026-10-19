from pathlib import Path
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

TEST_ROOT = Path(
    "/Users/reemahumayed/desktop/UDFD/split_dataset/test"
)

SEQUENCE_LENGTH = 30
FALL_THRESHOLD = 0.5

# Playback speed
FRAME_DELAY_MS = 40

LABEL_MAP = {
    "fall": 1,
    "non_fall": 0
}


# ============================================================
# LSTM ARCHITECTURE
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

        return self.classifier(
            final_output
        )


# ============================================================
# LOAD MODELS
# ============================================================

pose_model = YOLO(
    POSE_MODEL_PATH
)

print(
    f"YOLOv8-Pose loaded: {POSE_MODEL_PATH}"
)

lstm_model = FallDetectionLSTM()

lstm_model.load_state_dict(
    torch.load(
        LSTM_MODEL_PATH,
        map_location="cpu"
    )
)

lstm_model.eval()

print(
    f"LSTM loaded: {LSTM_MODEL_PATH}"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_pose(keypoints, confs):

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

        width = max_x - min_x
        height = max_y - min_y

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

    return np.stack(
        [
            normalized_x,
            normalized_y,
            conf
        ],
        axis=1
    )


# ============================================================
# LSTM PREDICTION
# ============================================================

def predict_sequence(pose_buffer):

    sequence = np.array(
        pose_buffer,
        dtype=np.float32
    )

    sequence = sequence.reshape(
        SEQUENCE_LENGTH,
        -1
    )

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
# COLLECT TEST SEQUENCES
# ============================================================

test_sequences = []

for class_name in [
    "fall",
    "non_fall"
]:

    class_folder = (
        TEST_ROOT
        / class_name
    )

    for sequence_folder in sorted(
        [
            p
            for p in class_folder.iterdir()
            if p.is_dir()
        ]
    ):

        test_sequences.append(
            (
                class_name,
                sequence_folder
            )
        )


# ============================================================
# PLAY EACH TEST SEQUENCE
# ============================================================

for class_name, sequence_folder in test_sequences:

    true_label = LABEL_MAP[
        class_name
    ]

    true_name = (
        "FALL"
        if true_label == 1
        else "NON-FALL"
    )

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

    pose_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    current_probability = None

    print()
    print("=" * 60)
    print(
        f"Playing: {sequence_folder.name}"
    )
    print(
        f"Ground truth: {true_name}"
    )
    print("=" * 60)


    for frame_index, frame_path in enumerate(
        frame_files
    ):

        frame = cv2.imread(
            str(frame_path)
        )

        if frame is None:
            continue


        # ====================================================
        # YOLOV8-POSE
        # ====================================================

        result = pose_model(
            frame,
            conf=0.5,
            verbose=False
        )[0]

        display_frame = result.plot()


        # ====================================================
        # EXTRACT FIRST PERSON
        # ====================================================

        valid_pose = False

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

                    valid_pose = True


        # ====================================================
        # RUN LSTM
        # ====================================================

        if len(pose_buffer) == SEQUENCE_LENGTH:

            current_probability = predict_sequence(
                pose_buffer
            )


        # ====================================================
        # MODEL STATUS
        # ====================================================

        if current_probability is None:

            prediction_text = (
                f"Collecting: "
                f"{len(pose_buffer)}/{SEQUENCE_LENGTH}"
            )

            prediction_color = (
                0,
                165,
                255
            )

        else:

            predicted_fall = (
                current_probability
                >= FALL_THRESHOLD
            )

            if predicted_fall:

                prediction_text = (
                    "Prediction: FALL"
                )

                prediction_color = (
                    0,
                    0,
                    255
                )

            else:

                prediction_text = (
                    "Prediction: NON-FALL"
                )

                prediction_color = (
                    0,
                    255,
                    0
                )


        # ====================================================
        # DISPLAY INFORMATION
        # ====================================================

        cv2.putText(
            display_frame,
            f"Sequence: {sequence_folder.name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Ground truth: {true_name}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            prediction_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            prediction_color,
            2
        )


        if current_probability is not None:

            cv2.putText(
                display_frame,
                f"Fall probability: "
                f"{current_probability:.3f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                prediction_color,
                2
            )


        cv2.putText(
            display_frame,
            f"Pose buffer: "
            f"{len(pose_buffer)}/{SEQUENCE_LENGTH}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Frame: "
            f"{frame_index + 1}/{len(frame_files)}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        if not valid_pose:

            cv2.putText(
                display_frame,
                "No usable pose detected",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )


        # ====================================================
        # SHOW LIKE A VIDEO
        # ====================================================

        cv2.imshow(
            "Test Pipeline Visualization",
            display_frame
        )


        key = (
            cv2.waitKey(
                FRAME_DELAY_MS
            )
            & 0xFF
        )


        # ESC = quit everything
        if key == 27:

            cv2.destroyAllWindows()
            exit()


        # N = skip to next sequence
        if key == ord("n"):

            break


    # ========================================================
    # PAUSE BETWEEN SEQUENCES
    # ========================================================

    print(
        "Sequence finished."
    )

    print(
        "Press N in the video window "
        "to move through sequences."
    )


cv2.destroyAllWindows()

print()
print("All test sequences finished.")