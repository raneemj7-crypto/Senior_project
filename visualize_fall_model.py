from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

# --------------------------------
# 1. PATHS
# --------------------------------

BASE = Path(__file__).resolve().parent

# Select a non-fall recording from the TEST dataset
NON_FALL_FOLDER = BASE / "split_dataset" / "test" / "non_fall"

sequences = sorted(
    p for p in NON_FALL_FOLDER.iterdir()
    if p.is_dir()
)

if not sequences:
    raise RuntimeError("No non-fall sequences found in the test dataset!")

FRAMES = sequences[0]

MODEL_PATH = BASE / "results" / "trained_model" / "fall_detection_lstm.pth"

OUTPUT = BASE / "results" / "visualization"
OUTPUT.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = OUTPUT / "non_fall_demo.mp4"

print("Testing non-fall recording:", FRAMES.name)

# --------------------------------
# 2. LOAD YOLO POSE
# --------------------------------

pose_model = YOLO(str(BASE / "yolov8n-pose.pt"))

# --------------------------------
# 3. DEFINE LSTM MODEL
# --------------------------------

class FallDetectionLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=51,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)


# --------------------------------
# 4. LOAD TRAINED MODEL
# --------------------------------

device = torch.device("cpu")

model = FallDetectionLSTM().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Trained fall model loaded!")

# --------------------------------
# 5. PREPARE VIDEO
# --------------------------------

frames = sorted(FRAMES.glob("*.png"))

if not frames:
    raise RuntimeError("No frames found in the selected recording!")

first_frame = cv2.imread(str(frames[0]))

if first_frame is None:
    raise RuntimeError("Could not read the first frame!")

height, width = first_frame.shape[:2]

video = cv2.VideoWriter(
    str(VIDEO_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (width, height)
)

if not video.isOpened():
    raise RuntimeError("Could not create output video!")

pose_buffer = deque(maxlen=30)

# Track predictions
fall_frames = 0
non_fall_frames = 0

# --------------------------------
# 6. PROCESS FRAMES
# --------------------------------

for frame_number, frame_path in enumerate(frames):

    image = cv2.imread(str(frame_path))

    if image is None:
        print("Skipping unreadable frame:", frame_path.name)
        continue

    result = pose_model.predict(
        image,
        conf=0.25,
        verbose=False
    )[0]

    if len(result.boxes) > 0 and result.keypoints is not None:
        idx = int(result.boxes.conf.argmax().item())
        keypoints = result.keypoints.data[idx].cpu().numpy().copy()
    else:
        keypoints = np.zeros((17, 3), dtype=np.float32)

    # Normalize coordinates
    keypoints[:, 0] /= width
    keypoints[:, 1] /= height

    pose_buffer.append(keypoints)

    # Draw detected skeleton
    annotated = result.plot()

    label = "Collecting frames..."
    color = (255, 255, 255)
    confidence = 0

    # Predict after collecting 30 frames
    if len(pose_buffer) == 30:

        sequence = np.array(
            pose_buffer,
            dtype=np.float32
        ).reshape(1, 30, 51)

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

            prediction = int(torch.argmax(probabilities))

            confidence = float(probabilities[prediction]) * 100

        if prediction == 1:
            label = "FALL"
            color = (0, 0, 255)
            fall_frames += 1
        else:
            label = "NO FALL"
            color = (0, 255, 0)
            non_fall_frames += 1

    # --------------------------------
    # 7. DISPLAY PREDICTION
    # --------------------------------

    cv2.rectangle(
        annotated,
        (0, 0),
        (width, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    if len(pose_buffer) == 30:
        cv2.putText(
            annotated,
            f"Confidence: {confidence:.1f}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    video.write(annotated)

    if frame_number % 20 == 0:
        print(f"Processing frame {frame_number + 1}/{len(frames)}")

video.release()

# --------------------------------
# 8. PRINT TEST RESULTS
# --------------------------------

total_predictions = fall_frames + non_fall_frames

print("\nNON-FALL TEST COMPLETE!")
print("Recording:", FRAMES.name)
print("Total predictions:", total_predictions)
print("NO FALL predictions:", non_fall_frames)
print("FALL predictions (false alarms):", fall_frames)

if total_predictions > 0:
    false_alarm_rate = 100 * fall_frames / total_predictions
    print(f"Percentage of windows flagged as FALL: {false_alarm_rate:.2f}%")

print("\nVIDEO CREATED!")
print("Saved to:", VIDEO_PATH)

