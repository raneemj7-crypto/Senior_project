from pathlib import Path

import numpy as np
import torch
from torch import nn

from visibility_detection import detect_visibility

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "results" / "trained_model" / "fall_detection_lstm.pth"

# --------------------------------
# 1. DEFINE LSTM MODEL
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
        return self.fc(output[:, -1, :])


# --------------------------------
# 2. LOAD TRAINED MODEL
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

print("LSTM model loaded successfully!")


# --------------------------------
# 3. VISIBILITY-AWARE PREDICTION
# --------------------------------

def predict_fall(sequence):
    """
    Input: 30 frames, 17 keypoints, 3 values
    Output: FALL / NO FALL / UNCERTAIN
    """

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (30, 17, 3):
        raise ValueError(
            f"Expected (30, 17, 3), got {sequence.shape}"
        )

    visibility_results = [
        detect_visibility(frame)
        for frame in sequence
    ]

    uncertain_frames = visibility_results.count("uncertain")

    if uncertain_frames > 15:
        return {
            "prediction": "UNCERTAIN",
            "confidence": None,
            "visibility": "uncertain"
        }

    visibility = max(
        set(visibility_results),
        key=visibility_results.count
    )

    input_data = sequence.reshape(1, 30, 51)

    tensor = torch.tensor(
        input_data,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        output = model(tensor)

        probabilities = torch.softmax(
            output, dim=1
        )[0]

        prediction = int(torch.argmax(probabilities))
        confidence = float(probabilities[prediction])

    return {
        "prediction": "FALL" if prediction == 1 else "NO FALL",
        "confidence": round(confidence, 4),
        "visibility": visibility
    }


# --------------------------------
# 4. TEST ON AN ACTUAL FALL
# --------------------------------

if __name__ == "__main__":

    DATA = BASE / "results" / "training_data"

    X = np.load(DATA / "X_train_sample.npy")
    y = np.load(DATA / "y_train_sample.npy")

    # Find a sequence labeled as a fall
    fall_indices = np.where(y == 1)[0]

    if len(fall_indices) == 0:
        raise RuntimeError("No fall sequences found!")

    # Choose the first fall sequence
    sample = X[fall_indices[0]]

    print("\n================================")
    print("TESTING AN ACTUAL FALL SEQUENCE")
    print("================================")

    print("\nTEST 1 — FULL BODY")
    print(predict_fall(sample))

    # Hide legs
    upper_body = sample.copy()
    upper_body[:, 13:17, :] = 0

    print("\nTEST 2 — UPPER BODY")
    print(predict_fall(upper_body))

    # Keep head and shoulders only
    head_shoulders = sample.copy()
    head_shoulders[:, 7:17, :] = 0

    print("\nTEST 3 — HEAD AND SHOULDERS")
    print(predict_fall(head_shoulders))

    # Hide all keypoints
    uncertain = np.zeros_like(sample)

    print("\nTEST 4 — INSUFFICIENT VISIBILITY")
    print(predict_fall(uncertain))

    print("\nFALL TEST COMPLETE!")