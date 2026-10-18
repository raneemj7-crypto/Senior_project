from pathlib import Path

import numpy as np
import torch

from stgcn_model import STGCNPlusPlus


# -----------------------------------
# 1. PATHS AND SETTINGS
# -----------------------------------

BASE = Path(__file__).resolve().parent

DATA = BASE / "data"
MODEL_PATH = BASE / "trained_model" / "best_stgcn_model.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CONF_THRESHOLD = 0.35

print("Using device:", device)


# -----------------------------------
# 2. LOAD TRAINED MODEL
# -----------------------------------

model = STGCNPlusPlus().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Trained model loaded successfully!")


# -----------------------------------
# 3. VISIBILITY CONDITIONS
# -----------------------------------

def apply_visibility(sequence, condition):

    sequence = sequence.copy()

    if condition == "full_body":
        pass

    elif condition == "upper_body":
        # Hide hips and legs
        sequence[:, :, :, 11:17, :] = 0

    elif condition == "head_shoulders":
        # Keep only keypoints 0-6
        sequence[:, :, :, 7:17, :] = 0

    elif condition == "insufficient":
        sequence[:] = 0

    return sequence


# -----------------------------------
# 4. VISIBILITY-AWARE PREDICTION
# -----------------------------------

def predict_batch(sequences):

    # Check how many joints are visible
    confidence = sequences[:, 2, :, :, 0]

    visible = confidence >= CONF_THRESHOLD

    # Count frames with at least 3 visible joints
    visible_per_frame = visible.sum(axis=2)

    reliable_frames = (
        visible_per_frame >= 3
    ).sum(axis=1)

    # Require at least 15 reliable frames
    reliable = reliable_frames >= 15

    predictions = np.full(
        len(sequences),
        -1,
        dtype=np.int64
    )

    if reliable.any():

        inputs = torch.from_numpy(
            sequences[reliable]
        ).float().to(device)

        with torch.no_grad():
            outputs = model(inputs)

            predicted = outputs.argmax(dim=1)

        predictions[reliable] = (
            predicted.cpu().numpy()
        )

    return predictions


# -----------------------------------
# 5. EVALUATION
# -----------------------------------

def evaluate_condition(X, y, condition):

    modified_X = apply_visibility(X, condition)

    predictions = predict_batch(modified_X)

    uncertain = predictions == -1
    decided = ~uncertain

    falls = y == 1
    nonfalls = y == 0

    correct_falls = np.sum(
        (predictions == 1) & falls
    )

    missed_falls = np.sum(
        (predictions == 0) & falls
    )

    false_alarms = np.sum(
        (predictions == 1) & nonfalls
    )

    correct_nonfalls = np.sum(
        (predictions == 0) & nonfalls
    )

    uncertain_falls = np.sum(
        uncertain & falls
    )

    uncertain_nonfalls = np.sum(
        uncertain & nonfalls
    )

    accuracy = (
        np.mean(predictions[decided] == y[decided])
        if decided.any()
        else None
    )

    print("\n" + "=" * 45)
    print("CONDITION:", condition.upper())
    print("=" * 45)

    print("Total sequences:", len(y))

    print("\nFALL DETECTION")
    print("Correctly detected falls:", correct_falls)
    print("Missed falls:", missed_falls)
    print("Uncertain falls:", uncertain_falls)

    print("\nNON-FALL DETECTION")
    print("Correct non-falls:", correct_nonfalls)
    print("False alarms:", false_alarms)
    print("Uncertain non-falls:", uncertain_nonfalls)

    print("\nUNCERTAINTY")
    print("Total uncertain:", uncertain.sum())

    if accuracy is not None:
        print(f"Accuracy on decided sequences: {accuracy:.2%}")
    else:
        print("Accuracy: N/A (all predictions uncertain)")


# -----------------------------------
# 6. RUN ALL TESTS
# -----------------------------------

if __name__ == "__main__":

    X_test = np.load(DATA / "X_test.npy")
    y_test = np.load(DATA / "y_test.npy")

    print("\nTest dataset:", X_test.shape)
    print("Fall sequences:", np.sum(y_test == 1))
    print("Non-fall sequences:", np.sum(y_test == 0))

    conditions = [
        "full_body",
        "upper_body",
        "head_shoulders",
        "insufficient"
    ]

    for condition in conditions:
        evaluate_condition(
            X_test,
            y_test,
            condition
        )

    print("\nVISIBILITY TESTING COMPLETE!")