from pathlib import Path

import numpy as np
import torch

from stgcn_model import STGCNPlusPlus


# --------------------------------
# SETTINGS
# --------------------------------

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "trained_model"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CONF_THRESHOLD = 0.35


# --------------------------------
# LOAD BOTH MODELS
# --------------------------------

def load_model(filename):
    model = STGCNPlusPlus().to(device)

    checkpoint = torch.load(
        MODELS / filename,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


model_a = load_model("fullbody_baseline.pth")
model_b = load_model("best_stgcn_model.pth")

print("Both models loaded successfully!")


# --------------------------------
# SIMULATE VISIBILITY CONDITIONS
# --------------------------------

def apply_visibility(X, condition):
    X = X.copy()

    if condition == "upper_body":
        # Hide hips and legs
        X[:, :, :, 11:17, :] = 0

    elif condition == "head_shoulders":
        # Hide everything except head and shoulders
        X[:, :, :, 7:17, :] = 0

    elif condition == "insufficient":
        X[:] = 0

    return X


# --------------------------------
# PREDICTION WITH UNCERTAIN OPTION
# --------------------------------

def predict(model, X):
    confidence = X[:, 2, :, :, 0]

    visible = confidence >= CONF_THRESHOLD
    visible_per_frame = visible.sum(axis=2)

    reliable_frames = (
        visible_per_frame >= 3
    ).sum(axis=1)

    reliable = reliable_frames >= 15

    predictions = np.full(
        len(X), -1, dtype=np.int64
    )

    if reliable.any():
        inputs = torch.from_numpy(
            X[reliable]
        ).float().to(device)

        with torch.no_grad():
            outputs = model(inputs)
            predictions[reliable] = (
                outputs.argmax(dim=1).cpu().numpy()
            )

    return predictions


# --------------------------------
# EVALUATE ONE MODEL
# --------------------------------

def evaluate(model, X, y):
    predictions = predict(model, X)

    fall_mask = y == 1
    nonfall_mask = y == 0
    decided = predictions != -1

    detected_falls = int(np.sum(
        (predictions == 1) & fall_mask
    ))

    missed_falls = int(np.sum(
        (predictions == 0) & fall_mask
    ))

    false_alarms = int(np.sum(
        (predictions == 1) & nonfall_mask
    ))

    uncertain = int(np.sum(predictions == -1))

    accuracy = (
        np.mean(predictions[decided] == y[decided])
        if decided.any()
        else float("nan")
    )

    return {
        "falls": detected_falls,
        "missed": missed_falls,
        "false_alarms": false_alarms,
        "uncertain": uncertain,
        "accuracy": accuracy
    }


# --------------------------------
# COMPARE BOTH MODELS
# --------------------------------

X_test = np.load(DATA / "X_test.npy")
y_test = np.load(DATA / "y_test.npy")

print("\nTEST DATASET")
print("Total sequences:", len(y_test))
print("Falls:", int(np.sum(y_test == 1)))
print("Non-falls:", int(np.sum(y_test == 0)))

conditions = [
    "full_body",
    "upper_body",
    "head_shoulders",
    "insufficient"
]

for condition in conditions:

    X_modified = apply_visibility(X_test, condition)

    result_a = evaluate(model_a, X_modified, y_test)
    result_b = evaluate(model_b, X_modified, y_test)

    print("\n" + "=" * 55)
    print("VISIBILITY:", condition.upper())
    print("=" * 55)

    print(f"{'Metric':<25} {'Model A':<12} {'Model B':<12}")
    print("-" * 55)

    for key, label in [
        ("falls", "Falls detected"),
        ("missed", "Missed falls"),
        ("false_alarms", "False alarms"),
        ("uncertain", "Uncertain")
    ]:
        print(
            f"{label:<25} "
            f"{result_a[key]:<12} "
            f"{result_b[key]:<12}"
        )

    print(
        f"{'Accuracy (decided)':<25} "
        f"{result_a['accuracy']:.2%}      "
        f"{result_b['accuracy']:.2%}"
    )

print("\nMODEL COMPARISON COMPLETE!")