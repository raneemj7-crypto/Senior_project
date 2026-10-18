from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE / "results" / "training_data"
OUTPUT = BASE / "stgcn_project" / "data"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Load original data
X = np.load(SOURCE / "X_train_sample.npy")
y = np.load(SOURCE / "y_train_sample.npy")
event_ids = np.load(SOURCE / "event_ids.npy")

print("Original data:", X.shape)

# -----------------------------------
# 1. SPLIT BY INDEPENDENT EVENTS
# -----------------------------------

# Both camera views of a fall share the same event ID.
# Keep every window from an event in the same split.

test_events = {
    "fall-04",
    "adl-19-cam0-rgb",
}

val_events = {
    "fall-03",
    "adl-16-cam0-rgb",
}

test_mask = np.isin(event_ids, list(test_events))
val_mask = np.isin(event_ids, list(val_events))
train_mask = ~(test_mask | val_mask)

splits = {
    "train": (X[train_mask], y[train_mask]),
    "val": (X[val_mask], y[val_mask]),
    "test": (X[test_mask], y[test_mask]),
}

# -----------------------------------
# 2. VISIBILITY AUGMENTATION
# -----------------------------------

def augment_visibility(sequence):
    """
    Return three versions:
    - Full body
    - Upper body
    - Head and shoulders
    """

    full_body = sequence.copy()

    upper_body = sequence.copy()
    upper_body[:, 11:17, :] = 0

    head_shoulders = sequence.copy()
    head_shoulders[:, 7:17, :] = 0

    return [
        full_body,
        upper_body,
        head_shoulders,
    ]


# -----------------------------------
# 3. FORMAT FOR ST-GCN++
# -----------------------------------

def convert_format(data):
    """
    From: (N, 30, 17, 3)
    To:   (N, 3, 30, 17, 1)

    N = sequences
    C = x, y, confidence
    T = frames
    V = joints
    M = people
    """

    return np.transpose(data, (0, 3, 1, 2))[:, :, :, :, None]


# -----------------------------------
# 4. SAVE DATASETS
# -----------------------------------

for split_name, (split_X, split_y) in splits.items():

    if split_name == "train":
        augmented_X = []
        augmented_y = []

        for sequence, label in zip(split_X, split_y):
            versions = augment_visibility(sequence)

            for version in versions:
                augmented_X.append(version)
                augmented_y.append(label)

        split_X = np.asarray(augmented_X, dtype=np.float32)
        split_y = np.asarray(augmented_y, dtype=np.int64)

    else:
        split_X = split_X.astype(np.float32)
        split_y = split_y.astype(np.int64)

    converted_X = convert_format(split_X)

    np.save(OUTPUT / f"X_{split_name}.npy", converted_X)
    np.save(OUTPUT / f"y_{split_name}.npy", split_y)

    falls = int(np.sum(split_y == 1))
    nonfalls = int(np.sum(split_y == 0))

    print(f"\n{split_name.upper()}")
    print("Shape:", converted_X.shape)
    print("Fall:", falls)
    print("Non-fall:", nonfalls)

print("\nST-GCN++ DATA PREPARATION COMPLETE!")