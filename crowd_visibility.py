import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent

DATA = BASE / "results" / "training_data"
OUTPUT = BASE / "results" / "crowd_training"
OUTPUT.mkdir(parents=True, exist_ok=True)

X = np.load(DATA / "X_train_sample.npy")
y = np.load(DATA / "y_train_sample.npy")

# COCO 17-keypoint format
UPPER_BODY = list(range(11))
HEAD_SHOULDERS = [0, 1, 2, 3, 4, 5, 6]

def mask_keypoints(sequence, visible_points):
    masked = sequence.copy()

    for point in range(17):
        if point not in visible_points:
            masked[:, point, :] = 0

    return masked

X_crowd = []
y_crowd = []
visibility_types = []

for sequence, label in zip(X, y):

    # Original full-body sequence
    X_crowd.append(sequence)
    y_crowd.append(label)
    visibility_types.append("full_body")

    # Simulate hidden lower bodyno 
    X_crowd.append(
        mask_keypoints(sequence, UPPER_BODY)
    )
    y_crowd.append(label)
    visibility_types.append("upper_body")

    # Simulate only head and shoulders visible
    X_crowd.append(
        mask_keypoints(sequence, HEAD_SHOULDERS)
    )
    y_crowd.append(label)
    visibility_types.append("head_shoulders")

X_crowd = np.array(X_crowd, dtype=np.float32)
y_crowd = np.array(y_crowd, dtype=np.int64)

np.save(OUTPUT / "X_crowd.npy", X_crowd)
np.save(OUTPUT / "y_crowd.npy", y_crowd)
np.save(
    OUTPUT / "visibility_types.npy",
    np.array(visibility_types)
)

print("\nCROWD DATA READY!")
print("Input shape:", X_crowd.shape)
print("Fall examples:", int(sum(y_crowd == 1)))
print("Non-fall examples:", int(sum(y_crowd == 0)))
print("Visibility scenarios:", set(visibility_types))