import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "urfd_pose_data.csv"

# Number of consecutive frames per LSTM sample
SEQUENCE_LENGTH = 30

# Sliding window step
# 10 means:
# frames 0-29, then 10-39, then 20-49, etc.
WINDOW_STEP = 10

# Output files
TRAIN_OUTPUT = "train_sequences.npz"
VAL_OUTPUT = "val_sequences.npz"
TEST_OUTPUT = "test_sequences.npz"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Loaded:", INPUT_CSV)
print()
print(df["split"].value_counts())


# ============================================================
# KEYPOINT COLUMN DEFINITIONS
# ============================================================

# YOLOv8-Pose gives us 17 body keypoints.
# Each keypoint has:
# x coordinate
# y coordinate
# confidence

x_cols = [f"kpt_{i}_x" for i in range(17)]
y_cols = [f"kpt_{i}_y" for i in range(17)]
conf_cols = [f"kpt_{i}_conf" for i in range(17)]


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_pose(row):
    """
    Normalize one frame relative to the person's own skeleton.

    Instead of using raw image pixel coordinates, we:
    1. Find the visible body keypoints.
    2. Find the minimum and maximum x/y positions.
    3. Convert every keypoint to a value relative to that body box.

    This makes the pose less dependent on:
    - image resolution
    - where the person is in the image
    - how close the person is to the camera
    """

    x = row[x_cols].to_numpy(dtype=np.float32)
    y = row[y_cols].to_numpy(dtype=np.float32)
    conf = row[conf_cols].to_numpy(dtype=np.float32)

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

        normalized_x = (x - min_x) / width
        normalized_y = (y - min_y) / height

    else:
        # If not enough keypoints are visible,
        # keep the coordinates at zero.
        normalized_x = np.zeros(17, dtype=np.float32)
        normalized_y = np.zeros(17, dtype=np.float32)

    # Keep confidence scores unchanged.
    pose = np.stack(
        [
            normalized_x,
            normalized_y,
            conf
        ],
        axis=1
    )

    # Final shape for one frame:
    # (17 keypoints, 3 values)
    return pose


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

def create_sequences(split_name):

    split_df = df[df["split"] == split_name].copy()

    X = []
    y = []

    sequence_names = []

    # Process each original URFD sequence separately.
    # This prevents us from accidentally joining frames
    # from different videos/sequences together.
    for sequence_name, sequence_df in split_df.groupby("sequence_name"):

        # Make sure frames are in chronological order.
        sequence_df = sequence_df.sort_values("frame_index")

        true_label = int(sequence_df["true_label"].iloc[0])

        poses = []

        # Convert every frame into normalized skeleton data.
        for _, row in sequence_df.iterrows():

            pose = normalize_pose(row)

            poses.append(pose)

        poses = np.array(poses, dtype=np.float32)

        # We need at least 30 consecutive frames
        # to create one LSTM sample.
        if len(poses) < SEQUENCE_LENGTH:

            print(
                f"Skipping {sequence_name}: "
                f"only {len(poses)} frames"
            )

            continue

        # Create sliding 30-frame windows.
        for start in range(
            0,
            len(poses) - SEQUENCE_LENGTH + 1,
            WINDOW_STEP
        ):

            end = start + SEQUENCE_LENGTH

            sample = poses[start:end]

            X.append(sample)
            y.append(true_label)

            sequence_names.append(sequence_name)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    sequence_names = np.array(sequence_names)

    return X, y, sequence_names


# ============================================================
# BUILD TRAIN / VAL / TEST SETS
# ============================================================

X_train, y_train, train_names = create_sequences("train")
X_val, y_val, val_names = create_sequences("val")
X_test, y_test, test_names = create_sequences("test")


# ============================================================
# SAVE OUTPUT
# ============================================================

np.savez(
    TRAIN_OUTPUT,
    X=X_train,
    y=y_train,
    sequence_names=train_names
)

np.savez(
    VAL_OUTPUT,
    X=X_val,
    y=y_val,
    sequence_names=val_names
)

np.savez(
    TEST_OUTPUT,
    X=X_test,
    y=y_test,
    sequence_names=test_names
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("Finished creating LSTM sequences.")
print()

print("Train:")
print("X shape:", X_train.shape)
print("y shape:", y_train.shape)
print("Fall samples:", np.sum(y_train == 1))
print("Non-fall samples:", np.sum(y_train == 0))

print()

print("Validation:")
print("X shape:", X_val.shape)
print("y shape:", y_val.shape)
print("Fall samples:", np.sum(y_val == 1))
print("Non-fall samples:", np.sum(y_val == 0))

print()

print("Test:")
print("X shape:", X_test.shape)
print("y shape:", y_test.shape)
print("Fall samples:", np.sum(y_test == 1))
print("Non-fall samples:", np.sum(y_test == 0))

print()
print("Saved:")
print(TRAIN_OUTPUT)
print(VAL_OUTPUT)
print(TEST_OUTPUT)