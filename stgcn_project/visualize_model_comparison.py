from pathlib import Path
import re

import cv2
import numpy as np
import torch

from stgcn_model import STGCNPlusPlus


# =====================================
# 1. PATHS
# =====================================

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent

DATA = BASE / "data"
MODELS = BASE / "trained_model"
OUTPUT = BASE / "visualizations"

OUTPUT.mkdir(parents=True, exist_ok=True)

SOURCE = PROJECT / "results" / "training_data"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CONF_THRESHOLD = 0.35

print("Using device:", device)


# =====================================
# 2. LOAD BOTH TRAINED MODELS
# =====================================

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


# =====================================
# 3. LOAD ORIGINAL TEST WINDOWS
# =====================================

X_original = np.load(
    SOURCE / "X_train_sample.npy"
)

y_original = np.load(
    SOURCE / "y_train_sample.npy"
)

event_ids = np.load(
    SOURCE / "event_ids.npy"
)

sequence_names = np.load(
    SOURCE / "sequence_names.npy"
)

# These are the same fall-04 windows
# used in our held-out ST-GCN test.

mask = (
    (event_ids == "fall-04")
    & (y_original == 1)
)

X_falls = X_original[mask]
fall_names = sequence_names[mask]

print("Fall windows found:", len(X_falls))

if len(X_falls) != 8:
    print(
        "WARNING: Expected 8 fall windows, "
        f"but found {len(X_falls)}."
    )


# =====================================
# 4. SKELETON CONNECTIONS
# =====================================

COCO_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# =====================================
# 5. VISIBILITY CONDITIONS
# =====================================

def apply_visibility(sequence, condition):

    result = sequence.copy()

    if condition == "upper_body":
        result[:, 11:17, :] = 0

    elif condition == "head_shoulders":
        result[:, 7:17, :] = 0

    elif condition == "insufficient":
        result[:] = 0

    return result


# =====================================
# 6. REPRODUCE MODEL PREDICTIONS
# =====================================

def predict(model, sequence):

    # Sequence shape: (30, 17, 3)

    confidence = sequence[:, :, 2]

    visible = confidence >= CONF_THRESHOLD

    visible_per_frame = visible.sum(axis=1)

    reliable_frames = np.sum(
        visible_per_frame >= 3
    )

    if reliable_frames < 15:
        return "UNCERTAIN"

    # Convert to ST-GCN input:
    # (1, 3, 30, 17, 1)

    input_data = np.transpose(
        sequence,
        (2, 0, 1)
    )[None, :, :, :, None]

    tensor = torch.from_numpy(
        input_data.astype(np.float32)
    ).to(device)

    with torch.no_grad():
        output = model(tensor)

        prediction = int(
            output.argmax(dim=1).item()
        )

    return "FALL" if prediction == 1 else "NO FALL"


# =====================================
# 7. LOCATE ORIGINAL FRAMES
# =====================================

def parse_sequence_name(name):

    name = str(name)

    match = re.match(
        r"^(fall-\d+-cam\d+-rgb)_(\d+)_(\d+)$",
        name
    )

    if not match:
        raise ValueError(
            f"Cannot read sequence name: {name}"
        )

    recording = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))

    return recording, start, end


def find_recording_folder(recording):

    candidates = [
        PROJECT / "split_dataset" / "train" / "fall" / recording,
        PROJECT / "split_dataset" / "val" / "fall" / recording,
        PROJECT / "split_dataset" / "test" / "fall" / recording,
    ]

    for folder in candidates:
        if folder.exists():
            return folder

    raise FileNotFoundError(
        f"Recording folder not found: {recording}"
    )


def natural_sort_key(path):

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_frames(recording, start, end):

    folder = find_recording_folder(recording)

    frames = sorted(
        list(folder.glob("*.png"))
        + list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg")),
        key=natural_sort_key
    )

    # The original preparation script used
    # zero-based frame positions.

    selected = frames[start:end]

    if len(selected) != 30:
        raise ValueError(
            f"{recording}: expected 30 frames, "
            f"found {len(selected)}"
        )

    return selected


# =====================================
# 8. DRAW SKELETON
# =====================================

def draw_skeleton(image, keypoints):

    height, width = image.shape[:2]

    visible = (
        keypoints[:, 2] >= CONF_THRESHOLD
    )

    for i, j in COCO_EDGES:

        if visible[i] and visible[j]:

            x1 = int(keypoints[i, 0] * width)
            y1 = int(keypoints[i, 1] * height)

            x2 = int(keypoints[j, 0] * width)
            y2 = int(keypoints[j, 1] * height)

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

    for index, point in enumerate(keypoints):

        if not visible[index]:
            continue

        x = int(point[0] * width)
        y = int(point[1] * height)

        cv2.circle(
            image,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    return image


# =====================================
# 9. CREATE ONE VIDEO PANEL
# =====================================

PANEL_WIDTH = 400
PANEL_HEIGHT = 360
HEADER_HEIGHT = 105


def create_panel(
    image,
    keypoints,
    title,
    prediction_a,
    prediction_b
):

    image = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT)
    )

    image = draw_skeleton(
        image,
        keypoints
    )

    panel = np.zeros(
        (
            PANEL_HEIGHT + HEADER_HEIGHT,
            PANEL_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    panel[HEADER_HEIGHT:] = image

    cv2.putText(
        panel,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    def prediction_color(prediction):

        if prediction == "FALL":
            return (0, 0, 255)

        if prediction == "NO FALL":
            return (0, 255, 0)

        return (0, 255, 255)

    cv2.putText(
        panel,
        f"Model A: {prediction_a}",
        (12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        prediction_color(prediction_a),
        2
    )

    cv2.putText(
        panel,
        f"Model B: {prediction_b}",
        (12, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        prediction_color(prediction_b),
        2
    )

    return panel


# =====================================
# 10. GENERATE COMPARISON VIDEO
# =====================================

OUTPUT_PATH = (
    OUTPUT / "model_comparison_fall04.mp4"
)

conditions = [
    ("full_body", "FULL BODY"),
    ("upper_body", "UPPER BODY"),
    ("head_shoulders", "HEAD + SHOULDERS")
]

video_width = PANEL_WIDTH * 3
video_height = PANEL_HEIGHT + HEADER_HEIGHT + 65

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    10,
    (video_width, video_height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )

print("\nCreating comparison video...")

for window_index, (sequence, name) in enumerate(
    zip(X_falls, fall_names),
    start=1
):

    recording, start, end = parse_sequence_name(
        name
    )

    frame_paths = get_frames(
        recording,
        start,
        end
    )

    predictions = {}

    for condition, _ in conditions:

        modified = apply_visibility(
            sequence,
            condition
        )

        predictions[condition] = (
            predict(model_a, modified),
            predict(model_b, modified)
        )

    print(
        f"Window {window_index}/{len(X_falls)}: "
        f"{name}",
        flush=True
    )

    for frame_index, frame_path in enumerate(
        frame_paths
    ):

        original_frame = cv2.imread(
            str(frame_path)
        )

        if original_frame is None:
            raise RuntimeError(
                f"Cannot read: {frame_path}"
            )

        panels = []

        for condition, title in conditions:

            modified = apply_visibility(
                sequence,
                condition
            )

            pred_a, pred_b = predictions[
                condition
            ]

            panel = create_panel(
                original_frame.copy(),
                modified[frame_index],
                title,
                pred_a,
                pred_b
            )

            panels.append(panel)

        combined = np.hstack(panels)

        footer = np.zeros(
            (65, video_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            footer,
            (
                f"FALL-LABELED WINDOW "
                f"{window_index}/{len(X_falls)}"
                f"  |  {recording}"
                f"  |  Frames {start}-{end}"
            ),
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        final_frame = np.vstack(
            [combined, footer]
        )

        writer.write(final_frame)

writer.release()

print("\nVIDEO CREATED SUCCESSFULLY!")
print("Saved to:", OUTPUT_PATH)

print("\nIMPORTANT:")
print("The prediction is for each full 30-frame window.")
print("It is displayed throughout that window.")
print("This is NOT a frame-by-frame prediction.")