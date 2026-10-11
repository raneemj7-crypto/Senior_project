"""Diagnostic head assistance; auxiliary observations are NOT MOT body tracks.

Run explicitly to process MOT20-01. Each run creates a new output directory.
ByteTrack receives only person detections and owns all person track IDs.
Head continuity is a conservative heuristic, not a verified identity label.
"""

import csv
import inspect
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


PROJECT_DIR = Path(__file__).resolve().parent
FRAMES_DIR = Path.home() / "Desktop/senior-project-storage/raw/mot20/train/MOT20-01/img1"
WEIGHTS = PROJECT_DIR / "weights/crowdhuman_yolov5m.pt"
OUTPUT_ROOT = PROJECT_DIR / "head_assisted_experiments"
NUM_FRAMES = 429
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
FPS = 25
TRACKER_SETTINGS = dict(
    track_activation_threshold=0.25, lost_track_buffer=30,
    minimum_matching_threshold=0.8, frame_rate=FPS,
    minimum_consecutive_frames=1,
)
# Deliberately conservative starting heuristics; tune only on separate data.
MIN_HEAD_CONF = 0.40
MIN_SCORE = 0.60
AMBIGUITY_MARGIN = 0.15
CONFIRM_BODY_PAIRS = 3
MAX_HEAD_ONLY_FRAMES = 5
BODY_COLOR = (0, 200, 0)
HEAD_COLOR = (255, 200, 0)
HEAD_ONLY_COLOR = (0, 140, 255)
BODY_COLUMNS = [
    "frame", "track_id", "x1", "y1", "x2", "y2", "center_x",
    "center_y", "width", "height", "confidence",
]
OBS_COLUMNS = [
    "frame", "track_id", "body_x1", "body_y1", "body_x2", "body_y2",
    "head_x1", "head_y1", "head_x2", "head_y2", "body_confidence",
    "head_confidence", "observation_type", "association_score",
    "track_margin", "head_margin", "confirmed_body_pairs",
    "frames_since_body", "temporal_consistent", "identity_verified",
    "eligible_for_body_evaluation",
]


def center(box):
    return (box[:2] + box[2:]) / 2


def area(box):
    return float(np.prod(np.maximum(box[2:] - box[:2], 0)))


def intersection(a, b):
    return float(np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0)))


def spatial_score(body, head):
    """Require most of the head inside the upper part of a plausible body."""
    bw, bh = body[2:] - body[:2]
    hw, hh = head[2:] - head[:2]
    if min(bw, bh, hw, hh) <= 0:
        return None
    relative = (center(head) - body[:2]) / (body[2:] - body[:2])
    containment = intersection(body, head) / area(head)
    if not (0.10 <= relative[0] <= 0.90 and 0 <= relative[1] <= 0.35):
        return None
    if not (0.08 <= hw / bw <= 0.70 and 0.04 <= hh / bh <= 0.35):
        return None
    if containment < 0.85:
        return None
    position = max(0.0, 1 - abs(relative[0] - 0.5) / 0.5)
    return 0.7 * containment + 0.3 * position


def temporal_score(previous, current):
    """Small adjacent-frame motion, overlap and similar head size are required."""
    if min(area(previous), area(current)) <= 0:
        return None
    ratio = area(current) / area(previous)
    motion = np.linalg.norm(center(current) - center(previous)) / max(
        float(np.linalg.norm(previous[2:] - previous[:2])), 1.0
    )
    overlap = intersection(previous, current)
    iou = overlap / (area(previous) + area(current) - overlap)
    if not 0.60 <= ratio <= 1.67 or motion > 0.60 or iou < 0.15:
        return None
    return 0.5 * iou + 0.5 * (1 - motion / 0.60)


def associate(scores):
    """Accept mutual best matches with margins on BOTH sides; no forced matches."""
    matches = {}
    if not scores.size:
        return matches
    for row in range(scores.shape[0]):
        col = int(np.argmax(scores[row]))
        score = scores[row, col]
        if score < MIN_SCORE or int(np.argmax(scores[:, col])) != row:
            continue
        row_alt = max(0.0, float(np.max(np.delete(scores[row], col), initial=-1)))
        col_alt = max(0.0, float(np.max(np.delete(scores[:, col], row), initial=-1)))
        row_margin, col_margin = score - row_alt, score - col_alt
        if min(row_margin, col_margin) >= AMBIGUITY_MARGIN:
            matches[row] = (col, float(score), float(row_margin), float(col_margin))
    return matches


def draw(frame, box, color, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main():
    expected = [FRAMES_DIR / f"{i:06d}.jpg" for i in range(1, NUM_FRAMES + 1)]
    if sorted(FRAMES_DIR.glob("*.jpg")) != expected:
        raise RuntimeError("Expected exactly MOT20-01 JPEG frames 000001–000429.")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    parameters = inspect.signature(sv.ByteTrack).parameters
    missing = set(TRACKER_SETTINGS) - set(parameters)
    if missing:
        raise RuntimeError(f"Incompatible Supervision ByteTrack API; missing: {sorted(missing)}")
    tracker = sv.ByteTrack(**TRACKER_SETTINGS)
    if not callable(getattr(tracker, "update_with_detections", None)):
        raise RuntimeError("Supervision must support update_with_detections().")
    if not callable(getattr(sv.Detections, "empty", None)):
        raise RuntimeError("Supervision must support Detections.empty().")

    device = torch.device("cpu")
    model = attempt_load(str(WEIGHTS), map_location=device)
    model.eval()
    names = model.module.names if hasattr(model, "module") else model.names
    if names[0] != "person" or names[1] != "head":
        raise RuntimeError(f"Expected class 0=person and 1=head; model reports {names}")
    stride = int(model.stride.max())
    first = cv2.imread(str(expected[0]))
    if first is None:
        raise RuntimeError(f"Could not read {expected[0]}")
    height, width = first.shape[:2]

    OUTPUT_ROOT.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(OUTPUT_ROOT)))
    metadata = {
        "purpose": "Diagnostic head assistance; not official MOTChallenge evaluation",
        "body_evaluation": "Use body_tracks.csv only. Never convert head_only rows to body boxes.",
        "identity_caveat": "All head associations are heuristic, not verified identities.",
        "source": str(FRAMES_DIR), "weights": str(WEIGHTS), "frames": NUM_FRAMES,
        "tracker": TRACKER_SETTINGS, "confidence": CONF_THRES, "nms_iou": IOU_THRES,
        "min_head_confidence": MIN_HEAD_CONF, "min_score": MIN_SCORE,
        "ambiguity_margin": AMBIGUITY_MARGIN, "confirm_body_pairs": CONFIRM_BODY_PAIRS,
        "max_head_only_frames": MAX_HEAD_ONLY_FRAMES,
        "supervision_version": getattr(sv, "__version__", "unknown"),
        "torch_version": str(torch.__version__),
    }
    (run_dir / "README.json").write_text(json.dumps(metadata, indent=2))
    video = cv2.VideoWriter(str(run_dir / "head_assisted.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    state = {}
    try:
        if not video.isOpened():
            raise RuntimeError("OpenCV could not open the mp4v video encoder.")
        with (run_dir / "observations.csv").open("x", newline="") as obs_file, \
                (run_dir / "body_tracks.csv").open("x", newline="") as body_file:
            observations = csv.writer(obs_file)
            baseline = csv.writer(body_file)
            observations.writerow(OBS_COLUMNS)
            baseline.writerow(BODY_COLUMNS)
            for frame_no, path in enumerate(expected, 1):
                frame = cv2.imread(str(path))
                if frame is None or frame.shape[:2] != (height, width):
                    raise RuntimeError(f"Unreadable or inconsistent frame: {path}")
                img = letterbox(frame, IMG_SIZE, stride=stride)[0]
                img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
                tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0
                with torch.no_grad():
                    prediction = model(tensor, augment=False)[0]
                    detections = non_max_suppression(
                        prediction, conf_thres=CONF_THRES, iou_thres=IOU_THRES,
                        classes=[0, 1], agnostic=False,
                    )[0]
                if len(detections):
                    detections[:, :4] = scale_coords(
                        tensor.shape[2:], detections[:, :4], frame.shape
                    ).round()
                data = detections.detach().cpu().numpy()
                people = data[data[:, 5] == 0]
                heads = data[(data[:, 5] == 1) & (data[:, 4] >= MIN_HEAD_CONF)]
                person_detections = sv.Detections(
                    xyxy=people[:, :4].copy(), confidence=people[:, 4].copy(),
                    class_id=np.zeros(len(people), dtype=int),
                ) if len(people) else sv.Detections.empty()
                # Heads never enter ByteTrack and never alter its identities.
                tracked = tracker.update_with_detections(person_detections)
                bodies = {
                    int(tid): (box.copy(), float(conf))
                    for box, conf, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id)
                    if tid is not None
                }
                # No extrapolation through missed head frames; expire immediately.
                state = {tid: s for tid, s in state.items() if s["last_head"] == frame_no - 1}
                auxiliary = [tid for tid, s in state.items() if tid not in bodies
                             and s["pairs"] >= CONFIRM_BODY_PAIRS
                             and frame_no - s["last_body"] <= MAX_HEAD_ONLY_FRAMES]
                candidates = list(bodies) + auxiliary
                scores = np.full((len(candidates), len(heads)), -1.0)
                for row, tid in enumerate(candidates):
                    old = state.get(tid)
                    for col, head in enumerate(heads):
                        temporal = temporal_score(old["head"], head[:4]) if old else None
                        if tid in bodies:
                            spatial = spatial_score(bodies[tid][0], head[:4])
                            if spatial is None or (old is not None and temporal is None):
                                continue
                            scores[row, col] = spatial if old is None else 0.5 * (spatial + temporal)
                        elif temporal is not None:
                            # A head compatible with ANY detected body cannot revive a missing ID.
                            if any(spatial_score(person[:4], head[:4]) is not None for person in people):
                                continue
                            scores[row, col] = temporal
                matches = associate(scores)
                next_state = {}
                for row, tid in enumerate(candidates):
                    body = bodies.get(tid)
                    match = matches.get(row)
                    if body is None and match is None:
                        continue
                    old = state.get(tid)
                    body_box = body[0].tolist() if body else [""] * 4
                    head_box, head_conf = [""] * 4, ""
                    score, row_margin, col_margin = "", "", ""
                    pairs, temporal_ok = 0, False
                    gap = 0 if body else frame_no - old["last_body"]
                    observation_type = "body"
                    if match is not None:
                        col, score, row_margin, col_margin = match
                        head_box, head_conf = heads[col, :4].tolist(), float(heads[col, 4])
                        temporal_ok = old is not None
                        pairs = ((old["pairs"] if old and old["last_body"] == frame_no - 1 else 0) + 1
                                 if body else old["pairs"])
                        next_state[tid] = dict(
                            head=heads[col, :4].copy(), last_head=frame_no,
                            last_body=frame_no if body else old["last_body"], pairs=pairs,
                        )
                        observation_type = "body+head" if body else "head_only"
                        draw(frame, heads[col, :4], HEAD_COLOR if body else HEAD_ONLY_COLOR,
                             f"ID {tid} head" if body else f"ID {tid} HEAD ONLY (aux)")
                    if body:
                        x1, y1, x2, y2 = body_box
                        baseline.writerow([frame_no, tid, *body_box, (x1+x2)/2, (y1+y2)/2,
                                           x2-x1, y2-y1, body[1]])
                        draw(frame, body[0], BODY_COLOR, f"ID {tid}")
                    observations.writerow([
                        frame_no, tid, *body_box, *head_box, body[1] if body else "",
                        head_conf, observation_type, score, row_margin, col_margin,
                        pairs, gap, int(temporal_ok), 0, int(body is not None),
                    ])
                state = next_state
                cv2.putText(frame, f"Frame {frame_no}/429 | green: body | cyan: head | orange: auxiliary",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                video.write(frame)
                if frame_no % 50 == 0:
                    print(f"Processed {frame_no}/{NUM_FRAMES}", flush=True)
    finally:
        video.release()
    print(f"Saved experiment to {run_dir}")
    print("Body evaluation: body_tracks.csv only; observations.csv includes auxiliary head-only rows.")


if __name__ == "__main__":
    main()
