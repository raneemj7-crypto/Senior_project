"""Fixed-parameter MOT20-03/05 validation; diagnostic, not official MOT scores.

Preflight only: ../.venv/bin/python validate_mot20_sequences.py --check-only
Explicit run:  ../.venv/bin/python validate_mot20_sequences.py
No output files are created in check-only mode. Existing outputs are never replaced.
"""

import argparse
import configparser
import csv
import importlib
import inspect
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path.home() / "Desktop/senior-project-storage/raw/mot20/train"
SEQUENCES = ("MOT20-03", "MOT20-05")
WEIGHTS = PROJECT_DIR / "weights/crowdhuman_yolov5m.pt"
OUTPUT_ROOT = PROJECT_DIR / "validation_mot20_additional"
IMG_SIZE = 640
CONF_THRES = 0.25
NMS_IOU = 0.45
EVAL_IOU = 0.5
TRACKER_SETTINGS = dict(
    track_activation_threshold=0.25, lost_track_buffer=30,
    minimum_matching_threshold=0.8, minimum_consecutive_frames=1,
)
NOTICE = (
    "Diagnostic evaluation, not official MOTChallenge scores: "
    "MOT20 ignored-region handling is not fully implemented."
)
CSV_COLUMNS = [
    "frame", "track_id", "x1", "y1", "x2", "y2",
    "center_x", "center_y", "width", "height", "confidence",
]
GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "mark", "class_id", "visibility"]
METRICS = [
    "idf1", "idp", "idr", "mota", "num_switches", "num_fragmentations",
    "precision", "recall", "num_misses", "num_false_positives",
]
METRIC_LABELS = dict(zip(METRICS, [
    "IDF1", "ID precision", "ID recall", "MOTA", "ID switches",
    "Fragmentations", "Tracking precision", "Tracking recall",
    "False negatives", "False positives",
]))


def inspect_inputs():
    """Validate the complete original frame sequence; never substitute data."""
    errors, sequences = [], []
    if not WEIGHTS.is_file():
        errors.append(f"Missing model weights: {WEIGHTS}")
    for name in SEQUENCES:
        root = DATA_ROOT / name
        try:
            if not root.is_dir() or not (root / "img1").is_dir():
                raise ValueError(f"Missing sequence/image folder: {root}")
            ini = root / "seqinfo.ini"
            if not ini.is_file():
                raise ValueError(f"Missing sequence metadata: {ini}")
            cfg = configparser.ConfigParser()
            cfg.read(ini)
            fps = cfg.getint("Sequence", "frameRate")
            length = cfg.getint("Sequence", "seqLength")
            if fps <= 0 or length <= 0:
                raise ValueError("Frame rate and sequence length must be positive")
            paths = list((root / "img1").glob("*.jpg"))
            frames = sorted((int(p.stem), p) for p in paths)
            if [number for number, _ in frames] != list(range(1, length + 1)):
                raise ValueError(f"JPEG frame numbers do not cover exactly 1–{length}")
            gt_path = root / "gt/gt.txt"
            gt_exists = gt_path.is_file()
            print(f"{name}: {length} JPEGs, {fps} FPS, ground truth: {gt_exists}")
            if not gt_exists:
                print(f"WARNING: missing {gt_path}; tracking only, metrics will be marked unavailable.")
            sequences.append(dict(name=name, frames=frames, fps=fps,
                                  length=length, gt_path=gt_path, gt_exists=gt_exists))
        except (ValueError, OSError, configparser.Error) as exc:
            errors.append(f"{name}: {exc}")
    return sequences, errors


def check_dependencies():
    """Import and smoke-check APIs in the project's .venv, without model loading."""
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    venv = venv_python.parent.parent
    if not venv_python.is_file():
        raise RuntimeError(f"Project Python environment is missing: {venv_python}")
    if Path(sys.prefix).resolve() != venv.resolve():
        raise RuntimeError(f"Use the project's Python at {venv_python}; no environment substitution is allowed.")
    modules, errors = {}, []
    for name in ("cv2", "numpy", "torch", "supervision", "pandas", "motmetrics"):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("Dependency import failures:\n" + "\n".join(errors))
    sv, np, mm = modules["supervision"], modules["numpy"], modules["motmetrics"]
    missing = (set(TRACKER_SETTINGS) | {"frame_rate"}) - set(inspect.signature(sv.ByteTrack).parameters)
    if missing:
        raise RuntimeError(f"Unsupported ByteTrack constructor parameters: {sorted(missing)}")
    tracker = sv.ByteTrack(**TRACKER_SETTINGS, frame_rate=25)
    tracker.update_with_detections(sv.Detections.empty())
    # Exercise IoU, assignment and all metric dependencies without dataset evaluation.
    boxes = np.array([[0., 0., 10., 10.]])
    distances = mm.distances.iou_matrix(boxes, boxes, max_iou=1.0 - EVAL_IOU)
    acc = mm.MOTAccumulator(auto_id=False)
    acc.update([1], [1], distances, frameid=1)
    mm.metrics.create().compute(acc, metrics=METRICS)
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    modules.update(attempt_load=attempt_load, letterbox=letterbox,
                   non_max_suppression=non_max_suppression, scale_coords=scale_coords)
    print("Dependency imports, YOLOv5 helpers, ByteTrack empty-frame update and metric smoke check passed.")
    print("Checkpoint loading and full inference are deferred until an explicit validation run.")
    return modules


def evaluate_sequence(sequence, tracking_path, modules):
    pd, mm = modules["pandas"], modules["motmetrics"]
    gt = pd.read_csv(sequence["gt_path"], header=None, names=GT_COLUMNS)
    gt = gt[(gt["mark"] == 1) & (gt["class_id"] == 1)].copy()
    pred = pd.read_csv(tracking_path)
    gt_frames = {int(k): g for k, g in gt.groupby("frame")}
    pred_frames = {int(k): p for k, p in pred.groupby("frame")}
    empty_gt, empty_pred = gt.iloc[:0], pred.iloc[:0]
    acc = mm.MOTAccumulator(auto_id=False)
    for frame_number, _ in sequence["frames"]:
        g = gt_frames.get(frame_number, empty_gt)
        p = pred_frames.get(frame_number, empty_pred)
        distances = mm.distances.iou_matrix(
            g[["x", "y", "w", "h"]].to_numpy(dtype=float),
            p[["x1", "y1", "width", "height"]].to_numpy(dtype=float),
            max_iou=1.0 - EVAL_IOU,
        )
        acc.update(g["id"].astype(int).tolist(), p["track_id"].astype(int).tolist(),
                   distances, frameid=frame_number)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRICS, name=sequence["name"])
    print(NOTICE)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=METRIC_LABELS))
    return summary.iloc[0].to_dict()


def track_sequence(sequence, model, stride, device, modules):
    cv2, np, torch, sv = (modules[k] for k in ("cv2", "numpy", "torch", "supervision"))
    tracker = sv.ByteTrack(**TRACKER_SETTINGS, frame_rate=sequence["fps"])
    path = OUTPUT_ROOT / sequence["name"] / "tracking.csv"
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for processed, (frame_number, image_path) in enumerate(sequence["frames"], 1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")
            img = modules["letterbox"](frame, IMG_SIZE, stride=stride)[0]
            img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
            img = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                prediction = model(img, augment=False)[0]
                detections = modules["non_max_suppression"](
                    prediction, conf_thres=CONF_THRES, iou_thres=NMS_IOU,
                    classes=[0], agnostic=False,
                )[0]
            if len(detections):
                detections[:, :4] = modules["scale_coords"](
                    img.shape[2:], detections[:, :4], frame.shape,
                ).round()
                data = detections.detach().cpu().numpy()
                people = sv.Detections(xyxy=data[:, :4].copy(), confidence=data[:, 4].copy(),
                                       class_id=np.zeros(len(data), dtype=int))
            else:
                people = sv.Detections.empty()
            tracked = tracker.update_with_detections(people)
            for box, confidence, track_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                if track_id is None:
                    continue
                x1, y1, x2, y2 = map(float, box)
                writer.writerow([frame_number, int(track_id), x1, y1, x2, y2,
                                 (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, float(confidence)])
            if processed % 100 == 0:
                print(f"{sequence['name']}: {processed}/{sequence['length']}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check inputs/APIs; no inference or output writes")
    args = parser.parse_args()
    print(NOTICE)
    sequences, errors = inspect_inputs()
    if OUTPUT_ROOT.exists():
        errors.append(f"Output directory already exists; refusing to overwrite or reuse: {OUTPUT_ROOT}")
    try:
        modules = check_dependencies()
    except Exception as exc:
        errors.append(f"Compatibility check failed: {type(exc).__name__}: {exc}")
    if errors:
        raise SystemExit("Preflight failed:\n" + "\n".join(errors))
    if args.check_only:
        print("Preflight passed. No model inference or dataset evaluation was run.")
        return

    torch, pd = modules["torch"], modules["pandas"]
    device = torch.device("cpu")
    # Load exactly once and reuse the unchanged model for both sequences.
    model = modules["attempt_load"](str(WEIGHTS), map_location=device)
    model.eval()
    names = model.module.names if hasattr(model, "module") else model.names
    if names[0] != "person":
        raise RuntimeError(f"Expected person at class 0; got {names}")
    stride = int(model.stride.max())
    # Exclusive root creation protects every artifact, even against concurrent runs.
    OUTPUT_ROOT.mkdir(exist_ok=False)
    config = dict(
        weights=str(WEIGHTS), image_size=IMG_SIZE, confidence=CONF_THRES,
        nms_iou=NMS_IOU, person_class=0, device="cpu", tracker=TRACKER_SETTINGS,
        evaluation_iou=EVAL_IOU, ground_truth_filter="mark == 1 and class_id == 1",
        evaluation_note=NOTICE, python=sys.version,
        versions={k: getattr(modules[k], "__version__", "unknown")
                  for k in ("torch", "supervision", "numpy", "pandas", "motmetrics", "cv2")},
        sequences=[dict(name=s["name"], frame_rate=s["fps"], num_frames=s["length"],
                        source=str(DATA_ROOT / s["name"] / "img1"),
                        ground_truth=str(s["gt_path"]), ground_truth_present=s["gt_exists"])
                   for s in sequences],
    )
    with (OUTPUT_ROOT / "configuration.json").open("x") as handle:
        json.dump(config, handle, indent=2)
    rows = []
    for sequence in sequences:
        directory = OUTPUT_ROOT / sequence["name"]
        directory.mkdir(exist_ok=False)
        tracking_path = track_sequence(sequence, model, stride, device, modules)
        row = dict(sequence=sequence["name"], num_frames=sequence["length"],
                   frame_rate=sequence["fps"], evaluation_note=NOTICE,
                   evaluation_status="evaluated" if sequence["gt_exists"] else "missing_ground_truth")
        row.update({metric: None for metric in METRICS})
        if sequence["gt_exists"]:
            row.update(evaluate_sequence(sequence, tracking_path, modules))
        pd.DataFrame([row]).to_csv(directory / "metrics.csv", index=False, mode="x")
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "metrics_comparison.csv", index=False, mode="x")
    print(f"Saved validation outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
