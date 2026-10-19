"""
Themam - example of how Model 1 and Model 2 connect.

A STAND-IN for Model 1 (YOLOv8n person detector + ByteTrack) produces track IDs and boxes;
Model 2 (fall_module.FallDetectionModule) turns them into fall probabilities and fall events.
Replace the stand-in with the real Model 1 output - only the `tracks` list has to look the same.

    python integration_example.py --video some_video.mp4
    python integration_example.py --video some_video.mp4 --save checked.mp4

Fall events are printed and written to fall_events.jsonl (one JSON object per line).
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

try:                                  # works both as a package (from the repo root) and as plain files
    from .fall_module import FallDetectionModule
except ImportError:
    from fall_module import FallDetectionModule

HERE = Path(__file__).resolve().parent


def model1_stand_in(detector, frame, tracker_file):
    """What Model 1 should hand over for each frame: a list of tracked people."""
    r = detector.track(frame, persist=True, tracker=str(tracker_file), classes=[0], conf=0.10, verbose=False)[0]
    if r.boxes is None or r.boxes.id is None:
        return []
    return [{"track_id": int(i), "box": b.tolist(), "confidence": float(c)}
            for i, b, c in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.cpu().numpy(), r.boxes.conf.tolist())]


def main():
    ap = argparse.ArgumentParser(description="Model 1 -> Model 2 integration example")
    ap.add_argument("--video", required=True)
    ap.add_argument("--save", default=None, help="save a video with boxes and fall probabilities")
    ap.add_argument("--detector", default="yolov8n.pt", help="Model 1 stand-in detector weights")
    ap.add_argument("--camera-id", default="cam0")
    args = ap.parse_args()

    tracker_file = HERE / "themam_bytetrack.yaml"
    tracker_file.write_text("tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.10\n"
                            "new_track_thresh: 0.25\ntrack_buffer: 40\nmatch_thresh: 0.80\nfuse_score: True\n")
    detector = YOLO(args.detector)
    fall = FallDetectionModule(camera_id=args.camera_id)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, n, events = None, 0, []
    with open(HERE / "fall_events.jsonl", "w") as log:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / fps                                       # live camera: use the capture time instead
            n += 1
            tracks = model1_stand_in(detector, frame, tracker_file)          # <- Model 1
            out = fall.process(frame, t, tracks)                             # <- Model 2
            for e in out["events"]:
                print("FALL EVENT:", json.dumps(e))
                log.write(json.dumps(e) + "\n")
                events.append(e)
            if args.save:
                for tr in tracks:
                    info = out["tracks"].get(tr["track_id"])
                    if info is None:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in tr["box"])
                    color = (0, 0, 255) if info["falling"] else (0, 200, 0)
                    p = info["fall_probability"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {tr['track_id']} " + ("..." if p is None else f"fall {p:.2f}"),
                                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
    print(f"\n{n} frames, {len(events)} fall events (saved in fall_events.jsonl)")


if __name__ == "__main__":
    main()
