"""
Themam - live fall detection on a laptop (YOLOv8n-pose + ByteTrack + ST-GCN++).

Needs, in this folder:
    live_fall_demo.py, fall_core.py
    models/stgcnpp_fall_v1_laptop.pt and models/stgcnpp_fall_v1_laptop.json
(the two model files may also sit next to this file)

Run with the webcam:        python live_fall_demo.py
Run on a recorded video:    python live_fall_demo.py --video my_fall.mp4
Also save the result video: python live_fall_demo.py --save result.mp4
Press Q in the video window to stop.
A fall alarm stays on screen (even if the person leaves the picture) until you press C.

Speed: if "openvino" is installed (pip install openvino), the pose model is
converted once (about 1 minute) and then runs about 3x faster on the CPU.
Use --normal to switch this off.

Tips: camera about 1 m high, 3-4 m away, whole body and floor visible, good light.
Fall ONLY onto a mattress / thick cushions, with someone nearby.
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch

try:                                  # works both as a package (from the repo root) and as plain files
    from .fall_core import COCO_EDGES, HERE, LiveFallDetector, load_fall_model, load_pose
except ImportError:
    from fall_core import COCO_EDGES, HERE, LiveFallDetector, load_fall_model, load_pose

AUTO_CLEAR_S = 10.0         # with --no-window (no keyboard): an alarm clears 10 s after its last fall signal
                            # (with a window, an alarm stays until you press C - like a real alert)

TRACKER_YAML = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 40
match_thresh: 0.80
fuse_score: True
"""


def draw(frame, ids, boxes, kxy, kconf, results, latched, fps, joint_ok, mode):
    visible = set()
    for tid, box, xy, kc in zip(ids, boxes, kxy, kconf):
        tid = int(tid)
        if tid not in results:
            continue                               # ignored (shadow, clothes, ...): not drawn
        visible.add(tid)
        prob, alarm = results[tid]
        # red = falling now | orange = fell earlier, alarm not cleared yet | green = OK
        color = (0, 0, 255) if alarm else (0, 165, 255) if tid in latched else (0, 200, 0)
        for p, q in COCO_EDGES:
            if kc[p] >= joint_ok and kc[q] >= joint_ok:
                cv2.line(frame, (int(xy[p, 0]), int(xy[p, 1])), (int(xy[q, 0]), int(xy[q, 1])), (0, 255, 255), 2)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"ID {tid}  " + ("waiting..." if prob is None else f"fall {prob:.2f}")
        cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for tid, a in latched.items():                 # fallen person no longer visible: show where they were
        if tid not in visible and "box" in a:
            x1, y1, x2, y2 = (int(v) for v in a["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f"ID {tid}: out of view", (x1, max(70, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if latched:
        who = ", ".join(f"ID {tid} at {a['since']:.0f}s" for tid, a in latched.items())
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(frame, "FALL DETECTED", (15, 37), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"{who}  (C = clear)", (300, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{fps:.1f} FPS  {mode}  (Q = quit)", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Themam live fall detection")
    ap.add_argument("--camera", type=int, default=0, help="webcam number (0 = built-in)")
    ap.add_argument("--video", default=None, help="run on a video file instead of the webcam")
    ap.add_argument("--save", default=None, help="save the result video, e.g. result.mp4")
    ap.add_argument("--no-window", action="store_true", help="do not open a window")
    ap.add_argument("--imgsz", type=int, default=640, help="pose model image size (lower = faster, less accurate)")
    ap.add_argument("--normal", action="store_true", help="do not use the fast OpenVINO pose model")
    ap.add_argument("--models", default=None, help="folder with the model files (default: ./models)")
    args = ap.parse_args()

    settings, predict = load_fall_model(args.models)
    device = 0 if torch.cuda.is_available() else "cpu"
    tracker_file = HERE / "themam_bytetrack.yaml"
    tracker_file.write_text(TRACKER_YAML)
    pose, engine = load_pose(settings["pose_model"], args.imgsz, use_fast=not args.normal)
    if engine == "OpenVINO":
        device = "cpu"
    print(f"Pose model: {engine} on {'GPU' if device == 0 else 'CPU'}")
    print(f"Alarm threshold: {settings['threshold']} (from stgcnpp_fall_v1_laptop.json)")
    detector = LiveFallDetector(predict, settings)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # lets many webcams send 30 FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                               # always use the newest frame
        video_fps = None
    if not cap.isOpened():
        raise SystemExit("Could not open the camera/video. Try --camera 1, or close other apps using the camera.")

    writer, log, events = None, [], []
    latched = {}            # track id -> {"since", "last", "box"}: the alarm stays until C is pressed
    mode = "fast mode" if engine == "OpenVINO" or device == 0 else "SLOW mode"
    t0, frame_i, fps, last_tick, first_tick = time.time(), 0, 0.0, None, None
    print("Running... press Q in the video window to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tick = time.time()
        t = frame_i / video_fps if video_fps else tick - t0          # video: use video time
        frame_i += 1

        r = pose.track(frame, persist=True, tracker=str(tracker_file), device=device,
                       imgsz=args.imgsz, conf=settings["detector_conf"], verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None and len(r.boxes):
            ids = r.boxes.id.int().cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy()
            kconf = r.keypoints.conf.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
        else:
            ids, boxes, kxy, kconf = np.zeros(0, int), np.zeros((0, 4)), np.zeros((0, 17, 2)), np.zeros((0, 17))
            confs = np.zeros(0)

        results = detector.update(t, ids, boxes, kxy, kconf, confs)
        box_of = {int(i): b for i, b in zip(ids, boxes)}
        for new, old in detector.renamed.items():        # same person got a new ID: keep their alarm
            if old in latched:
                latched[new] = latched.pop(old)
        for tid in detector.new_falls:                   # a NEW fall (even if an older alarm is still shown)
            latched.setdefault(tid, {})["since"] = t
            events.append((round(t, 1), tid))
            print(f"  FALL DETECTED: ID {tid} at {t:.1f} s")
        for tid, (p, alarm) in results.items():
            if p is not None:
                log.append((round(t, 3), tid, round(p, 4)))
            if tid in latched:                           # (C may have cleared it during the 3-s hold)
                latched[tid]["box"] = box_of[tid]
                if alarm:
                    latched[tid]["last"] = t
        if args.no_window:
            latched = {k: a for k, a in latched.items() if t - a["last"] <= AUTO_CLEAR_S}

        if last_tick is not None:
            inst = 1.0 / max(tick - last_tick, 1e-3)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        last_tick = tick
        frame = draw(frame, ids, boxes, kxy, kconf, results, latched, fps, settings["joint_ok"], mode)
        if first_tick is None:
            first_tick = time.time()                     # real speed is measured from here (after start-up)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), video_fps or 15.0, (w, h))
            writer.write(frame)
        if not args.no_window:
            cv2.imshow("Themam fall detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                latched.clear()                          # staff checked the person: clear the alarm

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_window:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    log_file = HERE / "fall_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerows([("seconds", "track_id", "fall_probability")] + log)
    real_fps = (frame_i - 1) / max(time.time() - first_tick, 1e-6) if first_tick else 0.0
    print(f"\nStopped after {frame_i} frames. Real average speed: {real_fps:.1f} FPS ({mode}).")
    if real_fps < 12 and not args.video:
        print("Below 12 FPS, fast falls can be missed (the person is lost between frames).")
    print(f"Fall alarms: {len(events)}" + "".join(f"\n  ID {tid} at {s} s" for s, tid in events))
    print(f"Probability log saved to {log_file}")


if __name__ == "__main__":
    main()
