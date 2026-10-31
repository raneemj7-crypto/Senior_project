"""Live Model 1 (YOLOv5m + ByteTrack) -> Raneem Model 2 bridge.

Run `python live_camera.py --help` for options.
Requires your actual Model 1 weights and the fall_module.py/fall_core.py package.
"""
from __future__ import annotations
import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from raneem_adapter import RaneemModel2

import cv2
import numpy as np


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', default='0', help='Webcam index (0,1...) or RTSP URL')
    p.add_argument('--model1-weights', required=True, type=Path, help='YOLOv5m CrowdHuman .pt')
    p.add_argument('--yolov5-repo', required=True, type=Path, help='Local YOLOv5 repository root')
    p.add_argument('--model2-root', required=True, type=Path, help='Raneem_Work directory')
    p.add_argument('--threshold', type=float, default=0.70)
    p.add_argument('--device', default=None, help='optional device override not yet wired to Model 1')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--confidence', type=float, default=0.25)
    p.add_argument('--record', type=Path, default=None, help='Optional annotated MP4 output')
    p.add_argument('--events', type=Path, default=Path('collapse_events.jsonl'))
    p.add_argument('--width', type=int, default=0)
    p.add_argument('--height', type=int, default=0)
    return p.parse_args()


def load_model1(cfg):
    try:
        import torch
        import supervision as sv
    except ImportError as e:
        raise RuntimeError('Install torch, supervision, opencv-python and numpy') from e
    repo = cfg.yolov5_repo.resolve()
    weights = cfg.model1_weights.resolve()
    if not (repo / 'hubconf.py').exists():
        raise FileNotFoundError(f'Not a YOLOv5 repo (missing hubconf.py): {repo}')
    if not weights.exists():
        raise FileNotFoundError(f'Model 1 weights missing: {weights}')
    detector = torch.hub.load(str(repo), 'custom', path=str(weights), source='local', trust_repo=True)
    detector.conf = cfg.confidence
    detector.iou = 0.45
    detector.classes = [0]  # Frozen Model 1 config: person class 0
    tracker = sv.ByteTrack(frame_rate=30, lost_track_buffer=60, track_activation_threshold=0.25, minimum_matching_threshold=0.8)
    return detector, tracker


def load_model2(cfg):
    return RaneemModel2(cfg.model2_root, threshold=cfg.threshold)


def extract_track_info(detections):
    """Output bridge: Model 1 -> [{track_id, box, confidence}, ...]."""
    tracks = []
    for xyxy, conf, tid in zip(detections.xyxy, detections.confidence, detections.tracker_id):
        if tid is None or int(tid) < 0:
            continue
        tracks.append({'track_id': int(tid), 'box': [float(v) for v in xyxy], 'confidence': float(conf)})
    return tracks


def normalize_model2_output(output):
    """Accept documented {tracks: ..., events: ...}, with defensive defaults."""
    if not isinstance(output, dict):
        raise TypeError(f'Model 2 process() should return a dict, got {type(output).__name__}')
    return output.get('tracks', []), output.get('events', [])


def to_track_map(items):
    mapped = {}
    if isinstance(items, dict):
        items = list(items.values())
    for obj in items or []:
        if isinstance(obj, dict):
            ident = obj.get('track_id')
            if ident is not None:
                mapped[int(ident)] = obj
    return mapped


# COCO 17-keypoint skeleton edges, using YOLOv8-Pose keypoint ordering.
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(frame, keypoints, conf_threshold=0.35):
    if keypoints is None:
        return
    joints = np.asarray(keypoints, dtype=float)
    if joints.shape != (17, 3):
        return
    visible = np.isfinite(joints).all(axis=1) & (joints[:, 2] >= conf_threshold)
    h, w = frame.shape[:2]
    def point(i):
        return (int(round(joints[i, 0])), int(round(joints[i, 1])))
    def in_frame(i):
        return 0 <= joints[i, 0] < w and 0 <= joints[i, 1] < h
    for a, b in SKELETON_EDGES:
        if visible[a] and visible[b] and in_frame(a) and in_frame(b):
            cv2.line(frame, point(a), point(b), (0, 255, 255), 2, cv2.LINE_AA)
    for i in range(17):
        if visible[i] and in_frame(i):
            cv2.circle(frame, point(i), 4, (255, 0, 255), -1, cv2.LINE_AA)


def draw(frame, tracks, scores):
    for tr in tracks:
        tid = tr['track_id']
        x1, y1, x2, y2 = map(int, tr['box'])
        info = scores.get(tid, {})
        draw_skeleton(frame, info.get('keypoints'))
        prob = info.get('fall_probability', info.get('fall_prob', info.get('confidence')))
        flagged = bool(info.get('fall_detected', False))
        color = (0, 0, 255) if flagged else (0, 210, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'ID {tid}' + (f' | fall {float(prob):.2f}' if isinstance(prob, (int, float)) else '')
        if flagged:
            label += ' | ALERT'
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    return frame


def main():
    cfg = args_parser()
    detector, tracker = load_model1(cfg)
    model2 = load_model2(cfg)
    source = int(cfg.camera) if cfg.camera.isdigit() else cfg.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera: {cfg.camera}; try --camera 1 or check permissions/RTSP')
    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    writer = None
    frame_no = 0
    start = time.monotonic()
    cfg.events.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cfg.events.open('a', encoding='utf-8') as events_file:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print('No camera frame; stopping.')
                    break
                t = time.monotonic() - start  # increasing timestamp for Model 2
                results = detector(frame, size=cfg.imgsz)
                raw = results.xyxy[0].detach().cpu().numpy()
                import supervision as sv
                if len(raw):
                    xyxy = raw[:, :4]
                    conf = raw[:, 4]
                    cls = raw[:, 5].astype(int)
                    # YOLOv5 CrowdHuman class mapping may include person/head.
                    # Choose person by class *name*, not blindly class index 0.
                    names = detector.names
                    person_mask = np.array([int(c) == 0 for c in cls])
                    xyxy, conf, cls = xyxy[person_mask], conf[person_mask], cls[person_mask]
                else:
                    xyxy = np.empty((0, 4), np.float32)
                    conf = np.empty((0,), np.float32)
                    cls = np.empty((0,), int)
                det = sv.Detections(xyxy=np.asarray(xyxy, dtype=np.float32),
                                    confidence=np.asarray(conf, dtype=np.float32),
                                    class_id=np.asarray(cls, dtype=int))
                tracked = tracker.update_with_detections(det)
                tracks = extract_track_info(tracked)
                # Model 2 expected interface, from prior fall_module.py documentation
                output = model2.process(frame, t, tracks)
                model2_tracks, new_events = normalize_model2_output(output)
                score_map = to_track_map(model2_tracks)
                view = draw(frame.copy(), tracks, score_map)
                for event in new_events or []:
                    payload = dict(event) if isinstance(event, dict) else {'event': str(event)}
                    payload.setdefault('elapsed_seconds', round(t, 3))
                    events_file.write(json.dumps(payload, ensure_ascii=False, default=str) + '\n')
                    events_file.flush()
                    print('COLLAPSE EVENT:', payload)
                cv2.putText(view, f'Frame {frame_no} | tracks {len(tracks)} | {t:.1f}s',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                if cfg.record and writer is None:
                    cfg.record.parent.mkdir(parents=True, exist_ok=True)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps < 1 or fps > 120:
                        fps = 20
                    writer = cv2.VideoWriter(str(cfg.record), cv2.VideoWriter_fourcc(*'mp4v'),
                                             float(fps), (view.shape[1], view.shape[0]))
                if writer:
                    writer.write(view)
                cv2.imshow('Model 1 + Raneem Model 2 - Press q to exit', view)
                frame_no += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
