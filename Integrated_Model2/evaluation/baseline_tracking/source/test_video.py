"""Run integrated Model 2 on one local video with a YOLO+ByteTrack stand-in for Model 1.

Example:
  python test_video.py --video ../test/example.mp4 --save evaluation/example_annotated.mp4
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO
from integrated_model2 import IntegratedModel2
from model2_fall_detection.fall_core import COCO_EDGES


# Display-only warning band; never used by the classifier or event logic.
POSSIBLE_FALL_DISPLAY = 0.50


def state_color(state):
    if state.get('falling'):
        return (0, 0, 255)
    if (state.get('fall_probability') or 0) >= POSSIBLE_FALL_DISPLAY:
        return (0, 165, 255)
    return (0, 200, 0)


def draw_skeleton(frame, state, color, joint_ok):
    if state.get('keypoints') is None:
        return
    xy = np.asarray(state['keypoints'])
    kc = np.asarray(state['keypoint_conf'])
    h, w = frame.shape[:2]
    visible = (np.isfinite(xy).all(axis=1) & np.isfinite(kc) & (kc >= joint_ok)
               & (xy[:, 0] > 0) & (xy[:, 0] < w)
               & (xy[:, 1] > 0) & (xy[:, 1] < h))
    for a, b in COCO_EDGES:
        if visible[a] and visible[b]:
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 2, cv2.LINE_AA)
    for point in xy[visible]:
        cv2.circle(frame, tuple(point.astype(int)), 3, color, -1, cv2.LINE_AA)


def annotate_frame(frame, tracks, out, joint_ok, show_skeleton=True):
    """Shared renderer for video and PNG-sequence evaluation."""
    for tr in tracks:
        s = out['tracks'].get(tr['track_id'], {})
        x1, y1, x2, y2 = map(int, tr['box'])
        color = state_color(s)
        if show_skeleton:
            draw_skeleton(frame, s, color, joint_ok)
        p = s.get('fall_probability')
        label = f"ID {tr['track_id']} P:{p:.2f}" if p is not None else f"ID {tr['track_id']} P:..."
        label += f" {s.get('visibility', 'uncertain')}"
        if s.get('falling'):
            label += ' FALL ALERT'
        elif (p or 0) >= POSSIBLE_FALL_DISPLAY:
            label += ' POSSIBLE FALL'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def extract_tracks(detector, frame):
    """YOLO + ByteTrack settings shared with batch evaluation."""
    r = detector.track(frame, persist=True, tracker='bytetrack.yaml',
                       classes=[0], conf=0.10, verbose=False)[0]
    tracks = []
    if r.boxes is not None and r.boxes.id is not None:
        tracks = [
            {'track_id': int(track_id), 'box': box.tolist(), 'confidence': float(conf)}
            for track_id, box, conf in zip(r.boxes.id.int().cpu().tolist(),
                                          r.boxes.xyxy.cpu().numpy(),
                                          r.boxes.conf.cpu().tolist())
        ]
    return tracks


def main():
    parser = argparse.ArgumentParser(description='Test integrated fall detection on a video')
    parser.add_argument('--video', required=True, help='Path to source .mp4/.avi/.mov')
    parser.add_argument('--save', default=None, help='Optional annotated output .mp4')
    parser.add_argument('--events', default='evaluation/fall_events.jsonl', help='Output event log')
    parser.add_argument('--detector', default='yolov8n.pt', help='YOLO person detector weights')
    parser.add_argument('--camera-id', default='test-video')
    parser.add_argument('--show-skeleton', action=argparse.BooleanOptionalAction, default=True,
                        help='Draw the actual fall-model pose (default: enabled)')
    args = parser.parse_args()

    video = Path(args.video).expanduser()
    if not video.is_file():
        parser.error(f'Video not found: {video}')
    events_path = Path(args.events).expanduser()
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save:
        save_path = Path(args.save).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    detector = YOLO(args.detector)
    # CPU-compatible default. The integrated module may download its pose weights on first use.
    fall = IntegratedModel2(camera_id=args.camera_id, use_fast=False)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not 1 <= fps <= 240:
        fps = 30.0
    writer = None
    frames = 0
    events_count = 0
    previous_ids = set()
    last_debug = {}
    max_probability = {}
    wall_start = time.perf_counter()
    try:
        with events_path.open('w', encoding='utf-8') as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                tracks = extract_tracks(detector, frame)
                # Timestamp in *seconds*, not an integer frame index.
                out = fall.process(frame, frames / fps, tracks)
                current_ids = {tr['track_id'] for tr in tracks}
                if current_ids != previous_ids:
                    print(f'[tracking t={frames / fps:.2f}] appeared={sorted(current_ids - previous_ids)} '
                          f'lost={sorted(previous_ids - current_ids)} '
                          f'history_transfers={out.get("track_id_changes", {})}')
                previous_ids = current_ids
                for tid, state in out['tracks'].items():
                    p = state.get('fall_probability')
                    if p is not None:
                        max_probability[tid] = max(max_probability.get(tid, 0), p)
                    d = state['diagnostics']
                    status = 'ready' if d['status'] == 'prediction_cached' else d['status']
                    if d.get('confident_joints', 17) < 10:
                        status += '/low_confidence_joints'
                    if p is not None and p < fall.detector.settings['threshold']:
                        status += '/below_fall_threshold'
                    previous = last_debug.get(tid)
                    if previous is None or previous[0] != status or frames / fps - previous[1] >= 2:
                        print(f'[pose/fall t={frames / fps:.2f} ID={tid}] {status} '
                              f'P={p} diagnostics={d}')
                        last_debug[tid] = (status, frames / fps)
                frames += 1
                for event in out['events']:
                    events_count += 1
                    log.write(json.dumps(event, default=float) + '\n')
                    print('FALL EVENT:', json.dumps(event, default=float))
                if save_path:
                    annotate_frame(frame, tracks, out, fall.detector.settings['joint_ok'], args.show_skeleton)
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f'Could not create annotated video: {save_path}')
                    writer.write(frame)
                if frames % 100 == 0:
                    print(f'Processed {frames} frames...')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - wall_start
    print(f'Finished: {frames} frames, {frames / fps:.1f} s of video, {events_count} fall events')
    print(f'Processing throughput: {frames / elapsed:.2f} FPS (includes detection, pose, fall inference)')
    print(f'Max fall probability per ID: {max_probability}')
    print('Below-threshold probabilities cannot confirm a fall, even if a fall is visible to a reviewer.')
    print(f'Events: {events_path}')
    if save_path:
        print(f'Annotated video: {save_path}')
    print('Note: these are detected events, not accuracy scores; scoring needs ground-truth labels.')


if __name__ == '__main__':
    main()
