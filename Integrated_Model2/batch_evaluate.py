"""Provisional sequence-level evaluation of PNG folders using the existing pipeline."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

import cv2
import numpy as np
from ultralytics import YOLO
from integrated_model2 import IntegratedModel2, INFERENCE_MODES
from test_video import annotate_frame, extract_tracks

CATEGORIES = ('full_body', 'upper_body', 'head_shoulders', 'uncertain')


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(path))]


def discover(root):
    groups = []
    for label in ('fall', 'non_fall'):
        base = root / label
        if not base.is_dir():
            raise ValueError(f'Missing dataset class directory: {base}')
        # A sequence is a directory with PNGs, or an empty leaf (reported as error).
        dirs = [base] + sorted((p for p in base.rglob('*') if p.is_dir()), key=natural_key)
        group = []
        for folder in dirs:
            children = list(folder.iterdir())
            frames = sorted((p for p in children if p.is_file() and p.suffix.lower() == '.png'), key=natural_key)
            if frames or (folder != base and not any(p.is_dir() for p in children)):
                group.append((label, folder, frames))
        groups.append(group)
    # Interleave classes so --limit 2 exercises a fall and a non-fall sequence.
    return [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]


def reset_tracking(detector):
    predictor = getattr(detector, 'predictor', None)
    if predictor is not None:
        for tracker in getattr(predictor, 'trackers', []):
            tracker.reset()
        if hasattr(predictor, 'vid_path'):
            predictor.vid_path = [None] * len(predictor.vid_path)


def evaluate_sequence(item, root, fps, detector, model, video_dir=None, onsets=None, trace_dir=None):
    label, folder, frames = item
    sequence_id = folder.relative_to(root).as_posix()
    camera = re.search(r'(?:^|[-_/])(cam[01])(?:[-_/]|$)', sequence_id)
    row = dict(sequence_name=folder.name, sequence_id=sequence_id,
               camera=camera.group(1) if camera else 'unknown', ground_truth=label,
               number_of_frames=len(frames), processed_frames=0, assumed_fps=fps,
               max_fall_probability=None, fall_events=0, predicted_class=None,
               first_fall_timestamp=None, ground_truth_onset=None, detection_latency=None,
               visibility_distribution={c: 0 for c in CATEGORIES}, dominant_visibility='no_tracks',
               diagnostics={}, frames_without_tracks=0, track_id_transfers=0,
               processing_fps=None, processing_time=0, error=None, annotated_video=None)
    row["model_results"] = {}
    writer = None
    trace = None
    started = time.perf_counter()
    diagnostics = Counter()
    try:
        reset_tracking(detector)
        model.reset(camera_id=row['camera'])
        if not frames:
            raise ValueError('Sequence contains no PNG frames')
        if trace_dir is not None:
            trace = (trace_dir / (sequence_id.replace('/', '__') + '.jsonl')).open('w')
        shape = None
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Unreadable PNG: {path.name}')
            if shape is None:
                shape = frame.shape
            elif frame.shape != shape:
                raise ValueError(f'Inconsistent image dimensions: {path.name}')
            tracks = extract_tracks(detector, frame)
            out = model.process(frame, index / fps, tracks)
            if trace is not None:
                trace.write(json.dumps({'frame': index, 'time': index / fps,
                    'tracks': tracks, 'output': out}) + '\n')
            for event in out.get('tracking_events', []):
                diagnostics[event['type']] += 1
                if event['type'] == 'handover_candidate' and not event['accepted']:
                    diagnostics['handover_rejected'] += 1
            row['frames_without_tracks'] += not bool(tracks)
            row['track_id_transfers'] += len(out.get('track_id_changes', {}))
            for state in out['tracks'].values():
                row['visibility_distribution'][state['visibility']] += 1
                for name, pred in state.get('predictions', {}).items():
                    result = row['model_results'].setdefault(name, dict(
                        max_fall_probability=None, fall_events=0, first_fall_timestamp=None,
                        predicted_class=None, prediction_frames=0))
                    probability = pred['fall_probability']
                    if probability is not None:
                        result['prediction_frames'] += 1
                        previous = result['max_fall_probability']
                        result['max_fall_probability'] = probability if previous is None else max(previous, probability)
                    # Souad retains event-based evaluation. Raneem uses original argmax.
                    if name != 'souad' and pred['status'] == 'ready' and pred['decision'] == 'fall':
                        result['fall_events'] += 1
                        if result['first_fall_timestamp'] is None:
                            result['first_fall_timestamp'] = index / fps
                p = state['fall_probability']
                if p is not None:
                    previous = row['max_fall_probability']
                    row['max_fall_probability'] = p if previous is None else max(previous, p)
                d = state['diagnostics']
                diagnostics[d['status']] += 1
                if abs(d.get('confidence_change') or 0) >= .15:
                    diagnostics['large_confidence_change'] += 1
                if (d.get('pose_gap_seconds') or 0) > .2 + 1e-6:
                    diagnostics['pose_interruption'] += 1
                diagnostics['missing_joint_observations'] += len(d.get('missing_joint_indices', []))
                if d.get('confident_joints', 17) < 10:
                    diagnostics['low_confidence_joints'] += 1
            for event in out['events']:
                row['fall_events'] += 1
                if row['first_fall_timestamp'] is None:
                    row['first_fall_timestamp'] = event['time']
            if video_dir is not None:
                if writer is None:
                    destination = video_dir / (sequence_id.replace('/', '__') + '.mp4')
                    row['annotated_video'] = str(destination)
                    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (shape[1], shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot create video: {destination}')
                annotate_frame(frame, tracks, out, model.detector.settings['joint_ok'])
                writer.write(frame)
            row['processed_frames'] += 1
        row['predicted_class'] = 'fall' if row['fall_events'] else 'non_fall'
        for name, result in row['model_results'].items():
            if name == 'souad':
                result['fall_events'] = row['fall_events']
                result['first_fall_timestamp'] = row['first_fall_timestamp']
            result['predicted_class'] = ('fall' if result['fall_events'] else 'non_fall') if result['prediction_frames'] else None
        onset = (onsets or {}).get(sequence_id)
        if onset is not None and label == 'fall':
            row['ground_truth_onset'] = onset
            if row['first_fall_timestamp'] is not None:
                row['detection_latency'] = row['first_fall_timestamp'] - onset
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
        row['predicted_class'] = None  # Partial sequences must not count as TN/FN.
    finally:
        if trace is not None:
            trace.close()
        if writer is not None:
            writer.release()
    row['processing_time'] = time.perf_counter() - started
    row['processing_fps'] = row['processed_frames'] / row['processing_time']
    row['diagnostics'] = dict(diagnostics)
    counts = row['visibility_distribution']
    if sum(counts.values()):
        winners = [c for c, n in counts.items() if n == max(counts.values())]
        row['dominant_visibility'] = winners[0] if len(winners) == 1 else 'mixed'
    return row


def divide(n, d):
    return n / d if d else None


def metrics(rows):
    valid = [r for r in rows if r['error'] is None and r['predicted_class'] is not None]
    tp = sum(r['ground_truth'] == 'fall' and r['predicted_class'] == 'fall' for r in valid)
    fn = sum(r['ground_truth'] == 'fall' and r['predicted_class'] == 'non_fall' for r in valid)
    fp = sum(r['ground_truth'] == 'non_fall' and r['predicted_class'] == 'fall' for r in valid)
    tn = sum(r['ground_truth'] == 'non_fall' and r['predicted_class'] == 'non_fall' for r in valid)
    speeds = [r['processing_fps'] for r in valid]
    alerts = sum(r['fall_events'] for r in valid if r['ground_truth'] == 'non_fall')
    return dict(evaluated=len(valid), errors=sum(r['error'] is not None for r in rows),
                excluded=len(rows)-len(valid),
                abstained=sum(r['error'] is None and r['predicted_class'] is None for r in rows), TP=tp, FP=fp, TN=tn, FN=fn,
                precision=divide(tp, tp+fp), recall=divide(tp, tp+fn),
                f1=divide(2*tp, 2*tp+fp+fn), specificity=divide(tn, tn+fp),
                false_positive_rate=divide(fp, fp+tn), false_alerts=alerts,
                false_alerts_per_non_fall_video=divide(alerts, fp+tn),
                average_processing_fps=statistics.mean(speeds) if speeds else None,
                median_processing_fps=statistics.median(speeds) if speeds else None)


def write_outputs(rows, output, metadata):
    with (output / 'batch_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()})
    summary = dict(metadata, completed_sequences=len(rows), overall=metrics(rows),
                   by_camera={c: metrics([r for r in rows if r['camera'] == c])
                              for c in sorted({'cam0', 'cam1'} | {r['camera'] for r in rows})},
                   by_dominant_visibility={c: metrics([r for r in rows if r['dominant_visibility'] == c])
                                           for c in sorted({r['dominant_visibility'] for r in rows})},
                   visibility_distribution=dict(sum((Counter(r['visibility_distribution']) for r in rows), Counter())),
                   failure_cases=[r for r in rows if r['error'] or r['predicted_class'] != r['ground_truth']])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    m = summary['overall']
    canvas = np.full((370, 650, 3), 255, np.uint8)
    def text(s, x, y, scale=.65):
        cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 1, cv2.LINE_AA)
    text('PROVISIONAL - sequence confusion matrix', 20, 30)
    text('Predicted fall', 230, 85); text('Predicted non-fall', 420, 85)
    for y, truth, entries in [(120, 'Actual fall', [('TP', m['TP']), ('FN', m['FN'])]),
                               (220, 'Actual non-fall', [('FP', m['FP']), ('TN', m['TN'])])]:
        text(truth, 15, y+45, .55)
        for x, (name, n) in zip((220, 420), entries):
            cv2.rectangle(canvas, (x, y), (x+185, y+80), (220, 240, 220) if name in ('TP', 'TN') else (220, 220, 250), -1)
            text(f'{name}: {n}', x+35, y+45, .8)
    text(f"Evaluated: {m['evaluated']}   Excluded: {m['excluded']}", 20, 350)
    if not cv2.imwrite(str(output / 'confusion_matrix.png'), canvas):
        raise RuntimeError('Could not save confusion matrix PNG')
    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}' if isinstance(v, float) else str(v)
    lines = ['# Provisional batch evaluation', '',
             f"Completed {len(rows)} / {metadata['selected_sequences']} selected sequences. Assumed source FPS: {metadata['assumed_fps']}.", '',
             'Souad predictions use original alert events; Raneem predictions use original argmax on ready windows. No fusion. Raneem event counts are positive windows.',
             'Model preprocessing is unchanged. Souad resamples 30 frames at 15 FPS; Raneem retains its 30-observation buffer, stride 10 and 15 FPS gap padding. Source timing must be considered when comparing windows.', '',
             '## Provenance and interpretation', '', *metadata['provenance_notes'], '',
             'Errors and sequences with no usable predictions are excluded from classification metrics and counted separately; partial results remain in the CSV. Undefined ratios are N/A/null.',
             'Visibility counts are person-frame observations, not independent samples. Grouping uses each sequence’s dominant category (ties: mixed; no detections: no_tracks). These groups are descriptive, not causal visibility-performance estimates.',
             'Camera views of the same action are correlated. cam1 has no non-fall examples in the current dataset; its specificity and false-positive rate are undefined.',
             'Processing FPS includes image reading, tracking, pose, classifier, and optional rendering/encoding. Model loading is excluded; first inference warm-up is included. These are observed wall-clock rates, not an isolated hardware benchmark; other machine activity can affect them.',
             'Timestamps use sorted frame position / assumed FPS. Numbering gaps are not reconstructed. Latency is only recorded with supplied onset annotations; negative latency means the first alert preceded the annotated onset.', '',
             '## Overall metrics', '', '| Metric | Value |', '|---|---|']
    lines += [f'| {k} | {fmt(v)} |' for k, v in m.items()]
    for title, groups in [('Camera comparison', summary['by_camera']), ('Dominant visibility comparison', summary['by_dominant_visibility'])]:
        lines += ['', f'## {title}', '', '| Group | N | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | FPR | False alerts / non-fall |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for name, values in groups.items():
            lines.append('| ' + ' | '.join([name] + [fmt(values[k]) for k in ('evaluated','TP','FP','TN','FN','precision','recall','f1','specificity','false_positive_rate','false_alerts_per_non_fall_video')]) + ' |')
    lines += ['', '## Failure cases', '', '| Sequence | Truth | Prediction | Max P | Events | Error |', '|---|---|---|---|---|---|']
    for r in summary['failure_cases']:
        lines.append('| ' + ' | '.join(str(r[k] if r[k] is not None else 'N/A').replace('|', '/').replace('\n', ' ') for k in ('sequence_id','ground_truth','predicted_class','max_fall_probability','fall_events','error')) + ' |')
    if not summary['failure_cases']:
        lines.append('| None | | | | | |')
    lines += ['', 'Inspect per-sequence diagnostics for missing skeletons, insufficient history, person confirmation, temporal gaps and tracking transfers. A ready classifier can still return a probability below threshold.', '']
    (output / 'performance_report.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('../test'))
    parser.add_argument('--output', type=Path, default=Path('evaluation'))
    parser.add_argument('--fps', type=float, default=15, help='Provisional source FPS; does not change model sampling FPS')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--save-videos', action='store_true')
    parser.add_argument('--trace', action='store_true', help='Save per-frame poses, quality and tracking decisions as JSONL')
    parser.add_argument('--onsets', type=Path, help='JSON mapping class/sequence folder to verified onset seconds')
    parser.add_argument('--mode', choices=INFERENCE_MODES, default='souad')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('--fps and --limit must be positive (FPS must be finite)')
    try:
        items = discover(args.dataset)
        onsets = json.loads(args.onsets.read_text()) if args.onsets else {}
        if not isinstance(onsets, dict) or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in onsets.values()):
            raise ValueError('Onsets must map sequence IDs to finite nonnegative seconds')
        unknown = set(onsets) - {folder.relative_to(args.dataset).as_posix() for _, folder, _ in items}
        if unknown:
            raise ValueError(f'Unknown onset sequence IDs: {sorted(unknown)}')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not items:
        parser.error('No sequence directories found')
    discovered = len(items)
    items = items[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    video_dir = args.output / 'annotated_videos' if args.save_videos else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output / 'traces' if args.trace else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
    notes = [
        'PROVISIONAL: original recording FPS is unverified; --fps is an assumption, not proof of source timing.',
        'Held-out status is unverified. model2_fall_detection/README.md describes 47 train/validation videos and 10 reserved cam0 test videos, but provides no sequence IDs or split manifest. Its referenced training directory/notebook is absent from this project.',
        'No dataset metadata or onset annotations were found locally. Directory names alone do not prove exclusion from training, including alternate-camera views of training actions.',
        'Required before final reporting: recover the checkpoint training split manifest, compare action identities across cameras, and verify source FPS and fall onset annotations.',
        'Latency omitted without supplied ground-truth onset timestamps.' if not onsets else f'Onset source: {args.onsets}; latency uses the assumed source FPS.',
    ]
    notes.append('Souad uses its original alert events; Raneem uses class argmax on ready windows. Raneem fall_events counts positive windows, not alerts. No fusion is performed.')
    model = IntegratedModel2(use_fast=False, mode=args.mode, checkpoint_dir=args.checkpoint_dir)
    if model.detector.settings['threshold'] != .70:
        raise RuntimeError('Expected existing threshold 0.70; refusing to change checkpoint settings')
    detector = YOLO('yolov8n.pt')
    checkpoint = Path('model2_fall_detection/models/stgcnpp_fall_v1_laptop.pt')
    metadata = dict(inference_mode=args.mode, status='provisional', dataset=str(args.dataset.resolve()), assumed_fps=args.fps,
                    source_fps_verified=False, held_out_verified=False, discovered_sequences=discovered,
                    selected_sequences=len(items), model_settings=model.detector.settings,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    raneem_checkpoint_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in args.checkpoint_dir.glob('*.pth')} if args.mode != 'souad' else {},
                    provenance_notes=notes, latency_onset_source=str(args.onsets) if args.onsets else None)
    print(f'PROVISIONAL: {len(items)}/{discovered} sequences; assumed source FPS={args.fps}', flush=True)
    rows = []
    for i, item in enumerate(items, 1):
        print(f'[{i}/{len(items)}] {item[1].name}: {len(item[2])} frames', flush=True)
        row = evaluate_sequence(item, args.dataset, args.fps, detector, model, video_dir, onsets, trace_dir)
        rows.append(row)
        # Keep separate reports: no cross-classifier sequence decision.
        names = ['souad', 'raneem_visibility', 'raneem_fullbody'] if args.mode == 'parallel' else [args.mode]
        for name in names:
            destination = args.output / name
            destination.mkdir(exist_ok=True)
            model_rows = []
            for original in rows:
                selected = dict(original)
                selected.pop('model_results')
                selected.update(original['model_results'].get(name, dict(
                    predicted_class=None, max_fall_probability=None, fall_events=0,
                    first_fall_timestamp=None, prediction_frames=0)))
                selected['detection_latency'] = (selected['first_fall_timestamp'] - selected['ground_truth_onset']
                    if selected['first_fall_timestamp'] is not None and selected['ground_truth_onset'] is not None else None)
                model_rows.append(selected)
            write_outputs(model_rows, destination, dict(metadata, classifier=name))
        (args.output / 'independent_results.json').write_text(json.dumps(rows, indent=2))
        if args.mode == 'souad':
            write_outputs(rows, args.output, metadata)
        print(f"  prediction={row['predicted_class']} events={row['fall_events']} max_P={row['max_fall_probability']} FPS={row['processing_fps']:.2f} error={row['error']}", flush=True)
    if args.mode == 'souad':
        print(json.dumps(metrics(rows), indent=2), flush=True)
    else:
        print('Independent classifier metrics are in each classifier subdirectory.', flush=True)
    print(f'Reports: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
