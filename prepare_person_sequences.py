"""Extract native-resolution RGB crops and strict temporal windows (no inference).

Usage from Senior_project:
  ../.venv/bin/python prepare_person_sequences.py --check-only
  ../.venv/bin/python prepare_person_sequences.py

Frames without detections produce no crops. Missing sampled observations remain
missing and cause window rejection; no duplication or interpolation is used.
The default gap limit is one source-frame interval: any missing observation
splits that ID into a new segment. --max-gap-seconds can relax this, but cannot
fill missing sample slots. ID switches without observable gaps cannot be detected.
Padding is a fraction of body width/height on EACH side, clamped to the image;
no synthetic pixels or resizing are applied. Truncation is geometric, not an
estimate of occlusion or anatomical visibility. All crop paths are run-relative.
SequenceBuffer accepts per-frame observation dictionaries, independently of CSV,
so a live producer can reuse it with the same metadata and output callbacks.
"""

import argparse
import configparser
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import tempfile
import time

from PIL import Image

PROJECT = Path(__file__).resolve().parent
TRACKING_PRESETS = {
    'MOT20-01': 'confidence_experiments/tracking_conf_0.25.csv',
    'MOT20-02': 'validation_mot20_02/tracking_A.csv',
    'MOT20-03': 'validation_mot20_additional/MOT20-03/tracking.csv',
    'MOT20-05': 'validation_mot20_additional/MOT20-05/tracking.csv',
}
METADATA = [
    'frame', 'timestamp', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence',
    'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2', 'crop_width', 'crop_height',
    'crop_path', 'body_truncated', 'padding_clipped', 'visible_box_fraction',
    'truncated_sides', 'visibility_note',
]
SEQUENCES = ['sequence_id', 'track_id', 'segment_id', 'start_frame', 'end_frame',
             'start_timestamp', 'end_timestamp', 'crop_count', 'manifest_path']
REJECTIONS = ['track_id', 'segment_id', 'start_frame', 'end_frame', 'reason',
              'expected_count', 'available_count', 'missing_frames']


def nearest(value):
    """Round nonnegative rational values to nearest integer, ties upward."""
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


class SequenceBuffer:
    """Consume EVERY source frame in order, including empty observation maps.

    emit(track_id, segment_id, slots) receives complete windows; reject receives
    structured rejection records. Slots are (original_frame_id, observation|None).
    Memory is bounded by window length per currently active identity.
    """

    def __init__(self, fps, target_fps, length, stride, max_gap, emit, reject):
        self.fps, self.target = Fraction(str(fps)), Fraction(str(target_fps))
        self.length, self.stride, self.max_gap = length, stride, max_gap
        self.emit, self.reject = emit, reject
        self.active, self.generations = {}, {}
        self.last_frame, self.sample_index = 0, 0

    def report(self, tid, state, reason, slots):
        self.reject(dict(track_id=tid, segment_id=state['segment'],
                         start_frame=slots[0][0] if slots else state['last'],
                         end_frame=slots[-1][0] if slots else state['last'],
                         reason=reason, expected_count=self.length,
                         available_count=sum(obs is not None for _, obs in slots),
                         missing_frames=json.dumps([f for f, obs in slots if obs is None])))

    def close(self, tid, reason):
        state = self.active.pop(tid)
        if state['slots']:
            self.report(tid, state, 'incomplete_tail:' + reason, state['slots'])
        if reason == 'tracking_interruption':
            self.report(tid, state, reason, [])

    def push(self, frame, observations):
        if frame != self.last_frame + 1:
            raise ValueError('Feed every original frame consecutively, including empty frames')
        self.last_frame = frame
        for tid in list(self.active):
            if (frame - self.active[tid]['last']) / float(self.fps) > self.max_gap + 1e-9:
                self.close(tid, 'tracking_interruption')
        for tid, obs in observations.items():
            if obs['track_id'] != tid or obs['frame'] != frame:
                raise ValueError('Observation identity/frame mismatch')
            if tid not in self.active:
                generation = self.generations.get(tid, 0) + 1
                self.generations[tid] = generation
                self.active[tid] = dict(segment=generation, slots=[], last=frame)
            self.active[tid]['last'] = frame
        sample_frame = 1 + nearest(self.sample_index * self.fps / self.target)
        if frame != sample_frame:
            return
        self.sample_index += 1
        for tid, state in self.active.items():
            state['slots'].append((frame, observations.get(tid)))
            if len(state['slots']) == self.length:
                slots = list(state['slots'])
                if all(obs is not None for _, obs in slots):
                    self.emit(tid, state['segment'], slots)
                else:
                    self.report(tid, state, 'missing_sampled_observation', slots)
                del state['slots'][:self.stride]

    def finish(self):
        for tid in list(self.active):
            self.close(tid, 'end_of_source')


def crop_observation(image, row, padding, destination, run_dir, fps):
    x1, y1, x2, y2 = (row[k] for k in ('x1', 'y1', 'x2', 'y2'))
    width, height = image.size
    bw, bh = x2 - x1, y2 - y1
    visible = max(0, min(x2, width) - max(x1, 0)) * max(0, min(y2, height) - max(y1, 0))
    if bw <= 0 or bh <= 0 or visible <= 0:
        raise ValueError('invalid_or_fully_outside_body_box')
    requested = [math.floor(x1-padding*bw), math.floor(y1-padding*bh),
                 math.ceil(x2+padding*bw), math.ceil(y2+padding*bh)]
    clipped = [max(0, requested[0]), max(0, requested[1]),
               min(width, requested[2]), min(height, requested[3])]
    sides = [side for side, flag in zip(('left', 'top', 'right', 'bottom'),
                                        (x1 < 0, y1 < 0, x2 > width, y2 > height)) if flag]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        image.crop(tuple(clipped)).save(handle, format='PNG')
    return dict(row, timestamp=(row['frame']-1)/fps,
                crop_x1=clipped[0], crop_y1=clipped[1], crop_x2=clipped[2], crop_y2=clipped[3],
                crop_width=clipped[2]-clipped[0], crop_height=clipped[3]-clipped[1],
                crop_path=destination.relative_to(run_dir).as_posix(),
                body_truncated=bool(sides), padding_clipped=requested != clipped,
                visible_box_fraction=visible/(bw*bh), truncated_sides='|'.join(sides),
                visibility_note='geometric_only; occlusion_unknown')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-name', '--dataset-name', dest='sequence_name',
                        help='Dataset label/preset, default MOT20-01 or sequence directory name')
    parser.add_argument('--sequence-dir', type=Path, help='Sequence folder containing seqinfo.ini and img1')
    parser.add_argument('--image-dir', type=Path, help='Override original JPEG directory')
    parser.add_argument('--seqinfo', type=Path, help='Override seqinfo.ini path')
    parser.add_argument('--tracking-csv', type=Path, help='Override the dataset tracking CSV preset')
    parser.add_argument('--output-root', '--output-dir', dest='output_root', type=Path,
                        default=PROJECT / 'module1_sequences', help='Runs are saved under ROOT/SEQUENCE_NAME/run_*')
    parser.add_argument('--padding', type=float, default=0.10)
    parser.add_argument('--target-fps', type=float, default=10)
    parser.add_argument('--sequence-length', type=int, default=30)
    parser.add_argument('--stride', type=int, default=None, help='Sample slots; default equals sequence length')
    parser.add_argument('--max-gap-seconds', type=float, default=None,
                        help='Maximum consecutive observation interval; default 1/source FPS')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.sequence_name = args.sequence_name or (args.sequence_dir.name if args.sequence_dir else 'MOT20-01')
    if Path(args.sequence_name).name != args.sequence_name or args.sequence_name in ('.', '..'):
        parser.error('Sequence name must be a single directory name')
    args.sequence_dir = args.sequence_dir or (Path.home() /
        'Desktop/senior-project-storage/raw/mot20/train' / args.sequence_name)
    args.image_dir = args.image_dir or args.sequence_dir / 'img1'
    args.seqinfo = args.seqinfo or args.sequence_dir / 'seqinfo.ini'
    if args.tracking_csv is None:
        if args.sequence_name not in TRACKING_PRESETS:
            parser.error('Provide --tracking-csv for a dataset without a preset')
        args.tracking_csv = PROJECT / TRACKING_PRESETS[args.sequence_name]
    return args


def preflight(args):
    if Path(sys.prefix).resolve() != (PROJECT.parent / '.venv').resolve():
        raise ValueError('Run with ../.venv/bin/python')
    args.sequence_dir = args.sequence_dir.expanduser().resolve()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.seqinfo = args.seqinfo.expanduser().resolve()
    args.tracking_csv = args.tracking_csv.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    cfg = configparser.ConfigParser()
    with args.seqinfo.open() as handle:
        cfg.read_file(handle)
    fps, count = cfg.getfloat('Sequence', 'frameRate'), cfg.getint('Sequence', 'seqLength')
    args.stride = args.sequence_length if args.stride is None else args.stride
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError('Invalid source FPS or sequence length')
    args.max_gap_seconds = 1/fps if args.max_gap_seconds is None else args.max_gap_seconds
    if not (math.isfinite(args.target_fps) and 0 < args.target_fps <= fps):
        raise ValueError('Target FPS must be positive and at most source FPS (no duplication)')
    if not (math.isfinite(args.padding) and args.padding >= 0 and
            math.isfinite(args.max_gap_seconds) and args.max_gap_seconds >= 1/fps - 1e-9):
        raise ValueError('Padding must be nonnegative; gap limit must be at least one source interval')
    if args.sequence_length < 1 or not 1 <= args.stride <= args.sequence_length:
        raise ValueError('Require length >= 1 and 1 <= stride <= length')
    image_paths = list(args.image_dir.glob('*.jpg'))
    frames = {int(p.stem): p for p in image_paths}
    if len(frames) != len(image_paths) or sorted(frames) != list(range(1, count+1)):
        raise ValueError('Original JPEG frame IDs must match the complete seqinfo.ini sequence')
    records, seen = {}, set()
    with args.tracking_csv.open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Tracking CSV lacks required columns')
        for line, raw in enumerate(reader, 2):
            row = {k: float(raw[k]) for k in required}
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError(f'Nonfinite tracking value at CSV line {line}')
            if not row['frame'].is_integer() or not row['track_id'].is_integer():
                raise ValueError(f'Noninteger frame/track ID at line {line}')
            row['frame'], row['track_id'] = int(row['frame']), int(row['track_id'])
            key = row['frame'], row['track_id']
            if key in seen or row['frame'] not in frames:
                raise ValueError(f'Duplicate observation or absent source frame at line {line}')
            seen.add(key)
            records.setdefault(row['frame'], []).append(row)
    return fps, frames, records


def main():
    args = parse_args()
    fps, frames, records = preflight(args)
    print(f'{args.sequence_name} preflight: {len(frames)} source frames, {fps:g} FPS, {sum(map(len, records.values()))} observations')
    if args.check_only:
        print('Inputs and dependencies verified; no crops or sequences generated.')
        return
    start = time.perf_counter()
    parent = args.output_root / args.sequence_name
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='run_', dir=parent))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(source_fps=fps, sampling='global time grid; nearest source frame, ties upward',
                  timestamp='(original_frame_id - 1) / source_fps', padding_policy='fraction per side; clamp, no resize',
                  gap_policy='split on observation interval > max_gap_seconds; never fill missing sample slots',
                  paths='relative to run directory', image_format='lossless PNG, RGB',
                  visibility='geometric truncation only; does not estimate occlusion',
                  identity_limit='undetectable ID switches/reuse without a gap are not corrected')
    with (run_dir / 'config.json').open('x') as handle:
        json.dump(config, handle, indent=2)
    counts = dict(extracted_crops=0, valid_sequences=0, rejected_windows=0,
                  tracking_interruptions=0, invalid_crops=0, rejection_records=0)
    tracks = set()
    with (run_dir / 'tracking_metadata.csv').open('x', newline='') as mf, \
            (run_dir / 'sequences.csv').open('x', newline='') as sf, \
            (run_dir / 'rejected_windows.csv').open('x', newline='') as rf:
        metadata = csv.DictWriter(mf, fieldnames=METADATA)
        sequences = csv.DictWriter(sf, fieldnames=SEQUENCES)
        rejections = csv.DictWriter(rf, fieldnames=REJECTIONS)
        for writer in (metadata, sequences, rejections):
            writer.writeheader()

        def reject(record):
            rejections.writerow(record)
            counts['rejection_records'] += 1
            reason = record['reason']
            if reason.startswith('incomplete_tail:') or reason == 'missing_sampled_observation':
                counts['rejected_windows'] += 1
            elif reason == 'tracking_interruption':
                counts['tracking_interruptions'] += 1
            else:
                counts['invalid_crops'] += 1

        def emit(tid, segment, slots):
            observations = [obs for _, obs in slots]
            ids = [o['frame'] for o in observations]
            timestamps = [o['timestamp'] for o in observations]
            if (len(observations) != args.sequence_length or len(set(ids)) != len(ids)
                    or ids != sorted(ids) or any(o['track_id'] != tid for o in observations)
                    or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
                raise ValueError('Sequence identity, length or chronological validation failed')
            for obs in observations:
                with Image.open(run_dir / obs['crop_path']) as image:
                    if image.mode != 'RGB' or min(image.size) <= 0:
                        raise ValueError('Invalid crop mode/dimensions')
                    image.verify()
            sequence_id = f'track_{tid}_segment_{segment}_{ids[0]:06d}'
            manifest = Path('manifests') / (sequence_id + '.json')
            (run_dir / manifest).parent.mkdir(exist_ok=True)
            with (run_dir / manifest).open('x') as handle:
                json.dump(dict(sequence_id=sequence_id, track_id=tid, segment_id=segment,
                               paths_relative_to='run_directory', observations=observations), handle, indent=2)
            sequences.writerow(dict(sequence_id=sequence_id, track_id=tid, segment_id=segment,
                                    start_frame=ids[0], end_frame=ids[-1], start_timestamp=timestamps[0],
                                    end_timestamp=timestamps[-1], crop_count=len(ids), manifest_path=manifest.as_posix()))
            counts['valid_sequences'] += 1

        buffer = SequenceBuffer(fps, args.target_fps, args.sequence_length, args.stride,
                                args.max_gap_seconds, emit, reject)
        for frame, path in sorted(frames.items()):
            observations = {}
            rows = records.get(frame, [])
            if rows:
                try:
                    with Image.open(path) as source:
                        image = source.convert('RGB')
                except OSError as exc:
                    raise RuntimeError(f'Cannot read original frame {path}; run is incomplete') from exc
                with image:
                    for row in rows:
                        tid = row['track_id']
                        destination = run_dir / 'crops' / f'track_{tid}' / f'{frame:06d}.png'
                        try:
                            obs = crop_observation(image, row, args.padding, destination, run_dir, fps)
                        except ValueError as exc:
                            reject(dict(track_id=tid, segment_id='', start_frame=frame, end_frame=frame,
                                        reason=str(exc), expected_count=1, available_count=0, missing_frames=json.dumps([frame])))
                            continue
                        metadata.writerow(obs)
                        observations[tid] = obs
                        counts['extracted_crops'] += 1
                        tracks.add(tid)
            buffer.push(frame, observations)
        buffer.finish()
    counts.update(sequence_name=args.sequence_name, unique_tracks=len(tracks), sequence_length=args.sequence_length,
                  processing_seconds=time.perf_counter()-start)
    with (run_dir / 'summary.json').open('x') as handle:
        json.dump(counts, handle, indent=2)
    print(json.dumps(counts, indent=2))
    print(f'Outputs: {run_dir}')


if __name__ == '__main__':
    main()
