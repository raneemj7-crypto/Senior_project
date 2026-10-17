"""Independent structural audit of Module 1 crops; no pose or identity inference.

Run explicitly: ../.venv/bin/python validate_person_sequences.py
Reads the source run without changes. Writes reports only to a unique validation
folder. Exact RGB comparisons cover a deterministic sample of sequences (default
10); all indexed crops receive path, image, coordinate and metadata checks.
Structural success does not establish person identity, visible joints, or YOLOv8
Pose accuracy. No generator code is imported or executed.
"""
import argparse
from collections import Counter, defaultdict
import configparser
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

from PIL import Image

PROJECT = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT / 'module1_sequences/MOT20-01/run_nb3r3kz2'


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def safe_path(root, value):
    relative = Path(value)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe run-relative path: {value}')
    return resolved


def stats(values):
    return dict(count=len(values), minimum=min(values), maximum=max(values),
                mean=statistics.mean(values), median=statistics.median(values)) if values else dict(count=0)


def sample_grid(total, fps, target):
    ratio = Fraction(str(fps)) / Fraction(str(target))
    frames, k = [], 0
    while True:
        value = k * ratio
        frame = 1 + (2 * value.numerator + value.denominator) // (2 * value.denominator)
        if frame > total:
            return frames
        frames.append(frame)
        k += 1


def replay(metadata, total, fps, gap, grid, length, stride):
    """Reconstruct segments from observation times, then slice global sample slots.

    Expiration occurs at the first processed source frame whose distance from the
    last observation exceeds gap + 1e-9, even if the identity never returns.
    """
    by_track = defaultdict(list)
    for tid, frame in metadata:
        by_track[tid].append(frame)
    segments, windows, events = {}, [], []
    for tid, frames in by_track.items():
        groups = []
        for frame in sorted(frames):
            if not groups or (frame - groups[-1][-1]) / fps > gap + 1e-9:
                groups.append([])
            groups[-1].append(frame)
        for segment, group in enumerate(groups, 1):
            last = group[-1]
            expiration = last + 1
            while expiration <= total and (expiration-last)/fps <= gap + 1e-9:
                expiration += 1
            interrupted = expiration <= total
            stop = expiration - 1 if interrupted else total
            slots = [f for f in grid if group[0] <= f <= stop]
            present = set(group)
            segments[tid, segment] = dict(frames=group, start=group[0], end=last,
                                          interrupted=interrupted)
            offset = 0
            while len(slots)-offset >= length:
                window = slots[offset:offset+length]
                missing = [f for f in window if f not in present]
                if missing:
                    events.append((tid, segment, window[0], window[-1],
                                   'missing_sampled_observation', length, length-len(missing), tuple(missing)))
                else:
                    windows.append((tid, segment, tuple(window)))
                offset += stride
            tail = slots[offset:]
            if tail:
                missing = [f for f in tail if f not in present]
                reason = 'incomplete_tail:' + ('tracking_interruption' if interrupted else 'end_of_source')
                events.append((tid, segment, tail[0], tail[-1], reason, length,
                               len(tail)-len(missing), tuple(missing)))
            if interrupted:
                events.append((tid, segment, last, last, 'tracking_interruption', length, 0, ()))
    return segments, Counter(windows), Counter(events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--sequence-name', '--dataset-name', dest='sequence_name', help='Dataset label; defaults to saved configuration')
    parser.add_argument('--sequence-dir', type=Path, help='Override original sequence folder')
    parser.add_argument('--image-dir', type=Path, help='Override original JPEG directory')
    parser.add_argument('--seqinfo', type=Path, help='Override original seqinfo.ini')
    parser.add_argument('--tracking-csv', type=Path, help='Override original tracking CSV')
    parser.add_argument('--output-root', '--output-dir', dest='output_root', type=Path, default=PROJECT / 'module1_sequences/validation')
    parser.add_argument('--sample-sequences', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (PROJECT.parent / '.venv').resolve():
        parser.error('Use ../.venv/bin/python')
    if args.sample_sequences < 1:
        parser.error('--sample-sequences must be positive')
    start = time.perf_counter()
    root = args.run_dir.expanduser().resolve()
    config = json.loads((root/'config.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    rows = read_csv(root/'sequences.csv')
    metadata_rows = read_csv(root/'tracking_metadata.csv')
    rejected = read_csv(root/'rejected_windows.csv')
    source = (args.sequence_dir or Path(config['sequence_dir'])).expanduser().resolve()
    sequence_name = args.sequence_name or config.get('sequence_name', source.name)
    image_dir = (args.image_dir or (source/'img1' if args.sequence_dir else
                 Path(config.get('image_dir', str(source/'img1'))))).expanduser().resolve()
    seqinfo = (args.seqinfo or (source/'seqinfo.ini' if args.sequence_dir else
               Path(config.get('seqinfo', str(source/'seqinfo.ini'))))).expanduser().resolve()
    tracking_csv = (args.tracking_csv or Path(config['tracking_csv'])).expanduser().resolve()
    ini = configparser.ConfigParser()
    with seqinfo.open() as handle:
        ini.read_file(handle)
    fps, total = ini.getfloat('Sequence','frameRate'), ini.getint('Sequence','seqLength')
    target, length, stride = float(config['target_fps']), int(config['sequence_length']), int(config['stride'])
    gap, padding = float(config['max_gap_seconds']), float(config['padding'])
    if not (0 < target <= fps and length > 0 and 0 < stride <= length and gap >= 0):
        raise ValueError('Invalid sampling/gap configuration')
    issues = []
    def issue(scope, reason):
        issues.append(dict(sequence_id=scope, reason=str(reason)))
    if (target, length) != (10, 30):
        issue('RUN', 'Configuration differs from proposed 10 FPS / 30 crops')
    if ini.get('Sequence', 'name', fallback=sequence_name) != sequence_name:
        issue('RUN', 'Dataset label disagrees with seqinfo.ini')
    if float(config['source_fps']) != fps:
        issue('RUN', 'Configured FPS disagrees with seqinfo.ini')
    metadata, crop_owners, source_sizes = {}, {}, {}
    bad_observations = set()
    baseline = {}
    for row in read_csv(tracking_csv):
        key = int(row['track_id']), int(row['frame'])
        if key in baseline:
            issue('RUN', f'Duplicate baseline tracking key {key}')
        baseline[key] = row
    for row in metadata_rows:
        key = int(row['track_id']), int(row['frame'])
        before = len(issues)
        try:
            if key in metadata:
                raise ValueError(f'Duplicate metadata key {key}')
            metadata[key] = row
            frame = key[1]
            if not 1 <= frame <= total:
                raise ValueError('Frame outside source timeline')
            if not math.isclose(float(row['timestamp']), (frame-1)/fps, abs_tol=1e-8):
                raise ValueError('Incorrect timestamp')
            original = baseline.get(key)
            if original is None or any(not math.isclose(float(row[k]), float(original[k]), abs_tol=1e-7)
                                       for k in ('x1','y1','x2','y2','confidence')):
                raise ValueError('Metadata does not match original tracking CSV')
            path = safe_path(root, row['crop_path'])
            if path in crop_owners and crop_owners[path] != key:
                raise ValueError('Crop path reused by different observations')
            crop_owners[path] = key
            if frame not in source_sizes:
                with Image.open(image_dir/f'{frame:06d}.jpg') as image:
                    source_sizes[frame] = image.size
            width, height = source_sizes[frame]
            x1,y1,x2,y2 = [float(row[k]) for k in ('x1','y1','x2','y2')]
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or x2 <= x1 or y2 <= y1:
                raise ValueError('Invalid source bounding box')
            bw,bh = x2-x1,y2-y1
            requested = (math.floor(x1-padding*bw),math.floor(y1-padding*bh),
                         math.ceil(x2+padding*bw),math.ceil(y2+padding*bh))
            expected = (max(0,requested[0]),max(0,requested[1]),min(width,requested[2]),min(height,requested[3]))
            actual = tuple(int(row[k]) for k in ('crop_x1','crop_y1','crop_x2','crop_y2'))
            visible = max(0,min(width,x2)-max(0,x1))*max(0,min(height,y2)-max(0,y1))
            sides = '|'.join(s for s, flag in zip(('left','top','right','bottom'),
                                                (x1<0,y1<0,x2>width,y2>height)) if flag)
            if actual != expected or visible <= 0 or not (0 <= actual[0] < actual[2] <= width and 0 <= actual[1] < actual[3] <= height):
                raise ValueError('Unexpected crop boundaries/padding')
            if (row['truncated_sides'] != sides or row['body_truncated'].lower() != str(bool(sides)).lower()
                    or row['padding_clipped'].lower() != str(requested != expected).lower()
                    or not math.isclose(float(row['visible_box_fraction']),visible/(bw*bh),abs_tol=1e-8)):
                raise ValueError('Incorrect truncation metadata')
            with Image.open(path) as image:
                if (image.mode != 'RGB' or image.size != (actual[2]-actual[0],actual[3]-actual[1])
                        or image.size != (int(row['crop_width']),int(row['crop_height']))):
                    raise ValueError('Invalid crop dimensions or RGB mode')
                image.verify()
        except (OSError, ValueError, KeyError) as exc:
            issue('OBSERVATION '+str(key), exc)
        if len(issues) > before:
            bad_observations.add(key)
    grid = sample_grid(total,fps,target)
    grid_index = {f:i for i,f in enumerate(grid)}
    segments, expected_windows, expected_events = replay(metadata,total,fps,gap,grid,length,stride)
    observed_windows = Counter()
    samples = set(random.Random(args.seed).sample(range(len(rows)), min(args.sample_sequences,len(rows))))
    seen_ids, manifests_seen, covered, sampled_coverage = set(), set(), set(), set()
    details, pixel_comparisons = [], 0
    for index, row in enumerate(rows):
        sid = row['sequence_id']
        try:
            if sid in seen_ids:
                raise ValueError('Duplicate sequence ID')
            seen_ids.add(sid)
            manifest_path = safe_path(root,row['manifest_path'])
            if manifest_path in manifests_seen:
                raise ValueError('Duplicate manifest path')
            manifests_seen.add(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            observations = manifest['observations']
            tid, segment = int(row['track_id']), int(row['segment_id'])
            if manifest['sequence_id'] != sid or int(manifest['track_id']) != tid or int(manifest['segment_id']) != segment:
                issue(sid,'Manifest/index identity mismatch')
            frames = [int(o['frame']) for o in observations]
            times = [float(o['timestamp']) for o in observations]
            if len(observations) != 30 or int(row['crop_count']) != 30:
                issue(sid,'Expected exactly 30 crops')
            if not frames:
                raise ValueError('Empty manifest')
            if len(set(frames)) != len(frames) or any(b<=a for a,b in zip(frames,frames[1:])):
                issue(sid,'Duplicate or nonchronological frame IDs')
            if any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
                issue(sid,'Nonfinite or nonchronological timestamps')
            indices = [grid_index.get(f,-1) for f in frames]
            if -1 in indices or any(b-a != 1 for a,b in zip(indices,indices[1:])):
                issue(sid,'Missing samples or deviation from rounded global sampling grid')
            span = times[-1]-times[0]
            intervals = [b-a for a,b in zip(times,times[1:])]
            if intervals and max(intervals) > math.ceil(fps/target)/fps + 1e-8:
                issue(sid,'Unusually long sampled temporal gap')
            if (int(row['start_frame']),int(row['end_frame'])) != (frames[0],frames[-1]) or not all(
                    math.isclose(float(row[k]),v,abs_tol=1e-8)
                    for k,v in [('start_timestamp',times[0]),('end_timestamp',times[-1])]):
                issue(sid,'Index range disagrees with manifest')
            signature = (tid,segment,tuple(frames))
            observed_windows[signature] += 1
            if signature not in expected_windows:
                issue(sid,'Window not reproducible within a single uninterrupted segment')
            paths = set()
            for obs in observations:
                key = tid,int(obs['frame'])
                if int(obs['track_id']) != tid:
                    issue(sid,'Mixed Track IDs')
                if key in bad_observations or key not in metadata:
                    issue(sid,f'Invalid or absent metadata/crop: {key}')
                else:
                    for k,v in metadata[key].items():
                        if str(obs.get(k)) != v:
                            issue(sid,f'Manifest/metadata mismatch: {key} {k}')
                path = safe_path(root,obs['crop_path'])
                if path in paths:
                    issue(sid,'Duplicate crop file within sequence')
                paths.add(path)
                if not path.is_file():
                    issue(sid,f'Missing crop: {obs["crop_path"]}')
                if index in samples:
                    # Reconstruct pixels from the ORIGINAL tracking row, independently.
                    original = baseline[key]
                    x1,y1,x2,y2 = [float(original[k]) for k in ('x1','y1','x2','y2')]
                    with Image.open(image_dir/f'{key[1]:06d}.jpg') as image:
                        rgb = image.convert('RGB')
                        w,h = rgb.size
                        box = (max(0,math.floor(x1-padding*(x2-x1))),max(0,math.floor(y1-padding*(y2-y1))),
                               min(w,math.ceil(x2+padding*(x2-x1))),min(h,math.ceil(y2+padding*(y2-y1))))
                        expected_crop = rgb.crop(box)
                        with Image.open(path) as saved:
                            if saved.mode != 'RGB' or saved.size != expected_crop.size or saved.tobytes() != expected_crop.tobytes():
                                issue(sid,f'Original-frame RGB pixel mismatch: {key}')
                        pixel_comparisons += 1
                sampled_coverage.add(key)
            covered.update(range(frames[0],frames[-1]+1))
            details.append(dict(sequence_id=sid,track_id=tid,segment_id=segment,crop_count=len(frames),
                                time_span_seconds=span,effective_fps=(len(frames)-1)/span if span>0 else None,
                                min_interval=min(intervals) if intervals else None,
                                max_interval=max(intervals) if intervals else None,pixel_sampled=index in samples))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            issue(sid,exc)
    if observed_windows != expected_windows:
        issue('RUN','Sequence inventory differs from independently reconstructed windows')
    actual_events = Counter()
    invalid_crop_events = 0
    for row in rejected:
        if row['reason']=='invalid_or_fully_outside_body_box':
            invalid_crop_events += 1
            continue
        try:
            actual_events[(int(row['track_id']),int(row['segment_id']),int(row['start_frame']),
                           int(row['end_frame']),row['reason'],int(row['expected_count']),
                           int(row['available_count']),tuple(json.loads(row['missing_frames'])))] += 1
        except (ValueError,KeyError,TypeError) as exc:
            issue('RUN',f'Malformed rejection record: {exc}')
    if actual_events != expected_events:
        issue('RUN','Rejection records differ from independent event reconstruction')
    causes = Counter(r['reason'] for r in rejected)
    windows_rejected = sum(v for k,v in causes.items() if k.startswith('incomplete_tail:') or k=='missing_sampled_observation')
    measured = dict(extracted_crops=len(metadata_rows),valid_sequences=len(rows),rejected_windows=windows_rejected,
                    tracking_interruptions=causes['tracking_interruption'],rejection_records=len(rejected),
                    unique_tracks=len({tid for tid,_ in metadata}),invalid_crops=invalid_crop_events)
    for k,v in measured.items():
        if summary.get(k) != v:
            issue('RUN',f'Summary count mismatch for {k}: saved={summary.get(k)}, observed={v}')
    unindexed_manifests = set((root/'manifests').glob('*.json'))-manifests_seen
    if unindexed_manifests:
        issue('RUN',f'{len(unindexed_manifests)} unindexed manifests')
    segment_counts = Counter(tid for tid,_ in segments)
    report = dict(
        sequence_name=sequence_name, image_dir=str(image_dir), seqinfo=str(seqinfo),
        tracking_csv=str(tracking_csv), source_run=str(root),structural_status='PASS' if not issues else 'REVIEW_REQUIRED',
        identity_accuracy='NOT VERIFIED: structural checks do not establish true identity or absence of ID switches.',
        module2_readiness='RGB native-resolution crop structure only; pose accuracy, joints and model input adaptation are not evaluated.',
        counts=measured, generator_reported_counts=summary, rejection_causes=dict(causes),
        rejection_definition='Incomplete sampled windows and residual tails; not necessarily independent lost-person events.',
        interruption_definition='Active segment expired when source-frame time since last crop exceeded max_gap_seconds. Includes departures and detection gaps, even without later ID reuse. Not confirmed identity switches.',
        reconstruction=dict(expected_windows=sum(expected_windows.values()),
                            missing_rejection_records=sum((expected_events-actual_events).values()),
                            unexpected_rejection_records=sum((actual_events-expected_events).values())),
        sampling=dict(source_fps=fps,target_fps=target,expected_intervals_seconds=sorted({(b-a)/fps for a,b in zip(grid,grid[1:])}),
                      explanation=f'{fps:g} to {target:g} FPS uses nearest source frames on the global timestamp grid. Intervals can alternate due to rounding. Effective rate uses (N-1)/(last-first).',
                      span_seconds=stats([r['time_span_seconds'] for r in details]),
                      effective_fps=stats([r['effective_fps'] for r in details if r['effective_fps'] is not None])),
        coverage=dict(source_frames=total,source_duration_seconds=total/fps,
                      union_sequence_frame_intervals=len(covered),timeline_fraction=len(covered)/total,
                      unique_sampled_track_observations=len(sampled_coverage),
                      fraction_of_extracted_observations=len(sampled_coverage)/len(metadata) if metadata else 0),
        continuity=dict(segments=len(segments),segments_per_track=stats(list(segment_counts.values())),
                        tracks_with_multiple_segments=sum(v>1 for v in segment_counts.values()),
                        segment_observation_counts=stats([len(s['frames']) for s in segments.values()])),
        pixel_checks=dict(sample_sequence_count=len(samples),seed=args.seed,crop_comparisons=pixel_comparisons),
        issues=issues,processing_seconds=time.perf_counter()-start,
    )
    args.output_root.mkdir(parents=True,exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run_',dir=args.output_root))
    (output/'validation_report.json').write_text(json.dumps(report,indent=2))
    with (output/'suspicious_sequences.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['sequence_id','reason'])
        writer.writeheader()
        writer.writerows(issues)
    with (output/'sequence_statistics.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['sequence_id','track_id','segment_id','crop_count','time_span_seconds',
                                              'effective_fps','min_interval','max_interval','pixel_sampled'])
        writer.writeheader()
        writer.writerows(details)
    (output/'README.txt').write_text(
        f"Structural status: {report['structural_status']}\n"
        f"Counts: {json.dumps(measured)}\nRejection causes: {json.dumps(dict(causes))}\n"
        +report['interruption_definition']+'\n'+report['identity_accuracy']+'\n'
        +report['module2_readiness']+'\nSee validation_report.json and sequence_statistics.csv for details.\n')
    print(f"{sequence_name}: {report['structural_status']}: {len(issues)} findings. Reports: {output}")


if __name__=='__main__':
    main()
