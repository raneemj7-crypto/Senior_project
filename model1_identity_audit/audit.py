"""Identity evidence for existing RGB windows; no inference or historical writes."""
import argparse
from collections import Counter
import configparser
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from model1_cross_validation.evaluation.mot20_protocol import (
    GT_COLUMNS, ious, preprocess, load_predictions, validate_rows,
)
from validate_person_sequences import safe_path, replay, sample_grid

RUNS = {
    'MOT20-01': 'module1_sequences/MOT20-01/run_nb3r3kz2',
    'MOT20-02': 'module1_sequences/MOT20-02/run_kr6ki56d',
    'MOT20-03': 'module1_sequences/MOT20-03/run_7dy_fhf1',
    'MOT20-05': 'module1_sequences/MOT20-05/run_lemmoul4',
}
CONSISTENT = 'Consistent matched identity'
CONFLICT = 'Observed identity conflict'
INCONCLUSIVE = 'Inconclusive'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def classify(ids):
    if len(ids) != 30:
        raise ValueError('Expected exactly 30 observations')
    present = {i for i in ids if i is not None}
    if len(present) > 1:
        return CONFLICT
    if len(present) == 1 and all(i is not None for i in ids):
        return CONSISTENT
    return INCONCLUSIVE


def match_frame(gt, predictions):
    """Full-frame, stateless Hungarian maximum-IoU matching after existing preprocessing.

    Stable ID ordering fixes input order for ties. No temporal ID carry-over.
    All predictions compete, including those not belonging to exported windows.
    """
    gt = gt.sort_values('id', kind='stable')
    predictions = predictions.sort_values('id', kind='stable')
    eligible, kept, _ = preprocess(gt, predictions)
    kept_ids = set(kept.id.astype(int))
    result = {int(tid): dict(gt_id=None, iou=None, gt_visibility=None,
                            status='unmatched' if int(tid) in kept_ids else 'distractor_removed',
                            eligible_candidates=0)
              for tid in predictions.id}
    scores = ious(eligible[['x', 'y', 'w', 'h']], kept[['x', 'y', 'w', 'h']])
    if scores.size:
        valid = scores >= .5
        for j, tid in enumerate(kept.id):
            result[int(tid)]['eligible_candidates'] = int(valid[:, j].sum())
        rows, cols = linear_sum_assignment(-np.where(valid, scores, 0.0))
        for i, j in zip(rows, cols):
            if valid[i, j]:
                g = eligible.iloc[i]
                result[int(kept.iloc[j].id)].update(
                    gt_id=int(g.id), iou=float(scores[i, j]),
                    gt_visibility=float(g.visibility), status='matched')
    return result


def historical_metrics(name, pred_hash, gt_hash):
    found = []
    protocol_hash = digest(PROJECT/'model1_cross_validation/evaluation/mot20_protocol.py')
    for path in sorted((PROJECT/'model1_cross_validation/evaluation/results').glob('run_*/metadata.json')):
        meta = json.loads(path.read_text())
        if meta.get('prediction_origin') != 'as-stored' or meta.get('protocol_sha256') != protocol_hash:
            continue
        for item in meta.get('inputs', []):
            if (item['sequence'], item['prediction_sha256'], item['gt_sha256']) == (name, pred_hash, gt_hash):
                for row in read_csv(path.parent/'metrics.csv'):
                    if row['sequence'] == name:
                        found.append(dict(metadata=str(path), metadata_sha256=digest(path),
                                          metrics_sha256=digest(path.parent/'metrics.csv'),
                                          idf1=float(row['idf1']), mota=float(row['mota']),
                                          id_switches=int(float(row['num_switches']))))
    if not found:
        raise ValueError('No historical diagnostic with identical prediction/GT/protocol hashes')
    return found


def inspect_run(name, root, metric_lookup=historical_metrics, scope=None):
    """Fail closed on any provenance or window inconsistency before identity scoring."""
    config = json.loads((root/'config.json').read_text())
    source = Path(config['sequence_dir'])
    ini_path = Path(config.get('seqinfo', source/'seqinfo.ini'))
    ini = configparser.ConfigParser()
    with ini_path.open() as f:
        ini.read_file(f)
    total, fps = ini.getint('Sequence', 'seqLength'), ini.getfloat('Sequence', 'frameRate')
    if ini.get('Sequence', 'name') != name or config['sequence_length'] != 30 or config['source_fps'] != fps:
        raise ValueError('Source/config identity mismatch')
    prediction_path = Path(config['tracking_csv'])
    gt_path = source/'gt/gt.txt'
    hashes = {str(p): digest(p) for p in [prediction_path, gt_path, ini_path,
              root/'config.json', root/'tracking_metadata.csv', root/'sequences.csv', root/'summary.json']}
    historical = metric_lookup(name, hashes[str(prediction_path)], hashes[str(gt_path)])
    raw = pd.read_csv(prediction_path, float_precision='round_trip').set_index(['track_id', 'frame'])
    pred = load_predictions(prediction_path, total, offset=0)
    gt = pd.read_csv(gt_path, header=None, names=GT_COLUMNS)
    validate_rows(gt, total, gt=True)
    metadata = {}
    for row in read_csv(root/'tracking_metadata.csv'):
        key = int(row['track_id']), int(row['frame'])
        if key in metadata or key not in raw.index:
            raise ValueError('Duplicate metadata or absent prediction')
        baseline = raw.loc[key]
        if any(float(row[k]) != float(baseline[k]) for k in ('x1','y1','x2','y2','confidence')):
            raise ValueError('Metadata box differs from configured prediction')
        metadata[key] = row
    # Selected archival runs have zero invalid crops: insist on complete row identity.
    if set(metadata) != set(raw.index):
        raise ValueError('Prediction/metadata inventory differs; manual provenance review required')
    expected = replay(metadata, total, fps, config['max_gap_seconds'],
                      sample_grid(total, fps, config['target_fps']), 30, config['stride'])[1]
    windows, observed, ids = [], Counter(), set()
    for index in read_csv(root/'sequences.csv'):
        path = safe_path(root, index['manifest_path'])
        hashes[str(path)] = digest(path)
        payload = json.loads(path.read_text())
        tid, segment = int(index['track_id']), int(index['segment_id'])
        sid = index['sequence_id']
        if sid in ids or (payload['sequence_id'], payload['track_id'], payload['segment_id']) != (sid, tid, segment):
            raise ValueError('Manifest/index identity mismatch')
        ids.add(sid)
        entries = payload['observations']
        if len(entries) != 30 or int(index['crop_count']) != 30:
            raise ValueError('Non-30-frame window')
        frames = []
        for o in entries:
            key = int(o['track_id']), int(o['frame'])
            if key[0] != tid or key not in metadata or any(str(o.get(k)) != v for k, v in metadata[key].items()):
                raise ValueError('Manifest does not exactly match crop metadata')
            if not np.isclose(float(o['timestamp']), (key[1]-1)/fps, rtol=0, atol=1e-8):
                raise ValueError('Wrong timestamp')
            if not safe_path(root, o['crop_path']).is_file():
                raise ValueError('Missing crop')
            frames.append(key[1])
        if (int(index['start_frame']), int(index['end_frame'])) != (frames[0], frames[-1]):
            raise ValueError('Index frame extent mismatch')
        observed[tid, segment, tuple(frames)] += 1
        windows.append((index, entries))
    if observed != expected:
        raise ValueError('Manifests do not reproduce sequence-generation policy')
    saved = json.loads((root/'summary.json').read_text())
    if saved['valid_sequences'] != len(windows) or saved['extracted_crops'] != len(metadata):
        raise ValueError('Saved summary inventory mismatch')
    return config, pred, gt, windows, dict(
        source_run=str(root), prediction=str(prediction_path), ground_truth=str(gt_path),
        file_sha256=hashes, verified_prediction_rows=len(metadata), verified_windows=len(windows),
        historical_metrics=historical,
        scope=scope or 'Archival buffer30 predictions; not frozen buffer60 CV/test predictions.',
        provenance_limit='Exact current row correspondence and historical evaluation hashes verified; '
                         'generation-time checkpoint/environment hashes were not recorded in these crop configs.')


def contact_sheet(path, root, config, index, observations):
    sheet = Image.new('RGB', (6*180, 5*260+48), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), f"{index['sequence_id']} | Track {index['track_id']}", fill='black')
    draw.text((8, 23), 'GT = diagnostic box association; not a visual identity guarantee', fill='black')
    image_dir = Path(config.get('image_dir', str(Path(config['sequence_dir'])/'img1')))
    for i, o in enumerate(observations):
        crop_path = safe_path(root, o['crop_path'])
        with Image.open(crop_path) as im:
            crop = im.convert('RGB')
        # Independently verify exported examples against original image pixels.
        with Image.open(image_dir/f"{o['frame']:06d}.jpg") as im:
            original = im.convert('RGB')
        box = tuple(int(o[k]) for k in ('crop_x1','crop_y1','crop_x2','crop_y2'))
        expected = original.crop(box)
        if crop.size != expected.size or crop.tobytes() != expected.tobytes():
            raise ValueError('Visual audit crop pixels differ from original frame')
        thumb = ImageOps.contain(crop, (174, 202))
        x, y = (i % 6)*180, (i // 6)*260+48
        sheet.paste(thumb, (x+(180-thumb.width)//2, y))
        color = 'darkgreen' if o['status'] == 'matched' else 'darkred'
        draw.text((x+3,y+204), f"frame {o['frame']}  track {o['track_id']}", fill='black')
        draw.text((x+3,y+220), f"GT {o['gt_id']} | {o['status']}", fill=color)
        draw.text((x+3,y+236), f"IoU {o['iou']:.3f}" if o['iou'] is not None else 'IoU --', fill='black')
    with path.open('xb') as f:
        sheet.save(f, format='PNG')


def evaluate_run(name, root, out, visual_count, metric_lookup=historical_metrics, scope=None):
    config, pred, gt, windows, provenance = inspect_run(name, root, metric_lookup, scope)
    print(f'{name}: provenance verified; matching {len(windows)} windows', flush=True)
    needed = {int(o['frame']) for _, entries in windows for o in entries}
    gs = dict(tuple(gt.groupby('frame')))
    matches = {}
    for frame, predictions in pred.groupby('frame'):
        if int(frame) in needed:
            matches[int(frame)] = match_frame(gs.get(frame, gt.iloc[:0]), predictions)
    results, all_observations, lookup = [], [], {}
    for index, entries in windows:
        obs = []
        for slot, entry in enumerate(entries):
            item = dict(sequence=name, sequence_id=index['sequence_id'], sample_index=slot,
                        segment_id=int(index['segment_id']), **entry,
                        **matches[int(entry['frame'])][int(entry['track_id'])])
            obs.append(item)
        ids = [o['gt_id'] for o in obs]
        results.append(dict(sequence=name, sequence_id=index['sequence_id'], track_id=int(index['track_id']),
                            segment_id=int(index['segment_id']), start_frame=int(index['start_frame']),
                            end_frame=int(index['end_frame']), classification=classify(ids),
                            matched_observations=sum(i is not None for i in ids),
                            gt_ids=json.dumps(sorted({i for i in ids if i is not None}))))
        all_observations.extend(obs)
        lookup[index['sequence_id']] = (index, obs)
    counts = Counter(r['classification'] for r in results)
    status = Counter(o['status'] for o in all_observations)
    unique = {(o['track_id'], o['frame']): o for o in all_observations}
    n = len(results)
    summary = dict(sequence=name, windows=n, sampled_observations=len(all_observations),
                   matched_observations=status['matched'], unmatched_observations=status['unmatched'],
                   distractor_removed_observations=status['distractor_removed'],
                   match_coverage=status['matched']/len(all_observations) if all_observations else None,
                   unique_observations=len(unique), unique_matched_observations=sum(o['gt_id'] is not None for o in unique.values()),
                   consistent=counts[CONSISTENT], conflicts=counts[CONFLICT], inconclusive=counts[INCONCLUSIVE],
                   consistent_fraction=counts[CONSISTENT]/n if n else None,
                   conflict_fraction=counts[CONFLICT]/n if n else None,
                   multi_candidate_observations=sum(o['eligible_candidates'] > 1 for o in all_observations),
                   historical_idf1=provenance['historical_metrics'][0]['idf1'],
                   historical_mota=provenance['historical_metrics'][0]['mota'],
                   historical_id_switches=provenance['historical_metrics'][0]['id_switches'])
    out.mkdir()
    write_csv(out/'windows.csv', results)
    write_csv(out/'observations.csv', all_observations)
    selected = sorted((r for r in results if r['classification'] != CONSISTENT),
                      key=lambda r: (0 if r['classification'] == CONFLICT else 1,
                                     r['track_id'], r['start_frame'], r['sequence_id']))[:visual_count]
    visual_records = []
    for r in selected:
        index, obs = lookup[r['sequence_id']]
        filename = f"contact_{len(visual_records)+1:02d}.png"
        contact_sheet(out/filename, root, config, index, obs)
        visual_records.append(dict(sequence_id=r['sequence_id'], classification=r['classification'],
                                   file=filename, sha256=digest(out/filename), verified_original_pixel_crops=30))
    provenance['visual_audit'] = visual_records
    provenance['crop_pixel_verification'] = 'Only exported contact-sheet crops; all window crop paths checked for existence.'
    for path, before in provenance['file_sha256'].items():
        if digest(path) != before:
            raise ValueError('Source changed during audit: '+path)
    write_json(out/'provenance.json', provenance)
    write_json(out/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequences', nargs='+', choices=list(RUNS), default=list(RUNS))
    p.add_argument('--visual-per-sequence', type=int, default=0)
    p.add_argument('--check-only', action='store_true', help='Read-only provenance validation, no results written')
    args = p.parse_args()
    if args.visual_per_sequence < 0:
        p.error('Visual count must be nonnegative')
    if args.check_only:
        for name in args.sequences:
            *_, provenance = inspect_run(name, PROJECT/RUNS[name])
            print(name, provenance['verified_windows'], 'windows: exact provenance verified', flush=True)
        return
    output = Path(tempfile.mkdtemp(prefix='run_', dir=Path(__file__).resolve().parent))
    metadata = dict(status='running', python=sys.version, command=sys.argv,
                    versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','Pillow','motmetrics']},
                    code_sha256={str(f):digest(f) for f in [Path(__file__), PROJECT/'validate_person_sequences.py',
                                  PROJECT/'model1_cross_validation/evaluation/mot20_protocol.py']},
                    convention='as-stored coordinates; existing distractor preprocessing; marked class-1 pedestrians; '
                               'no visibility cutoff; framewise maximum-total-IoU Hungarian at IoU >= 0.5; '
                               'stable ID ordering; no temporal carry-over; all frame predictions compete.',
                    interpretation='Diagnostic GT association evidence only, not official benchmark accuracy or visual proof. '
                                   'Unmatched observations are not identity switches. Selected complete windows exclude rejected tails/gaps. '
                                   'Results do not evaluate frozen buffer60 predictions.')
    summaries, limitations = [], []
    for name in args.sequences:
        try:
            summaries.append(evaluate_run(name, PROJECT/RUNS[name], output/name, args.visual_per_sequence))
        except (OSError, ValueError, KeyError) as exc:
            limitations.append(dict(sequence=name, error=str(exc)))
            print(name, 'NOT SCORED:', exc, flush=True)
    metadata.update(status='complete' if not limitations else 'incomplete', limitations=limitations)
    write_json(output/'run_metadata.json', metadata)
    write_csv(output/'summary.csv', summaries)
    print(json.dumps(summaries, indent=2)); print('Results:', output)
    if limitations:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
