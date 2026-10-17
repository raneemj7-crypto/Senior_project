"""Frozen Model 1 official-test runner. Default: preflight only, no inference.

../.venv/bin/python run_model1_test.py
Add --execute explicitly to run MOT20-04. --sequence-name also accepts 06/07/08.
Uses prepare_person_sequences.py unchanged as a subprocess after tracking.
No GT is opened, and no IDF1/MOTA or other accuracy score is computed.
"""
import argparse
import configparser
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parent
TRACKING_COLUMNS = ['frame', 'timestamp', 'track_id', 'x1', 'y1', 'x2', 'y2',
                    'center_x', 'center_y', 'width', 'height', 'confidence']
PARAMETERS = dict(track_activation_threshold=.25, lost_track_buffer=60,
                  minimum_matching_threshold=.8, minimum_consecutive_frames=1)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate frozen configuration field: {key}')
        result[key] = value
    return result


def validate_frozen(config):
    """Reject missing fields or changes to this runner's approved frozen contract."""
    expected = {
        'schema_version': 1, 'configuration_name': 'buffer60', 'frozen': True,
        'coordinate_convention': 'as-stored',
        'detector': dict(name='YOLOv5m CrowdHuman', weights='weights/crowdhuman_yolov5m.pt',
                         image_size=640, confidence_threshold=.25, nms_iou_threshold=.45, person_class=0),
        'tracker': dict(name='Supervision ByteTrack', **PARAMETERS,
                        frame_rate=dict(source='seqinfo.ini', section='Sequence', key='frameRate')),
    }
    def check(actual, required, prefix=''):
        if not isinstance(actual, dict):
            raise ValueError(f'Expected object: {prefix}')
        for key, value in required.items():
            name = prefix + key
            if key not in actual:
                raise ValueError(f'Missing frozen configuration field: {name}')
            if isinstance(value, dict):
                check(actual[key], value, name + '.')
            elif actual[key] != value or (isinstance(actual[key], bool) != isinstance(value, bool)):
                raise ValueError(f'Unsupported frozen value for {name}: {actual[key]!r}; expected {value!r}')
    check(config, expected)
    required_text = [(config, 'coordinate_policy'), (config['tracker']['frame_rate'], 'policy')]
    provenance = config.get('selection_provenance')
    if not isinstance(provenance, dict):
        raise ValueError('Missing selection_provenance object')
    for field in ('cross_validation_run_id','run_directory','method','selection_score',
                  'checkpoint_hash_source','evaluation_status','limitations'):
        required_text.append((provenance, field))
    for obj, field in required_text:
        if not isinstance(obj.get(field), str) or not obj[field].strip():
            raise ValueError('Missing/invalid frozen metadata field: ' + field)
    if provenance.get('selected_in_all_four_folds') is not True or provenance.get('sequences') != [
            'MOT20-01','MOT20-02','MOT20-03','MOT20-05']:
        raise ValueError('Invalid frozen fold selection provenance')
    sha = config['detector'].get('checkpoint_sha256', '')
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Missing/invalid frozen checkpoint_sha256')


def peak_memory():
    # Darwin reports bytes; Linux reports KiB. Child peak is not concurrent total.
    scale = 1 if sys.platform == 'darwin' else 1024
    return dict(parent_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale),
                child_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*scale),
                note='Process lifetime RSS high-water marks, reported separately; not a combined simultaneous peak.')


def preflight(args):
    if Path(sys.prefix).resolve() != (PROJECT.parent/'.venv').resolve():
        raise ValueError('Run with ../.venv/bin/python')
    config = json.loads(args.config.read_text(), object_pairs_hook=no_duplicate_keys)
    validate_frozen(config)
    import cv2
    import numpy as np
    import torch
    import supervision as sv
    from PIL import Image
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    import prepare_person_sequences as prep
    for component in ('SequenceBuffer','crop_observation','main'):
        if not callable(getattr(prep, component, None)):
            raise ValueError('Incompatible sequence preparation component: '+component)
    weights = PROJECT/config['detector']['weights']
    if not weights.is_file():
        raise FileNotFoundError(f'Frozen checkpoint missing: {weights}')
    checkpoint_hash = digest(weights)
    if checkpoint_hash != config['detector']['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA-256 differs from frozen configuration; refusing to continue')
    root = args.data_root.expanduser().resolve()/args.sequence_name
    ini = root/'seqinfo.ini'
    cfg = configparser.ConfigParser()
    with ini.open() as handle:
        cfg.read_file(handle)
    rate = config['tracker']['frame_rate']
    fps = cfg.getfloat(rate['section'], rate['key'])
    count = cfg.getint('Sequence','seqLength')
    width, height = cfg.getint('Sequence','imWidth'), cfg.getint('Sequence','imHeight')
    if cfg.get('Sequence','name') != args.sequence_name or cfg.get('Sequence','imDir') != 'img1' or cfg.get('Sequence','imExt').lower() != '.jpg':
        raise ValueError('Unexpected sequence identity/image layout in seqinfo.ini')
    if not math.isfinite(fps) or fps <= 0 or min(count,width,height) <= 0:
        raise ValueError('Invalid FPS, frame count or image dimensions')
    frames = sorted((int(p.stem),p) for p in (root/'img1').glob('*.jpg'))
    if [number for number,_ in frames] != list(range(1,count+1)):
        raise ValueError('Original JPEG frame IDs must be exactly 1..seqLength, with no duplicates')
    for _, path in frames:
        with Image.open(path) as image:
            if image.size != (width,height) or image.format != 'JPEG':
                raise ValueError(f'Invalid JPEG header/dimensions: {path}')
    args.stride = args.sequence_length if args.stride is None else args.stride
    args.max_gap_seconds = 1/fps if args.max_gap_seconds is None else args.max_gap_seconds
    if not math.isfinite(args.target_fps) or not 0 < args.target_fps <= fps:
        raise ValueError('Target sampling FPS must be positive and <= source FPS')
    if args.sequence_length < 1 or not 1 <= args.stride <= args.sequence_length:
        raise ValueError('Require length >= 1 and 1 <= stride <= length')
    if not math.isfinite(args.padding) or args.padding < 0 or not math.isfinite(args.max_gap_seconds) or args.max_gap_seconds < 1/fps-1e-9:
        raise ValueError('Padding must be nonnegative and gap limit >= one source interval')
    tracker_args = {key: config['tracker'][key] for key in PARAMETERS}
    tracker_args['frame_rate'] = fps
    missing = set(tracker_args)-set(inspect.signature(sv.ByteTrack).parameters)
    if missing:
        raise ValueError('Incompatible ByteTrack API: missing '+str(sorted(missing)))
    tracker = sv.ByteTrack(**tracker_args)
    tracker.update_with_detections(sv.Detections.empty())
    # Check the checkpoint/model interface without running a forward pass.
    started = time.perf_counter()
    model = attempt_load(str(weights), map_location=torch.device('cpu'))
    model.eval()
    names = model.module.names if hasattr(model,'module') else model.names
    if names[config['detector']['person_class']] != 'person':
        raise ValueError('Frozen detector class 0 is not person')
    stride = int(model.stride.max())
    if stride <= 0:
        raise ValueError('Invalid model stride')
    components = dict(cv2=cv2, np=np, torch=torch, sv=sv, model=model,
                      letterbox=letterbox, nms=non_max_suppression, scale=scale_coords)
    metadata = dict(sequence=args.sequence_name, source=str(root), fps=fps, frame_count=count,
                    width=width, height=height, checkpoint_sha256=checkpoint_hash,
                    frozen_config_sha256=digest(args.config), seqinfo_sha256=digest(ini),
                    runner_sha256=digest(Path(__file__)), crop_generator_sha256=digest(PROJECT/'prepare_person_sequences.py'),
                    model_load_seconds=time.perf_counter()-started, device='cpu',
                    frame_inventory_sha256=hashlib.sha256(json.dumps([(n,str(p),p.stat().st_size,p.stat().st_mtime_ns) for n,p in frames]).encode()).hexdigest(),
                    versions={k:importlib.metadata.version(k) for k in ['torch','supervision','numpy','Pillow']},
                    opencv_version=cv2.__version__, tracker_arguments=tracker_args,
                    scope='Test inference/output integrity only; no local ground truth accuracy evaluation')
    return config, frames, root, components, metadata


def run_tracking(path, frames, config, components, metadata):
    c = components
    tracker = c['sv'].ByteTrack(**metadata['tracker_arguments'])
    model = c['model']; stride = int(model.stride.max()); settings = config['detector']
    compute_seconds = 0.0
    wall_start = time.perf_counter()
    rows = 0
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=TRACKING_COLUMNS); writer.writeheader()
        for number, source in frames:
            frame = c['cv2'].imread(str(source))
            if frame is None or frame.shape[:2] != (metadata['height'],metadata['width']):
                raise ValueError('Unreadable/invalid frame: '+str(source))
            begin = time.perf_counter()
            img = c['letterbox'](frame, settings['image_size'], stride=stride)[0]
            img = c['np'].ascontiguousarray(img[:,:,::-1].transpose(2,0,1))
            tensor = c['torch'].from_numpy(img).float().unsqueeze(0)/255.0
            with c['torch'].no_grad():
                prediction = model(tensor, augment=False)[0]
                det = c['nms'](prediction,conf_thres=settings['confidence_threshold'],
                               iou_thres=settings['nms_iou_threshold'],classes=[settings['person_class']],agnostic=False)[0]
            if len(det):
                det[:,:4] = c['scale'](tensor.shape[2:],det[:,:4],frame.shape).round()
                data = det.cpu().numpy()
                detections = c['sv'].Detections(xyxy=data[:,:4].copy(),confidence=data[:,4].copy(),
                                               class_id=data[:,5].astype(int))
            else:
                detections = c['sv'].Detections.empty()
            tracked = tracker.update_with_detections(detections)
            compute_seconds += time.perf_counter()-begin
            seen = set()
            for box, confidence, tid in zip(tracked.xyxy,tracked.confidence,tracked.tracker_id):
                if tid is None:
                    continue
                tid = int(tid)
                x1,y1,x2,y2 = map(float,box)
                if tid in seen or not all(math.isfinite(x) for x in (*box,confidence)) or x2<=x1 or y2<=y1:
                    raise ValueError(f'Invalid tracker output at frame {number}')
                seen.add(tid)
                writer.writerow(dict(frame=number,timestamp=(number-1)/metadata['fps'],track_id=tid,
                                     x1=x1,y1=y1,x2=x2,y2=y2,center_x=(x1+x2)/2,center_y=(y1+y2)/2,
                                     width=x2-x1,height=y2-y1,confidence=float(confidence)))
                rows += 1
            if number%100==0:
                print(f'{metadata["sequence"]}: tracked {number}/{len(frames)}',flush=True)
    return dict(tracking_rows=rows, detection_tracking_compute_seconds=compute_seconds,
                detection_tracking_wall_seconds=time.perf_counter()-wall_start)


def validate_outputs(prepared, tracking, length, fps):
    """Check all crop files plus manifest/index identity, geometry and chronology."""
    from PIL import Image
    observations = {}; tracking_keys = set()
    with tracking.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = int(row['track_id']),int(row['frame'])
            if key in tracking_keys or not math.isclose(float(row['timestamp']),(key[1]-1)/fps,abs_tol=1e-8):
                raise ValueError('Invalid tracking key/timestamp')
            tracking_keys.add(key)
    with (prepared/'tracking_metadata.csv').open(newline='') as handle:
        paths = set()
        for row in csv.DictReader(handle):
            key = int(row['track_id']),int(row['frame'])
            path = (prepared/row['crop_path']).resolve()
            if not path.is_relative_to(prepared.resolve()) or path in paths or key in observations or key not in tracking_keys:
                raise ValueError('Invalid crop path or observation identity')
            paths.add(path)
            with Image.open(path) as image:
                if image.mode!='RGB' or image.size!=(int(row['crop_width']),int(row['crop_height'])) or min(image.size)<=0:
                    raise ValueError('Invalid RGB crop dimensions')
                image.verify()
            if not math.isclose(float(row['timestamp']),(key[1]-1)/fps,abs_tol=1e-8):
                raise ValueError('Invalid crop timestamp')
            observations[key] = row
    seen_sequences = set()
    with (prepared/'sequences.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            sid = row['sequence_id']
            manifest = (prepared/row['manifest_path']).resolve()
            if sid in seen_sequences or not manifest.is_relative_to(prepared.resolve()):
                raise ValueError('Invalid sequence ID/path')
            seen_sequences.add(sid)
            payload = json.loads(manifest.read_text())
            entries = payload['observations']; tid = int(row['track_id'])
            if payload['sequence_id']!=sid or int(payload['track_id'])!=tid or len(entries)!=length or int(row['crop_count'])!=length:
                raise ValueError('Invalid manifest identity/length')
            frames = [int(o['frame']) for o in entries]; times = [float(o['timestamp']) for o in entries]
            if len(set(frames))!=length or any(b<=a for a,b in zip(frames,frames[1:])) or any(b<=a for a,b in zip(times,times[1:])):
                raise ValueError('Duplicate/nonchronological sequence observations')
            for entry in entries:
                key = int(entry['track_id']),int(entry['frame'])
                if key[0]!=tid or key not in observations or any(str(entry.get(k))!=v for k,v in observations[key].items()):
                    raise ValueError('Manifest differs from crop metadata')
    return dict(status='passed',checked_crops=len(observations),checked_sequences=len(seen_sequences),
                identity_accuracy='Not verified: structural continuity is not person identity ground truth.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Explicitly start full inference and crops')
    parser.add_argument('--sequence-name',choices=['MOT20-04','MOT20-06','MOT20-07','MOT20-08'],default='MOT20-04')
    parser.add_argument('--data-root',type=Path,default=Path.home()/'Desktop/senior-project-storage/raw/mot20/test')
    parser.add_argument('--config',type=Path,default=PROJECT/'model1_cross_validation/final_model1_config.json')
    parser.add_argument('--output-root',type=Path,default=PROJECT/'model1_final_test')
    parser.add_argument('--target-fps',type=float,default=10)
    parser.add_argument('--sequence-length',type=int,default=30)
    parser.add_argument('--stride',type=int)
    parser.add_argument('--padding',type=float,default=.1)
    parser.add_argument('--max-gap-seconds',type=float)
    args = parser.parse_args()
    start = time.perf_counter()
    config,frames,source,components,metadata = preflight(args)
    metadata['preflight_seconds'] = time.perf_counter()-start
    print(f'Preflight passed: {args.sequence_name}, {len(frames)} frames, {metadata["fps"]:g} FPS. Checkpoint hash verified.')
    if not args.execute:
        print('Checkpoint loaded for compatibility, but no forward pass, test tracking, crops or evaluation ran.')
        return
    parent = args.output_root/args.sequence_name; parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run_',dir=parent))
    shutil.copyfile(args.config,run/'final_model1_config.json')
    shutil.copyfile(source/'seqinfo.ini',run/'seqinfo.ini')
    metadata['preprocessing'] = dict(target_fps=args.target_fps,sequence_length=args.sequence_length,
                                     stride=args.stride,padding=args.padding,max_gap_seconds=args.max_gap_seconds)
    save_json(run/'processing_metadata.json',metadata)
    try:
        processing_start = time.perf_counter()
        tracking = run/'tracking.csv'
        timing = run_tracking(tracking,frames,config,components,metadata)
        command = [sys.executable,str(PROJECT/'prepare_person_sequences.py'),
                   '--sequence-name',args.sequence_name,'--sequence-dir',str(source),
                   '--tracking-csv',str(tracking.resolve()),'--output-root',str((run/'prepared').resolve()),
                   '--target-fps',str(args.target_fps),'--sequence-length',str(args.sequence_length),
                   '--stride',str(args.stride),'--padding',str(args.padding),'--max-gap-seconds',str(args.max_gap_seconds)]
        crop_start = time.perf_counter()
        with (run/'crop_processing.log').open('x') as log:
            subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        timing['crop_sequence_wall_seconds'] = time.perf_counter()-crop_start
        outputs = list((run/'prepared'/args.sequence_name).glob('run_*'))
        if len(outputs)!=1:
            raise ValueError('Expected exactly one new crop run')
        validation_start = time.perf_counter()
        integrity = validate_outputs(outputs[0],tracking,args.sequence_length,metadata['fps'])
        timing['integrity_validation_seconds'] = time.perf_counter()-validation_start
        timing['processing_seconds'] = time.perf_counter()-processing_start
        timing['total_seconds_including_preflight'] = time.perf_counter()-start
        timing['end_to_end_source_fps'] = len(frames)/timing['total_seconds_including_preflight']
        timing['processing_source_fps'] = len(frames)/timing['processing_seconds']
        result = dict(status='complete',sequence=args.sequence_name,frames=len(frames),timing=timing,
                      peak_memory=peak_memory(),output_validation=integrity,
                      prepared_run=str(outputs[0].relative_to(run)),
                      crop_summary=json.loads((outputs[0]/'summary.json').read_text()),
                      accuracy_evaluation='Not performed; no official test GT is used.')
        save_json(run/'summary.json',result)
        print(json.dumps(result,indent=2)); print('Outputs:',run)
    except Exception as exc:
        save_json(run/'failure.json',dict(status='incomplete',error=str(exc),peak_memory=peak_memory()))
        raise RuntimeError(f'Run failed; partial outputs preserved at {run}: {exc}') from exc


if __name__=='__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as exc:
        raise SystemExit(f'Model 1 test runner stopped: {exc}') from exc
