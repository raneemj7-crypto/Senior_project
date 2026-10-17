"""Visualize saved tracking only. Default: input/encoder preflight; --render runs.

../.venv/bin/python visualize_model1_tracking.py
--sequence-name MOT20-06 --run-dir /path/to/completed/run --highlight-id 12 --trail-frames 25
Odd dimensions receive one black column on the right and/or row at the bottom.
Source pixels are never resized or cropped; MP4 compression remains lossy.
"""
import argparse
import colorsys
import configparser
import csv
import hashlib
import json
from collections import deque
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT=Path(__file__).resolve().parent
REQUIRED={'frame','track_id','x1','y1','x2','y2','confidence'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify_video(path,count,fps,size):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise ValueError('Cannot open encoded video')
        declared=int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        actual_fps=cap.get(cv2.CAP_PROP_FPS)
        if declared!=count or abs(actual_fps-fps)>0.01:
            raise ValueError(f'Video metadata mismatch: frames={declared}, FPS={actual_fps}')
        decoded=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if (frame.shape[1],frame.shape[0])!=size:
                raise ValueError(f'Decoded resolution {(frame.shape[1],frame.shape[0])}, expected {size}')
            decoded+=1
        if decoded!=count: raise ValueError(f'Decoded {decoded}/{count} frames')
        return dict(decoded_frames=decoded,fps=actual_fps,width=size[0],height=size[1])
    finally: cap.release()


def pad_for_video(frame):
    """Retain every input pixel at its original coordinate before compression."""
    height,width=frame.shape[:2]
    padded=cv2.copyMakeBorder(frame,0,height%2,0,width%2,
                              cv2.BORDER_CONSTANT,value=(0,0,0))
    if not np.array_equal(padded[:height,:width],frame):
        raise ValueError('Padding altered original pixels')
    if np.any(padded[height:]) or np.any(padded[:,width:]):
        raise ValueError('Padding is not black')
    return padded


def probe_encoder(size,fps):
    encoded_size=(size[0]+size[0]%2,size[1]+size[1]%2)
    attempts=[]
    for backend in (cv2.CAP_FFMPEG,cv2.CAP_AVFOUNDATION):
        for codec in ('mp4v','avc1'):
            with tempfile.TemporaryDirectory(prefix='model1_encoder_') as temp:
                path=Path(temp)/'probe.mp4'
                writer=cv2.VideoWriter(str(path),backend,cv2.VideoWriter_fourcc(*codec),fps,encoded_size)
                try:
                    if not writer.isOpened(): raise ValueError('VideoWriter did not open')
                    for i in range(3):
                        frame=np.zeros((size[1],size[0],3),dtype=np.uint8)
                        frame[:,:]=(i*60,40,180)
                        writer.write(pad_for_video(frame))
                except Exception as exc:
                    attempts.append(dict(backend=backend,codec=codec,error=str(exc)))
                    continue
                finally: writer.release()
                try:
                    verification=verify_video(path,3,fps,encoded_size)
                    return dict(backend=backend,codec=codec,probe_verification=verification),attempts
                except ValueError as exc:
                    attempts.append(dict(backend=backend,codec=codec,error=str(exc)))
    return None,attempts


def color_map(ids):
    # Stable sorted-ID assignment; unique BGR tuples, though many IDs cannot all
    # be perceptually distinguishable. Labels remain the identity reference.
    result={}; used=set()
    for index,tid in enumerate(sorted(ids)):
        rgb=colorsys.hsv_to_rgb((index*.61803398875)%1,.65+.1*(index%3),.9)
        bgr=tuple(int(round(v*255)) for v in rgb[::-1])
        while bgr in used: bgr=((bgr[0]+1)%256,bgr[1],bgr[2])
        used.add(bgr); result[int(tid)]=bgr
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-name',choices=['MOT20-04','MOT20-06','MOT20-07','MOT20-08'],default='MOT20-04')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--data-root',type=Path,default=Path.home()/'Desktop/senior-project-storage/raw/mot20/test')
    p.add_argument('--output-root',type=Path,default=PROJECT/'model1_visualizations')
    p.add_argument('--highlight-id',type=int)
    p.add_argument('--trail-frames',type=int,default=0,help='Number of consecutive source-frame centers to retain; 0 disables trails')
    p.add_argument('--render',action='store_true')
    args=p.parse_args()
    if Path(sys.prefix).resolve()!=(PROJECT.parent/'.venv').resolve(): p.error('Use ../.venv/bin/python')
    if args.trail_frames<0: p.error('--trail-frames must be nonnegative')
    if args.run_dir is None:
        if args.sequence_name!='MOT20-04': p.error('Specify --run-dir for this sequence')
        args.run_dir=PROJECT/'model1_final_test/MOT20-04/run_551un9g_'
    run=args.run_dir.expanduser().resolve()
    summary=json.loads((run/'summary.json').read_text())
    processing=json.loads((run/'processing_metadata.json').read_text())
    if summary.get('status')!='complete' or summary.get('sequence')!=args.sequence_name or processing.get('sequence')!=args.sequence_name:
        raise ValueError('Run is incomplete or belongs to a different sequence')
    # Discover top-level CSVs by schema; deliberately exclude nested crop metadata.
    candidates=[]; schemas={}
    for path in sorted(run.glob('*.csv')):
        with path.open(newline='') as f: fields=next(csv.reader(f),[])
        schemas[path.name]=fields
        if REQUIRED<=set(fields): candidates.append(path)
    if len(candidates)!=1: raise ValueError(f'Expected one tracking CSV; found {candidates}; schemas={schemas}')
    csv_path=candidates[0]
    root=args.data_root.expanduser().resolve()/args.sequence_name
    cfg=configparser.ConfigParser()
    with (root/'seqinfo.ini').open() as f: cfg.read_file(f)
    fps=cfg.getfloat('Sequence','frameRate'); count=cfg.getint('Sequence','seqLength')
    size=(cfg.getint('Sequence','imWidth'),cfg.getint('Sequence','imHeight'))
    if cfg.get('Sequence','name')!=args.sequence_name or fps<=0 or not np.isfinite(fps) or min(*size,count)<=0:
        raise ValueError('Invalid seqinfo metadata')
    if summary['frames']!=count or processing['fps']!=fps or (processing['width'],processing['height'])!=size:
        raise ValueError('Saved run/source dimensions or timeline differ')
    frames=sorted((int(path.stem),path) for path in (root/'img1').glob('*.jpg'))
    if [n for n,_ in frames]!=list(range(1,count+1)): raise ValueError('JPEG timeline is incomplete or duplicated')
    for _,path in frames:
        with Image.open(path) as im:
            if im.size!=size: raise ValueError('Source resolution mismatch: '+str(path))
    data=pd.read_csv(csv_path)
    if not np.isfinite(data[list(REQUIRED)].to_numpy(dtype=float)).all(): raise ValueError('Nonfinite tracking values')
    if ((data.frame%1!=0)|(data.track_id%1!=0)|(~data.frame.between(1,count))|(data.track_id<0)).any(): raise ValueError('Invalid frame/track ID')
    if data.duplicated(['frame','track_id']).any(): raise ValueError('Duplicate frame/track ID')
    if ((data.x2<=data.x1)|(data.y2<=data.y1)|(~data.confidence.between(0,1))).any(): raise ValueError('Invalid bounding box/confidence')
    if 'timestamp' in data and not np.allclose(data.timestamp,(data.frame-1)/fps,atol=1e-8,rtol=0): raise ValueError('Timestamp mismatch')
    if args.highlight_id is not None and args.highlight_id not in set(data.track_id): raise ValueError('Highlighted Track ID does not exist')
    colors=color_map(data.track_id.unique())
    print(f'Inputs verified: {csv_path.name}, {len(data)} tracking rows, {count} JPEGs, {size}, {fps:g} FPS',flush=True)
    encoder,attempts=probe_encoder(size,fps)
    encoded_size=(size[0]+size[0]%2,size[1]+size[1]%2)
    # Confirm lossless padding on an actual source frame as well as the probes.
    sample=cv2.imread(str(frames[0][1]))
    if sample is None or (sample.shape[1],sample.shape[0])!=size:
        raise ValueError('Cannot decode first source frame at expected resolution')
    pad_for_video(sample)
    parent=args.output_root/args.sequence_name; parent.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='run_' if args.render else 'preflight_',dir=parent))
    metadata=dict(sequence=args.sequence_name,source_run=str(run),source_images=str(root/'img1'),
                  tracking_csv=str(csv_path),tracking_sha256=digest(csv_path),tracking_columns=list(data.columns),
                  source_configuration=json.loads((run/'final_model1_config.json').read_text()),
                  source_processing_metadata=processing,frame_count=count,fps=fps,width=size[0],height=size[1],
                  original_resolution=dict(width=size[0],height=size[1]),
                  encoded_resolution=dict(width=encoded_size[0],height=encoded_size[1]),
                  padding=dict(left=0,top=0,right=size[0]%2,bottom=size[1]%2,color_bgr=[0,0,0]),
                  pixel_preservation='Exact input-region equality and black borders checked before encoding; lossy MP4 does not preserve exact decoded pixel values.',
                  highlight_id=args.highlight_id,trail_frames=args.trail_frames,encoder=encoder,encoder_attempts=attempts,
                  opencv_version=cv2.__version__,colors_bgr=colors,render_requested=args.render,
                  original_video_note='JPEG pixels retained at original coordinates, then minimally padded right/bottom and re-encoded with a lossy MP4 codec; no resize or crop.',
                  trail_policy='Clear at any missing track observation; trails never bridge gaps.',
                  status='preflight_passed' if encoder else 'encoder_incompatible')
    def save():
        # This file belongs only to the newly created run.
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    save()
    if encoder is None:
        raise RuntimeError(f'No tested OpenCV MP4 encoder preserves padded size {encoded_size}. No full videos rendered. See {output}/metadata.json')
    if not args.render:
        print('Preflight passed; no full videos rendered. Metadata:',output); return
    writers=[]; trails={}; last={}; start=time.perf_counter()
    try:
        for name in ('original.mp4','tracked.mp4'):
            writer=cv2.VideoWriter(str(output/name),encoder['backend'],cv2.VideoWriter_fourcc(*encoder['codec']),fps,encoded_size)
            writers.append(writer)
            if not writer.isOpened(): raise RuntimeError('VideoWriter could not open '+name)
        groups=dict(tuple(data.groupby('frame')))
        for number,path in frames:
            image=cv2.imread(str(path))
            if image is None or (image.shape[1],image.shape[0])!=size: raise ValueError('Unreadable/invalid source frame '+str(path))
            writers[0].write(pad_for_video(image))
            annotated=image.copy()
            rows=groups.get(number,data.iloc[:0]); current=set(rows.track_id.astype(int))
            for tid in list(trails):
                if tid not in current: trails.pop(tid); last.pop(tid,None)
            for row in rows.itertuples(index=False):
                tid=int(row.track_id); color=colors[tid]
                x1,y1,x2,y2=map(lambda v:int(round(v)),(row.x1,row.y1,row.x2,row.y2))
                x1,x2=np.clip([x1,x2],0,size[0]-1); y1,y2=np.clip([y1,y2],0,size[1]-1)
                if args.trail_frames:
                    if last.get(tid)!=number-1: trails[tid]=deque(maxlen=args.trail_frames)
                    trails[tid].append(((int(x1)+int(x2))//2,(int(y1)+int(y2))//2)); last[tid]=number
                    if len(trails[tid])>1: cv2.polylines(annotated,[np.array(trails[tid],np.int32)],False,color,2)
                if tid==args.highlight_id:
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,255,255),6)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                label=f'ID {tid} {row.confidence:.2f}'+(' SELECTED' if tid==args.highlight_id else '')
                cv2.putText(annotated,label,(int(x1),max(15,int(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(annotated,label,(int(x1),max(15,int(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
            writers[1].write(pad_for_video(annotated))
            if number%100==0: print(f'Rendered {number}/{count}',flush=True)
        for writer in writers: writer.release()
        metadata['video_verification']={name:verify_video(output/name,count,fps,encoded_size) for name in ('original.mp4','tracked.mp4')}
        metadata.update(status='complete',elapsed_seconds=time.perf_counter()-start)
        print('Videos verified:',output)
    except Exception as exc:
        metadata.update(status='failed',error=str(exc)); raise
    finally:
        for writer in writers: writer.release()
        save()


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,KeyError) as exc: raise SystemExit(f'Visualization stopped: {exc}') from exc
