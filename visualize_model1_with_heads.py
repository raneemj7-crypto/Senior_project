"""Auxiliary head visualization, NOT a measured tracking improvement.

Default: preflight only. --execute caches heads and renders saved person tracks.
--head-cache /prior/output/head_detections.csv reuses a verified completed cache
without loading the model or running inference. No ByteTrack instance is created.
Empty frames have one CSV row with has_head=0 and blank detection fields.
"""
import argparse
from collections import deque
import configparser
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from visualize_model1_tracking import digest, color_map, pad_for_video, probe_encoder, verify_video

PROJECT=Path(__file__).resolve().parent
FIELDS=['frame','timestamp','has_head','x1','y1','x2','y2','confidence','class_id']
HEAD_COLOR=(255,0,255)
NOTICE='Auxiliary head-visualization experiment; not a measured tracking improvement. Heads have no person Track IDs.'


def save(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def validate_cache(path,signature):
    meta=json.loads(path.with_name('head_cache.json').read_text())
    if meta.get('signature')!=signature or meta.get('complete') is not True or meta.get('sha256')!=digest(path):
        raise ValueError('Head cache provenance, completion or checksum mismatch')
    df=pd.read_csv(path)
    if list(df.columns)!=FIELDS or not np.isfinite(df[['frame','timestamp','has_head']].to_numpy()).all():
        raise ValueError('Invalid head CSV schema')
    count=signature['frame_count']; fps=signature['fps']
    if sorted(df.frame.unique())!=list(range(1,count+1)) or not np.allclose(df.timestamp,(df.frame-1)/fps,rtol=0,atol=1e-8):
        raise ValueError('Head cache timeline incomplete or timestamps invalid')
    if not df.has_head.isin([0,1]).all(): raise ValueError('Invalid has_head indicator')
    groups={}
    width,height=signature['resolution']
    for frame,rows in df.groupby('frame',sort=True):
        if (rows.has_head==0).any():
            if len(rows)!=1 or not rows[FIELDS[3:]].isna().all().all():
                raise ValueError('Empty frame must have exactly one blank detection row')
            groups[int(frame)]=[]
        else:
            a=rows[FIELDS[3:]].to_numpy(dtype=float)
            if (not np.isfinite(a).all() or (a[:,5]!=1).any() or
                (a[:,2]<=a[:,0]).any() or (a[:,3]<=a[:,1]).any() or
                (a[:,:2]<0).any() or (a[:,2]>width).any() or (a[:,3]>height).any() or
                (a[:,4]<.25).any() or (a[:,4]>1).any()):
                raise ValueError('Invalid cached head detection')
            if rows.duplicated(['x1','y1','x2','y2','confidence','class_id']).any():
                raise ValueError('Duplicate head detection')
            groups[int(frame)]=a.tolist()
    return groups


def draw_box(image,box,color,label):
    height,width=image.shape[:2]
    x1,y1,x2,y2=map(lambda x:int(round(x)),box)
    x1,x2=np.clip([x1,x2],0,width-1); y1,y2=np.clip([y1,y2],0,height-1)
    cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
    anchor=(int(x1),max(15,int(y1)-4))
    cv2.putText(image,label,anchor,cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,label,anchor,cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-name',choices=['MOT20-04','MOT20-06','MOT20-07','MOT20-08'],default='MOT20-04')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--data-root',type=Path,default=Path.home()/'Desktop/senior-project-storage/raw/mot20/test')
    p.add_argument('--output-root',type=Path,default=PROJECT/'model1_visualizations')
    p.add_argument('--head-cache',type=Path)
    p.add_argument('--trail-frames',type=int,default=0)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if Path(sys.prefix).resolve()!=(PROJECT.parent/'.venv').resolve(): p.error('Use ../.venv/bin/python')
    if args.trail_frames<0: p.error('Trail length must be nonnegative')
    if args.run_dir is None:
        if args.sequence_name!='MOT20-04': p.error('Specify --run-dir for this sequence')
        args.run_dir=PROJECT/'model1_final_test/MOT20-04/run_551un9g_'
    run=args.run_dir.expanduser().resolve(); source=args.data_root.expanduser().resolve()/args.sequence_name
    summary=json.loads((run/'summary.json').read_text())
    config=json.loads((run/'final_model1_config.json').read_text())
    if summary['status']!='complete' or summary['sequence']!=args.sequence_name: raise ValueError('Wrong or incomplete person-tracking run')
    cfg=configparser.ConfigParser()
    with (source/'seqinfo.ini').open() as f: cfg.read_file(f)
    fps=cfg.getfloat('Sequence','frameRate'); count=cfg.getint('Sequence','seqLength')
    size=(cfg.getint('Sequence','imWidth'),cfg.getint('Sequence','imHeight'))
    if cfg.get('Sequence','name')!=args.sequence_name or not np.isfinite(fps) or fps<=0 or min(count,*size)<=0 or summary['frames']!=count:
        raise ValueError('Invalid source sequence metadata')
    frames=sorted((int(path.stem),path) for path in (source/'img1').glob('*.jpg'))
    if [n for n,_ in frames]!=list(range(1,count+1)): raise ValueError('Incomplete or duplicate JPEG frame IDs')
    for _,path in frames:
        with Image.open(path) as im:
            if im.size!=size: raise ValueError('JPEG dimensions differ from seqinfo')
    tracking=run/'tracking.csv'; people=pd.read_csv(tracking)
    required=['frame','track_id','x1','y1','x2','y2','confidence']
    if not np.isfinite(people[required].to_numpy(dtype=float)).all(): raise ValueError('Invalid person CSV values')
    if ((people.frame%1!=0)|(people.track_id%1!=0)|(~people.frame.between(1,count))|
        (people.track_id<0)|(people.x2<=people.x1)|(people.y2<=people.y1)|(~people.confidence.between(0,1))).any() or people.duplicated(['frame','track_id']).any():
        raise ValueError('Invalid person identities/boxes/frame IDs')
    if 'timestamp' in people and not np.allclose(people.timestamp,(people.frame-1)/fps,atol=1e-8,rtol=0): raise ValueError('Person timestamps disagree with source')
    weights=PROJECT/'weights/crowdhuman_yolov5m.pt'
    weight_hash=digest(weights)
    if weight_hash!=config['detector']['checkpoint_sha256']: raise ValueError('Checkpoint does not match completed run')
    signature=dict(schema=1,sequence=args.sequence_name,frame_count=count,fps=fps,resolution=list(size),
                   weights_sha256=weight_hash,classes={'person':0,'head':1},image_size=640,confidence=.25,nms_iou=.45,
                   device='cpu',source_inventory_sha256=hashlib.sha256(json.dumps([(n,str(path),path.stat().st_size,path.stat().st_mtime_ns) for n,path in frames]).encode()).hexdigest(),
                   detector_code={str(path.relative_to(PROJECT)):digest(path) for folder in ('models','utils') for path in (PROJECT/folder).rglob('*.py')},
                   versions={k:importlib.metadata.version(k) for k in ('torch','numpy')},opencv=cv2.__version__)
    model=None; heads=None
    if args.head_cache:
        args.head_cache=args.head_cache.expanduser().resolve()
        heads=validate_cache(args.head_cache,signature)
    else:
        import torch
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import non_max_suppression,scale_coords
        model=attempt_load(str(weights),map_location=torch.device('cpu')); model.eval()
        names=model.module.names if hasattr(model,'module') else model.names
        if names[0]!='person' or names[1]!='head': raise ValueError(f'Expected person=0/head=1; found {names}')
        stride=int(model.stride.max())
    encoder,attempts=probe_encoder(size,fps)
    if encoder is None: raise ValueError('No compatible padded MP4 encoder: '+str(attempts))
    print(f'Preflight passed: {args.sequence_name}, {count} frames, {fps:g} FPS, class mapping verified, codec {encoder["codec"]}.',flush=True)
    if not args.execute:
        print('No inference or full video rendering performed.'); return
    parent=args.output_root/args.sequence_name; parent.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='heads_',dir=parent))
    encoded=(size[0]+size[0]%2,size[1]+size[1]%2)
    metadata=dict(experiment=NOTICE,source_run=str(run),tracking_csv=str(tracking),tracking_sha256=digest(tracking),
                  source_configuration=config,head_cache_source=str(args.head_cache) if args.head_cache else None,
                  inference_settings=signature,original_resolution=list(size),encoded_resolution=list(encoded),
                  padding=dict(left=0,top=0,right=size[0]%2,bottom=size[1]%2),fps=fps,encoder=encoder,
                  trail_frames=args.trail_frames,head_color_bgr=list(HEAD_COLOR),
                  pixel_note='No resizing/cropping; padding preserves pixels before lossy MP4 encoding.',
                  cache_note='Source fingerprint uses path/size/mtime, not byte hashes of JPEGs.',
                  runner_sha256=digest(Path(__file__)))
    save(output/'metadata.json',metadata)
    cache=output/'head_detections.csv'; started=time.perf_counter(); video=None
    try:
        if heads is None:
            with cache.open('x',newline='') as f:
                writer=csv.writer(f); writer.writerow(FIELDS)
                for number,path in frames:
                    image=cv2.imread(str(path))
                    if image is None or (image.shape[1],image.shape[0])!=size: raise ValueError('Unreadable source '+str(path))
                    img=letterbox(image,640,stride=stride)[0]
                    tensor=torch.from_numpy(np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
                    with torch.no_grad():
                        det=non_max_suppression(model(tensor,augment=False)[0],.25,.45,classes=[1],agnostic=False)[0]
                    if len(det):
                        det[:,:4]=scale_coords(tensor.shape[2:],det[:,:4],image.shape).round()
                        for x1,y1,x2,y2,conf,cls in det.cpu().numpy():
                            writer.writerow([number,(number-1)/fps,1,float(x1),float(y1),float(x2),float(y2),float(conf),int(cls)])
                    else: writer.writerow([number,(number-1)/fps,0,'','','','','',''])
                    if number%100==0: print(f'Cached heads {number}/{count}',flush=True)
            save(output/'head_cache.json',dict(signature=signature,complete=True,sha256=digest(cache)))
            heads=validate_cache(cache,signature)
        else:
            shutil.copyfile(args.head_cache,cache)
            shutil.copyfile(args.head_cache.with_name('head_cache.json'),output/'head_cache.json')
            heads=validate_cache(cache,signature)
        cache_seconds=time.perf_counter()-started
        colors=color_map(people.track_id.unique())
        # Saturated magenta is reserved for heads; body palette has lower saturation.
        if HEAD_COLOR in colors.values(): raise ValueError('Head/person color collision')
        groups=dict(tuple(people.groupby('frame'))); trails={}; last={}
        video=cv2.VideoWriter(str(output/'tracked_with_heads.mp4'),encoder['backend'],cv2.VideoWriter_fourcc(*encoder['codec']),fps,encoded)
        if not video.isOpened(): raise ValueError('VideoWriter failed')
        for number,path in frames:
            image=cv2.imread(str(path))
            if image is None or (image.shape[1],image.shape[0])!=size: raise ValueError('Unreadable source '+str(path))
            current=groups.get(number,people.iloc[:0]); present=set(current.track_id.astype(int))
            for tid in list(trails):
                if tid not in present: trails.pop(tid); last.pop(tid,None)
            for row in current.itertuples(index=False):
                tid=int(row.track_id); color=colors[tid]
                if args.trail_frames:
                    if last.get(tid)!=number-1: trails[tid]=deque(maxlen=args.trail_frames)
                    trails[tid].append((int((row.x1+row.x2)/2),int((row.y1+row.y2)/2))); last[tid]=number
                    if len(trails[tid])>1: cv2.polylines(image,[np.array(trails[tid],np.int32)],False,color,2)
                draw_box(image,(row.x1,row.y1,row.x2,row.y2),color,f'ID {tid} {row.confidence:.2f}')
            for x1,y1,x2,y2,conf,_ in heads[number]: draw_box(image,(x1,y1,x2,y2),HEAD_COLOR,f'Head {conf:.2f}')
            cv2.putText(image,'Auxiliary heads | not a tracking improvement metric',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            video.write(pad_for_video(image))
            if number%100==0: print(f'Rendered {number}/{count}',flush=True)
        video.release(); video=None
        verification=verify_video(output/'tracked_with_heads.mp4',count,fps,encoded)
        save(output/'summary.json',dict(status='complete',experiment=NOTICE,video=verification,
                                       cached_frames=len(heads),head_detections=sum(map(len,heads.values())),
                                       cache_seconds=cache_seconds,total_seconds=time.perf_counter()-started))
        print('Completed auxiliary visualization:',output)
    except Exception as exc:
        save(output/'failure.json',dict(status='incomplete',error=str(exc))); raise
    finally:
        if video is not None: video.release()


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,RuntimeError,ImportError) as exc:
        raise SystemExit(f'Head visualization stopped: {exc}') from exc
