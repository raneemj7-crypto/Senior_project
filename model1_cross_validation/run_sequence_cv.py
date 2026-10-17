"""Plan by default; --execute explicitly runs retrospective sequence-held-out CV.

../.venv/bin/python model1_cross_validation/run_sequence_cv.py --prediction-origin as-stored
No training; fixed detector. Never accesses MOT20-04/06/07/08.
"""
import argparse
import configparser
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(Path(__file__).parent/'evaluation'))
import numpy as np
import pandas as pd
import supervision as sv
from mot20_protocol import GT_COLUMNS, METRICS, STATUS, evaluate, load_predictions, validate_rows

NAMES=('MOT20-01','MOT20-02','MOT20-03','MOT20-05')
CANDIDATES={
    'baseline':dict(track_activation_threshold=.25,lost_track_buffer=30,minimum_matching_threshold=.8,minimum_consecutive_frames=1),
    'buffer60':dict(track_activation_threshold=.25,lost_track_buffer=60,minimum_matching_threshold=.8,minimum_consecutive_frames=1),
    'match07':dict(track_activation_threshold=.25,lost_track_buffer=30,minimum_matching_threshold=.7,minimum_consecutive_frames=1),
}
COLUMNS=['frame','track_id','x1','y1','x2','y2','center_x','center_y','width','height','confidence']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write_json(path,data):
    with path.open('x') as f: json.dump(data,f,indent=2,allow_nan=False)


def select_candidate(rows,development):
    """Equal-sequence mean IDF1; exact ties favor predefined candidate order."""
    scores={}
    for candidate in CANDIDATES:
        selected=[r for r in rows if r['candidate']==candidate and r['sequence'] in development]
        if len(selected)!=len(development) or {r['sequence'] for r in selected}!=set(development):
            raise ValueError('Incomplete/duplicate development evidence')
        values=[float(r['idf1']) for r in selected]
        if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite selection score')
        scores[candidate]=sum(values)/len(values)
    return max(CANDIDATES,key=lambda c:scores[c]),scores


def validate_cache(path,signature):
    meta=json.loads((path/'metadata.json').read_text())
    if meta['signature']!=signature or not meta.get('complete'):
        raise ValueError('Cache provenance mismatch/incomplete: '+str(path))
    data_path=path/'detections.jsonl'
    if digest(data_path)!=meta['sha256']: raise ValueError('Cache checksum mismatch')
    count=0
    with data_path.open() as f:
        for count,line in enumerate(f,1):
            record=json.loads(line)
            if record['frame']!=count: raise ValueError('Missing/duplicate cache frame')
            a=np.asarray(record['detections'],dtype=float).reshape(-1,6)
            if not np.isfinite(a).all(): raise ValueError('Nonfinite cache value')
            if len(a) and ((a[:,2]<=a[:,0]).any() or (a[:,3]<=a[:,1]).any() or
                           (a[:,4]<.25).any() or (a[:,4]>1).any() or (a[:,5]!=0).any()):
                raise ValueError('Invalid cache detection')
    if count!=signature['frames']: raise ValueError('Incomplete cache timeline')
    return data_path


def make_cache(path,sequence,signature,model):
    import cv2
    import torch
    from utils.datasets import letterbox
    from utils.general import non_max_suppression,scale_coords
    path.mkdir()
    stride=int(model.stride.max())
    with (path/'detections.jsonl').open('x') as handle:
        for number,source in sequence['images']:
            frame=cv2.imread(str(source))
            if frame is None: raise ValueError('Unreadable frame '+str(source))
            if frame.shape[:2]!=(sequence['height'],sequence['width']): raise ValueError('Image dimensions disagree with seqinfo')
            img=letterbox(frame,640,stride=stride)[0]
            img=np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1))
            tensor=torch.from_numpy(img).float().unsqueeze(0)/255
            with torch.no_grad():
                pred=model(tensor,augment=False)[0]
                det=non_max_suppression(pred,.25,.45,classes=[0],agnostic=False)[0]
            if len(det): det[:,:4]=scale_coords(tensor.shape[2:],det[:,:4],frame.shape).round()
            handle.write(json.dumps(dict(frame=number,detections=det.cpu().numpy().tolist()),allow_nan=False)+'\n')
            if number%250==0: print(signature['sequence'],number,'detections cached',flush=True)
    write_json(path/'metadata.json',dict(signature=signature,complete=True,sha256=digest(path/'detections.jsonl')))
    return validate_cache(path,signature)


def track(cache,sequence,params,path):
    import csv
    tracker=sv.ByteTrack(**params,frame_rate=sequence['fps'])
    with path.open('x',newline='') as f,cache.open() as source:
        writer=csv.writer(f); writer.writerow(COLUMNS)
        for line in source:
            r=json.loads(line); a=np.asarray(r['detections'],dtype=np.float32).reshape(-1,6)
            detections=sv.Detections(xyxy=a[:,:4].copy(),confidence=a[:,4].copy(),class_id=np.zeros(len(a),dtype=int)) if len(a) else sv.Detections.empty()
            tracked=tracker.update_with_detections(detections)
            for box,conf,tid in zip(tracked.xyxy,tracked.confidence,tracked.tracker_id):
                if tid is None: continue
                x1,y1,x2,y2=map(float,box)
                writer.writerow([r['frame'],int(tid),x1,y1,x2,y2,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,float(conf)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='Explicitly authorize inference/tracking/scoring; default is plan only')
    p.add_argument('--prediction-origin',required=True,choices=['as-stored','zero-to-one'])
    p.add_argument('--data-root',type=Path,default=Path.home()/'Desktop/senior-project-storage/raw/mot20/train')
    p.add_argument('--output-root',type=Path,default=PROJECT/'model1_cross_validation/runs')
    p.add_argument('--cache-dir',type=Path,help='Optional completed cache root from a prior run; reject mismatches, do not overwrite')
    args=p.parse_args()
    if Path(sys.prefix).resolve()!=(PROJECT.parent/'.venv').resolve(): p.error('Use ../.venv/bin/python')
    import torch
    import cv2
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression,scale_coords
    weights=PROJECT/'weights/crowdhuman_yolov5m.pt'
    versions={n:importlib.metadata.version(n) for n in ['torch','supervision','numpy','pandas','motmetrics','scipy']}
    detector_sources={str(path.relative_to(PROJECT)):digest(path) for folder in ['models','utils'] for path in (PROJECT/folder).rglob('*.py')}
    sequences={}; signatures={}; reusable={}
    for name in NAMES:
        root=args.data_root/name; cfg=configparser.ConfigParser()
        with (root/'seqinfo.ini').open() as f: cfg.read_file(f)
        total=cfg.getint('Sequence','seqLength'); fps=cfg.getfloat('Sequence','frameRate')
        if fps<=0 or cfg.get('Sequence','name')!=name: raise ValueError('Invalid sequence metadata')
        images=sorted((int(path.stem),path) for path in (root/'img1').glob('*.jpg'))
        if [f for f,_ in images]!=list(range(1,total+1)): raise ValueError('Incomplete source frame IDs')
        gt=pd.read_csv(root/'gt/gt.txt',header=None,names=GT_COLUMNS); validate_rows(gt,total,gt=True)
        sequences[name]=dict(images=images,frames=total,fps=fps,gt=root/'gt/gt.txt',
                             width=cfg.getint('Sequence','imWidth'),height=cfg.getint('Sequence','imHeight'))
        fingerprint=hashlib.sha256(json.dumps([(f,str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns) for f,path in images]).encode()).hexdigest()
        signatures[name]=dict(schema=1,sequence=name,frames=total,fps=fps,source_inventory=fingerprint,
                              weights_sha256=digest(weights),detector='YOLOv5 CrowdHuman',person_class=0,
                              image_size=640,confidence=.25,nms_iou=.45,device='cpu',detector_sources=detector_sources,
                              versions=versions,opencv=cv2.__version__)
        if args.cache_dir and (args.cache_dir/name).exists(): reusable[name]=validate_cache(args.cache_dir/name,signatures[name])
    for params in CANDIDATES.values(): sv.ByteTrack(**params,frame_rate=25).update_with_detections(sv.Detections.empty())
    args.output_root.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='cv_' if args.execute else 'plan_',dir=args.output_root))
    folds=[dict(fold=i+1,held_out=n,development=[x for x in NAMES if x!=n]) for i,n in enumerate(NAMES)]
    total=sum(s['frames'] for s in sequences.values())
    plan=dict(status='retrospective sequence-held-out cross-validation; not pristine scene-disjoint testing',
              evaluator_status=STATUS,detector='YOLOv5 CrowdHuman',tracker='Supervision ByteTrack',
              candidates=CANDIDATES,selection='macro development IDF1; exact ties favor baseline then buffer60 then match07',
              folds=folds,prediction_origin=args.prediction_origin,versions=versions,
              inputs={n:dict(frames=s['frames'],fps=s['fps'],gt_sha256=digest(s['gt'])) for n,s in sequences.items()},
              cache_signatures=signatures,reusable_caches={n:str(path) for n,path in reusable.items()},
              workload=dict(unique_source_frames=total,new_detector_frames=sum(s['frames'] for n,s in sequences.items() if n not in reusable),
                            development_tracking_evaluations=36,outer_evaluations=4,tracking_frame_updates=10*total),
              runner_sha256=digest(Path(__file__)),evaluator_sha256=digest(Path(__file__).parent/'evaluation/mot20_protocol.py'),
              cache_reuse_verdict='Existing tracking CSVs are filtered track observations, not raw detections. Legacy YOLO26 labels are wrong detector. Only validated caches made by this runner are reusable.',
              execute=args.execute)
    write_json(run/'plan.json',plan)
    shutil.copyfile(Path(__file__).with_name('CV_PROTOCOL.md'),run/'CV_PROTOCOL.md')
    print(json.dumps(plan['workload'],indent=2)); print('Output:',run,flush=True)
    if not args.execute:
        print('Plan checks passed. No model loaded, inference, tracking datasets or scoring performed.'); return
    (run/'cache').mkdir()
    model=None; caches=dict(reusable)
    for name,s in sequences.items():
        if name not in caches:
            if model is None:
                model=attempt_load(str(weights),map_location=torch.device('cpu')); model.eval()
                names=model.module.names if hasattr(model,'module') else model.names
                if names[0]!='person': raise ValueError('Class 0 must be person')
            caches[name]=make_cache(run/'cache'/name,s,signatures[name],model)
    outer=[]
    for fold in folds:
        directory=run/f"fold_{fold['fold']}"; directory.mkdir()
        evidence=[]
        for name in fold['development']:
            s=sequences[name]; gt=pd.read_csv(s['gt'],header=None,names=GT_COLUMNS)
            for candidate,params in CANDIDATES.items():
                output=directory/f'development_{name}_{candidate}.csv'
                track(caches[name],s,params,output)
                predictions=load_predictions(output,s['frames'],1 if args.prediction_origin=='zero-to-one' else 0)
                evidence.append(dict(sequence=name,candidate=candidate,**evaluate(gt,predictions,s['frames'])))
        pd.DataFrame(evidence).to_csv(directory/'development_metrics.csv',index=False)
        selected,scores=select_candidate(evidence,fold['development'])
        # Persist selection before held-out GT is loaded for scoring.
        write_json(directory/'selection.json',dict(**fold,selected=selected,parameters=CANDIDATES[selected],scores=scores))
        name=fold['held_out']; s=sequences[name]
        path=directory/f'held_out_{name}.csv'; track(caches[name],s,CANDIDATES[selected],path)
        predictions=load_predictions(path,s['frames'],1 if args.prediction_origin=='zero-to-one' else 0)
        gt=pd.read_csv(s['gt'],header=None,names=GT_COLUMNS)
        result=dict(fold=fold['fold'],sequence=name,selected=selected,**evaluate(gt,predictions,s['frames']))
        pd.DataFrame([result]).to_csv(directory/'held_out_metrics.csv',index=False); outer.append(result)
    df=pd.DataFrame(outer); df.to_csv(run/'outer_metrics.csv',index=False)
    df[METRICS].agg(['mean','std']).to_csv(run/'outer_summary.csv')
    print('Completed retrospective diagnostic CV:',run)


if __name__=='__main__': main()
