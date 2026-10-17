"""Corrected replacement for mislabeled evaluate_mot20.py: YOLO26 detections.

Detection precision/recall/F1 only, not identity tracking or AP. No score filtering
is introduced; five-column labels have unknown confidence represented as 1.
"""
import argparse
import configparser
import json
from pathlib import Path
import shutil
import sys
import tempfile
import pandas as pd
from mot20_protocol import GT_COLUMNS, STATUS, evaluate, validate_rows
from evaluate_person_tracking import PROJECT, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-dir',type=Path,default=Path.home()/'Desktop/senior-project-storage/raw/mot20/train/MOT20-01')
    p.add_argument('--labels-dir',type=Path,default=PROJECT/'runs/yolo26/mot20_01/labels')
    p.add_argument('--detector-name',required=True,help='Actual detector identity; legacy input is YOLO26, not YOLOv5')
    p.add_argument('--prediction-origin',choices=['as-stored','zero-to-one'],required=True)
    p.add_argument('--output-root',type=Path,default=PROJECT/'model1_cross_validation/evaluation/results')
    args=p.parse_args()
    if Path(sys.prefix).resolve()!=(PROJECT.parent/'.venv').resolve(): p.error('Use ../.venv/bin/python')
    if not args.labels_dir.is_dir(): raise FileNotFoundError(args.labels_dir)
    cfg=configparser.ConfigParser(); cfg.read(args.sequence_dir/'seqinfo.ini')
    total=cfg.getint('Sequence','seqLength'); w=cfg.getint('Sequence','imWidth'); h=cfg.getint('Sequence','imHeight')
    gt=pd.read_csv(args.sequence_dir/'gt/gt.txt',header=None,names=GT_COLUMNS)
    rows=[]; files={}
    for path in sorted(args.labels_dir.glob('*.txt')):
        frame=int(path.stem)
        if not 1<=frame<=total: raise ValueError('Invalid label frame '+str(path))
        files[str(path)]=digest(path)
        for line in path.read_text().splitlines():
            values=list(map(float,line.split()))
            if len(values) not in (5,6): raise ValueError('Expected YOLO class cx cy w h [confidence]')
            if values[0]!=0: continue
            cx,cy,bw,bh=values[1:5]
            offset=1 if args.prediction_origin=='zero-to-one' else 0
            rows.append([frame,len(rows)+1,(cx-bw/2)*w+offset,(cy-bh/2)*h+offset,bw*w,bh*h,
                         values[5] if len(values)==6 else 1.0])
    if not files: raise ValueError('No label files: refuse an accidental empty input directory')
    pred=pd.DataFrame(rows,columns=['frame','id','x','y','w','h','confidence'])
    validate_rows(pred,total)
    result=evaluate(gt,pred,total,detection_only=True)
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='detection_',dir=args.output_root))
    metadata=dict(detector=args.detector_name,tracker='none: detection-only',status=STATUS,
                  sequence=args.sequence_dir.name,origin=args.prediction_origin,label_hashes=files,
                  gt_sha256=digest(args.sequence_dir/'gt/gt.txt'),
                  warning='Missing per-frame label files are treated as empty detections; completeness requires external run logs.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    pd.DataFrame([dict(detector=args.detector_name,tracker='none',**result)]).to_csv(output/'detection_metrics.csv',index=False)
    shutil.copyfile(Path(__file__).with_name('PROTOCOL.md'),output/'PROTOCOL.md')
    print(args.detector_name,'detection-only diagnostic results:',output)


if __name__=='__main__': main()
