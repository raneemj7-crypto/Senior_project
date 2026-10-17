"""Standardized MOT20 evaluator. --check-only inspects inputs without scoring."""
import argparse
import configparser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
import pandas as pd
from mot20_protocol import GT_COLUMNS, STATUS, load_predictions, validate_rows, evaluate

PROJECT=Path(__file__).resolve().parents[2]
PRESETS={'MOT20-01':'confidence_experiments/tracking_conf_0.25.csv',
         'MOT20-02':'validation_mot20_02/tracking_A.csv',
         'MOT20-03':'validation_mot20_additional/MOT20-03/tracking.csv',
         'MOT20-05':'validation_mot20_additional/MOT20-05/tracking.csv'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences',nargs='+',choices=list(PRESETS),default=list(PRESETS))
    parser.add_argument('--data-root',type=Path,default=Path.home()/'Desktop/senior-project-storage/raw/mot20/train')
    parser.add_argument('--prediction',action='append',default=[],metavar='SEQUENCE=CSV')
    parser.add_argument('--detector-name',default='YOLOv5 CrowdHuman')
    parser.add_argument('--tracker-name',default='Supervision ByteTrack')
    parser.add_argument('--prediction-origin',choices=['as-stored','zero-to-one'],required=True,
                        help='Explicit coordinate policy: preserve stored values, or add 1 to prediction x/y only')
    parser.add_argument('--output-root',type=Path,default=PROJECT/'model1_cross_validation/evaluation/results')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if Path(sys.prefix).resolve() != (PROJECT.parent/'.venv').resolve(): parser.error('Use ../.venv/bin/python')
    paths={k:PROJECT/v for k,v in PRESETS.items()}
    for override in args.prediction:
        name,path=override.split('=',1)
        if name not in PRESETS: parser.error('Unknown prediction sequence '+name)
        paths[name]=Path(path).expanduser().resolve()
    provenance=[]; results=[]
    for name in args.sequences:
        root=args.data_root/name; cfg=configparser.ConfigParser()
        with (root/'seqinfo.ini').open() as f: cfg.read_file(f)
        total=cfg.getint('Sequence','seqLength'); fps=cfg.getfloat('Sequence','frameRate')
        w,h=cfg.getint('Sequence','imWidth'),cfg.getint('Sequence','imHeight')
        frame_ids=sorted(int(p.stem) for p in (root/'img1').glob('*.jpg'))
        if frame_ids!=list(range(1,total+1)): raise ValueError('Incomplete original frame set: '+name)
        gt_path=root/'gt/gt.txt'; gt=pd.read_csv(gt_path,header=None,names=GT_COLUMNS)
        validate_rows(gt,total,gt=True)
        pred=load_predictions(paths[name],total,1 if args.prediction_origin=='zero-to-one' else 0)
        provenance.append(dict(sequence=name,frames=total,fps=fps,gt_classes=gt.class_id.value_counts().to_dict(),
                               mark_counts=gt.mark.value_counts().to_dict(),visibility=gt.visibility.describe().to_dict(),
                               zero_visibility_pedestrians=int(((gt.class_id==1)&(gt.visibility==0)).sum()),
                               prediction=str(paths[name]),prediction_sha256=digest(paths[name]),gt_sha256=digest(gt_path),
                               boundary_crossings=int(((pred.x<0)|(pred.y<0)|(pred.x+pred.w>w)|(pred.y+pred.h>h)).sum())))
        print(name,'input checks passed; GT visibility retained without cutoff')
        if not args.check_only:
            results.append(dict(sequence=name,detector=args.detector_name,tracker=args.tracker_name,
                                status=STATUS,**evaluate(gt,pred,total)))
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='check_' if args.check_only else 'run_',dir=args.output_root))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(status=STATUS,backend='motmetrics',inputs=provenance,
                  versions={k:importlib.metadata.version(k) for k in ['motmetrics','numpy','scipy','pandas']},
                  evaluator_sha256=digest(Path(__file__)),protocol_sha256=digest(Path(__file__).with_name('mot20_protocol.py')))
    (output/'metadata.json').write_text(json.dumps(config,indent=2))
    shutil.copyfile(Path(__file__).with_name('PROTOCOL.md'),output/'PROTOCOL.md')
    if results: pd.DataFrame(results).to_csv(output/'metrics.csv',index=False)
    print('No scoring performed.' if args.check_only else STATUS)
    print(output)


if __name__=='__main__': main()
