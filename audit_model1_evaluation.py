"""Read-only lightweight audit; no experiment imports, model loading or inference.

Run: ../.venv/bin/python audit_model1_evaluation.py
Writes only a unique report directory under model1_cross_validation/audit/.
Scans all selected CSV rows and JPEG headers, not decoded video/image streams.
"""
import argparse
import ast
import configparser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parent
NAMES = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path.home()/'Desktop/senior-project-storage/raw/mot20/train')
    parser.add_argument('--output-root',type=Path,default=PROJECT/'model1_cross_validation/audit')
    args=parser.parse_args()
    if Path(sys.prefix).resolve() != (PROJECT.parent/'.venv').resolve():
        parser.error('Use ../.venv/bin/python')
    findings=[]
    def add(level,area,evidence,finding,action):
        findings.append(dict(severity=level,area=area,evidence=evidence,finding=finding,action=action))
    scripts=[]
    for path in sorted(PROJECT.glob('*.py')):
        if not (path.name.startswith(('evaluate_','experiment_','track_mot20','validate_mot20','prepare_person','validate_person'))):
            continue
        text=path.read_text()
        tree=ast.parse(text)
        settings=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and (t.id.isupper() or t.id=='configs') for t in node.targets):
                settings.append(dict(line=node.lineno,expression=ast.get_source_segment(text,node)))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='ByteTrack':
                settings.append(dict(line=node.lineno,expression=ast.get_source_segment(text,node)))
        evaluator_lines=[dict(line=i,text=line.strip()) for i,line in enumerate(text.splitlines(),1)
                         if any(token in line for token in ('mark', 'class_id', 'max_iou=', 'ignored', 'frameid=', 'classes='))]
        scripts.append(dict(path=path.name,sha256=sha(path),settings=settings,evaluation_evidence=evaluator_lines))
    sequence_stats=[]
    meta={}
    for name in NAMES:
        root=args.data_root/name
        try:
            cfg=configparser.ConfigParser()
            with (root/'seqinfo.ini').open() as f: cfg.read_file(f)
            section=cfg['Sequence']
            count,fps,w,h=(int(section[k]) for k in ('seqLength','frameRate','imWidth','imHeight'))
            paths=list((root/'img1').glob('*.jpg'))
            ids=[int(p.stem) for p in paths]
            missing=sorted(set(range(1,count+1))-set(ids))
            extra=sorted(set(ids)-set(range(1,count+1)))
            bad_images=[]
            for path in paths:
                try:
                    with Image.open(path) as image:
                        if image.size!=(w,h): bad_images.append(path.name)
                except OSError: bad_images.append(path.name)
            gt_path=root/'gt/gt.txt'
            gt=pd.read_csv(gt_path,header=None,names=['frame','id','x','y','w','h','mark','class','visibility'])
            retained=gt[(gt['mark']==1)&(gt['class']==1)]
            row=dict(sequence=name,frames=count,jpeg_count=len(ids),fps=fps,width=w,height=h,
                     missing_frames=len(missing),extra_frames=len(extra),duplicate_frame_ids=len(ids)-len(set(ids)),
                     bad_image_headers=len(bad_images),gt_rows=len(gt),retained_gt_rows=len(retained),
                     excluded_gt_rows=len(gt)-len(retained),gt_classes=json.dumps(gt['class'].value_counts().to_dict()),
                     gt_invalid_boxes=int(((gt.w<=0)|(gt.h<=0)).sum()),
                     gt_out_of_range_frames=int((~gt.frame.between(1,count)).sum()),
                     gt_boundary_crossing_boxes=int(((gt.x<0)|(gt.y<0)|(gt.x+gt.w>w)|(gt.y+gt.h>h)).sum()),
                     gt_sha256=sha(gt_path),seqinfo_sha256=sha(root/'seqinfo.ini'))
            sequence_stats.append(row); meta[name]=row
            level='ERROR' if missing or extra or bad_images or row['duplicate_frame_ids'] else 'INFO'
            add(level,'input frames',name,f'{len(paths)}/{count} JPEGs; missing={len(missing)}, extra={len(extra)}, bad headers={len(bad_images)}',
                'Header checks do not prove full JPEG decode or video equivalence.')
            add('WARN','GT filtering',name,f"{len(gt)-len(retained)} annotations excluded by mark==1 and class==1; classes={row['gt_classes']}",
                'Audit benchmark-specific distractor preprocessing before formal scoring; excluded GT alone does not remove associated predictions.')
        except Exception as exc:
            add('ERROR','inputs',name,str(exc),'Restore the exact sequence/metadata; do not substitute data.')
    outputs=[]
    for pattern in ('confidence_experiments/tracking*.csv','matching_experiments/tracking*.csv',
                    'yolov5_mot20*.csv','yolo26_mot20_tracking.csv','head_assisted_experiments/*/body_tracks.csv'):
        outputs.extend((p,'MOT20-01') for p in PROJECT.glob(pattern))
    outputs.extend((p,'MOT20-02') for p in PROJECT.glob('validation_mot20_02/tracking*.csv'))
    for name in NAMES[2:]:
        outputs.append((PROJECT/'validation_mot20_additional'/name/'tracking.csv',name))
    tracking=[]
    for path,name in outputs:
        try:
            data=pd.read_csv(path)
            m=meta[name]
            numeric=['frame','track_id','x1','y1','x2','y2','width','height','center_x','center_y','confidence']
            a=data[numeric].to_numpy(dtype=float)
            bad_finite=~np.isfinite(a).all(axis=1)
            bad_size=(data.x2<=data.x1)|(data.y2<=data.y1)
            inconsistent=(~np.isclose(data.width,data.x2-data.x1)|~np.isclose(data.height,data.y2-data.y1)
                          |~np.isclose(data.center_x,(data.x1+data.x2)/2)|~np.isclose(data.center_y,(data.y1+data.y2)/2))
            boundary=(data.x1<0)|(data.y1<0)|(data.x2>m['width'])|(data.y2>m['height'])
            result=dict(path=str(path.relative_to(PROJECT)),sequence=name,rows=len(data),tracks=int(data.track_id.nunique()),
                        represented_frames=int(data.frame.nunique()),frames_without_rows=m['frames']-int(data.frame.nunique()),
                        duplicate_keys=int(data.duplicated(['frame','track_id']).sum()),nonfinite_rows=int(bad_finite.sum()),
                        invalid_sizes=int(bad_size.sum()),inconsistent_geometry=int(inconsistent.sum()),boundary_crossings=int(boundary.sum()),
                        invalid_frames=int((~data.frame.between(1,m['frames'])|(data.frame%1!=0)).sum()),
                        noninteger_ids=int((data.track_id%1!=0).sum()),invalid_confidence=int((~data.confidence.between(0,1)).sum()),
                        frame_order_decreases=int((data.frame.diff()<0).sum()),sha256=sha(path))
            tracking.append(result)
            defects=sum(result[k] for k in ('duplicate_keys','nonfinite_rows','invalid_sizes','inconsistent_geometry','invalid_frames','noninteger_ids','invalid_confidence','frame_order_decreases'))
            add('ERROR' if defects else 'INFO','tracking structure',result['path'],
                f"rows={len(data)}, structural defects={defects}, boundary crossings={int(boundary.sum())}, frames without rows={result['frames_without_rows']}",
                'Empty frames are not proof of skipped processing. Retain out-of-image full-body boxes for evaluator convention review; do not silently clamp GT.')
        except Exception as exc:
            add('ERROR','tracking structure',str(path),str(exc),'Resolve input/schema mismatch before comparison.')
    # Preserve small saved metrics/configuration as evidence without recomputing scores.
    artifacts={}
    for pattern in ('*metrics*.csv','*experiments/metrics*.csv','validation_mot20_02/metrics*.csv',
                    'validation_mot20_additional/**/metrics*.csv','validation_mot20_additional/configuration.json'):
        for path in PROJECT.glob(pattern):
            artifacts[str(path.relative_to(PROJECT))]=dict(sha256=sha(path),text=path.read_text())
    observations=[
      ('HIGH','ignored-region handling','evaluate_tracking_*.py; evaluate_confidence_mot20.py; evaluate_matching_mot20.py; validate_mot20_sequences.py',
       'Marked pedestrian GT and IoU 0.5 are used, but no equivalent MOT20 distractor preprocessing is implemented. Scores are diagnostic.',
       'Pin TrackEval MOT20 preprocessing with DO_PREPROC enabled; verify coordinate/export conventions and score all candidates identically.'),
      ('HIGH','evaluator mismatch','evaluate_mot20.py: PRED_DIR and final printed label',
       'Reads runs/yolo26/mot20_01/labels while printing YOLOv5 CrowdHuman. Detection-only custom ignored-box matching differs from motmetrics identity tracking.',
       'Do not attribute these scores to YOLOv5 or compare its F1 directly with IDF1/MOTA; verify provenance first.'),
      ('HIGH','runtime defaults','track_mot20_yolov5.py: sv.ByteTrack()',
       'Original baseline delegates tracker parameters to library defaults; tuned script uses lost buffer 60 and C uses 60/matching 0.7.',
       'Do not assume baseline matches explicit 30/0.8 runs; recover historical Supervision version/defaults.'),
      ('WARN','source equivalence','track_mot20_yolov5*.py versus experiment_* and validate_*',
       'Earlier runs use mot20_01.mp4; later runs use original JPEGs. Enumerating JPEGs is valid only if original filenames are complete and ordered.',
       'Do not infer video frame/pixel equivalence from CSV counts; compare source conversion and alignment before paired claims.'),
      ('HIGH','exploratory selection','confidence_experiments; matching_experiments; validation_mot20_02 A/B',
       'Confidence/matching sweeps on 01 and buffer alternatives on 02 expose those labels to selection. This is evidence of exploration, not proof which parameters were chosen from which results.',
       'Record selection chronology. Treat existing scores as exploratory; no retrospective claim of pristine held-out evaluation.'),
      ('WARN','historical reproducibility','experiment_confidence_mot20.py versus tracking_conf_0.15.csv and tracking_conf_0.20.csv',
       'Current script only runs confidence 0.25 although older threshold artifacts remain.',
       'Recover generating commit/configuration; current source hash does not authenticate past artifact settings.'),
      ('HIGH','scene independence','MOTChallenge sequence descriptions; seqinfo.ini',
       'Different sequence IDs do not establish independent scenes. MOT20-01 and 02 both describe an indoor train station; shared camera/site must be checked.',
       'Call the four-fold design leave-one-sequence-out until source/camera provenance proves scene disjointness. If scenes overlap, group them; four independent scene folds may be impossible.'),
      ('WARN','pretraining provenance','weights/crowdhuman_yolov5m.pt',
       'Filename does not establish training data, checkpoint lineage, or absence of MOT20 exposure.',
       'Document checkpoint source/hash/training datasets; do not load the checkpoint during this audit.'),
      ('WARN','coordinate origin','scale_coords + x2-x1 exports; GT xywh read directly',
       'CSV geometry can be internally consistent while pixel-origin conventions differ. No explicit origin conversion is documented.',
       'Validate source/GT coordinate convention with an overlay and benchmark-format round trip before changing offsets; no arbitrary +/-1 correction.'),
      ('INFO','auxiliary observations','experiment_head_assisted_tracking.py; module1_sequences',
       'Head-only observations and pose crop windows are not full-body MOT tracks; crop continuity is not verified identity accuracy.',
       'Use body_tracks.csv for body tracking only, and keep auxiliary coverage/pose readiness separate.'),
    ]
    for item in observations: add(*item)
    folds=[dict(fold=i+1,held_out=name,development=[n for n in NAMES if n!=name]) for i,name in enumerate(NAMES)]
    protocol='''Freeze this protocol before any further parameter selection. The four outer folds hold out 01, 02, 03 and 05 once each, respectively; the other three are development data.
For each outer fold, perform inner leave-one-development-sequence-out selection: train on two and validate on the third, rotating all three. If the detector stays frozen, select only tracker/detection parameters using these inner validations; record that no detector training occurred. If fine-tuning is allowed, initialize each inner fit independently from the same documented external checkpoint, and use only its inner training data. Never reuse a checkpoint trained on another fold's held-out sequence.
Predeclare candidate ranges, primary metric (e.g. mean inner IDF1), tie-breakers, budget, seed, augmentations, stopping rule and evaluator version. Freeze all choices after inner selection. Refit on the three development sequences if training is part of the declared pipeline; otherwise use the frozen detector. Reset ByteTrack per sequence. Evaluate the outer held-out sequence once, with no adaptation or choice based on its labels, images or outputs beyond fixed inference and seqinfo decoding.
Keep whole source sequences and every derived crop/window within their fold. Never randomly split adjacent frames or crops. Verify scene/camera groups before calling this scene-disjoint; if a group spans IDs, use grouped outer AND inner splits and accept fewer than four scene folds.
Export every frame with original IDs, person-only boxes, explicit coordinate convention and identical pinned TrackEval MOT20 preprocessing. Preserve diagnostic motmetrics separately if desired. Save fold configs, source/checkpoint hashes, training logs, selection scores and locked predictions before exposing outer scores.
Report all four outer sequence scores, macro mean and variability, plus evaluator-computed pooled counts/metrics (do not average IDF1 or MOTA as a pooled score). Sequence-level uncertainty with four units is limited; do not treat frames as independent samples. Do not tune again using the outer mean.
All four sequences have already been inspected and 01/02 were explored. A new nested run is a reproducible retrospective evaluation, not an untouched prospective test. For a stronger generalization claim obtain an additional independent scene held out from all development. Prior human decisions cannot be unlearned by relabeling files as folds.
'''
    comparison='''| Results | Comparison status |
|---|---|
| MOT20-01 confidence variants | Within-sequence diagnostic ablation only; recover historical configs, same detector/source/evaluator required. |
| MOT20-01 matching 0.7/0.8/0.9 | Controlled within-sequence diagnostic matching ablation according to source; not independent test folds. |
| MOT20-02 A vs B | Within-sequence buffer ablation (30 vs 60); potential development use. |
| 01 confidence .25, 02 A, 03/05 fixed runs | Nominal common 640/.25/.45, activation .25, buffer 30, matching .8; useful descriptive per-scene performance, not paired algorithm improvement or completed LOSO. Historical configs must be verified. |
| MP4 baseline/tuned/C vs JPEG runs | Conditional only: source equivalence and historical default tracker settings unverified. |
| YOLO26 detection/custom ignore scores vs YOLOv5 motmetrics | Not directly comparable detector attribution, metric, ignore handling or source provenance. |
| Head-only/crop-window statistics vs body MOT scores | Different tasks; do not count auxiliary observations as body tracking successes. |
'''
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='run_',dir=args.output_root))
    for filename,rows in [('findings.csv',findings),('sequence_inventory.csv',sequence_stats),('tracking_inventory.csv',tracking),('folds.csv',folds)]:
        pd.DataFrame(rows).to_csv(output/filename,index=False)
    evidence=dict(created_utc=datetime.now(timezone.utc).isoformat(),scripts=scripts,saved_artifacts=artifacts,
                  dependencies={n:importlib.metadata.version(n) for n in ('pandas','numpy','Pillow','supervision','motmetrics')},
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
                  limits='No inference, training, score recomputation, full JPEG decode, video decode, or checkpoint loading. CSV hashes identify current files only.')
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2))
    lines=['# Model 1 evaluation audit','',evidence['limits'],'',
           '## Input and tracking checks','',pd.DataFrame(sequence_stats).to_csv(index=False),
           f'Inspected {len(scripts)} scripts and {len(tracking)} tracking CSVs. Detailed counts are in the inventory CSVs.',
           '', '## Findings','']
    for f in findings:
        lines.append(f"- **{f['severity']} — {f['area']}** ({f['evidence']}): {f['finding']} Action: {f['action']}")
    lines.extend(['','## Comparability','',comparison,'## Four-fold protocol','',
                  '\n'.join(f"- Fold {f['fold']}: test {f['held_out']}; develop on {', '.join(f['development'])}." for f in folds),
                  '',protocol,'## References','',
                  '- [MOT20 sequence descriptions](https://motchallenge.net/data/MOT20/)',
                  '- [TrackEval MOTChallenge preprocessing source](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py)',
                  'References were consulted when preparing this audit; the script performs no network requests. Pin an exact evaluator commit before formal evaluation.'])
    (output/'audit_report.md').write_text('\n'.join(lines))
    print(f'Audit complete: {output}')
    print(f'{len(sequence_stats)} sequences; {len(tracking)} tracking CSVs; {len(findings)} findings.')
    print(pd.DataFrame(tracking)[['path','rows','duplicate_keys','invalid_sizes','inconsistent_geometry','boundary_crossings','invalid_frames']].to_string(index=False))


if __name__=='__main__':
    main()
