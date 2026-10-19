"""Controlled, provisional comparison; inference receives frames, never labels."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from ultralytics import YOLO
from integrated_model2 import IntegratedModel2
from batch_evaluate import discover, reset_tracking
from test_video import extract_tracks
from evaluation.model_comparison.adapters import RaneemAdapter, SouadAdapter, RANEEM, digest

FPS = 15.0  # Provisional source FPS: guarantees aligned 30-frame shared clips.
STRIDE = 10  # Original prepare_training_data.py stride, selected before evaluation.
MODEL_IDS = ['souad_integrated', 'souad_shared_clips', 'raneem_stgcn_visibility', 'raneem_stgcn_fullbody', 'raneem_lstm']


def select_native_pose(result):
    """Raneem's native highest-confidence actor, conf >0.25, zeros if absent.

    The underlying YOLO pass is shared at conf=0.10; filtering at 0.25 keeps the
    native candidates while letting Souad use its original lower detector gate.
    """
    if result.boxes is None or result.keypoints is None or not len(result.boxes):
        return np.zeros((17, 3), np.float32)
    confidence = result.boxes.conf.cpu().numpy()
    i = int(confidence.argmax())
    if confidence[i] <= .25:
        return np.zeros((17, 3), np.float32)
    return result.keypoints.data[i].cpu().numpy().copy()


def inventory(model):
    checkpoint = ROOT / 'model2_fall_detection/models/stgcnpp_fall_v1_laptop.pt'
    adapters, details = {}, {}
    details['souad_integrated'] = dict(available=model is not None, model_error=None if model is not None else 'Souad initialization failed',
        decision_rule='Actual integrated fall events at 0.70; existing confirmation, handover and alarm hold retained')
    if model is not None:
        shared = SouadAdapter(model.detector.core.predict, model.detector.settings, checkpoint)
        adapters[shared.model_id] = shared
        details[shared.model_id] = shared.metadata
        details['souad_integrated'].update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
    else:
        details['souad_shared_clips'] = dict(available=False, model_error='Souad initialization failed')
    for kind in ('stgcn_visibility', 'stgcn_fullbody', 'lstm'):
        key = 'raneem_' + kind
        try:
            adapter = RaneemAdapter(kind)
            adapters[key], details[key] = adapter, adapter.metadata
        except Exception as error:
            details[key] = dict(available=False, model_error=f'{type(error).__name__}: {error}')
    return adapters, details


def empty_row(model_id, details):
    return dict(model=model_id, available=details['available'], status='unavailable',
        prediction=None, max_fall_probability=None, total_windows=0, scored_windows=0,
        abstained_windows=0, positive_windows=0, fall_events=None, first_detection_seconds=None,
        error=details.get('model_error'), disagreements=[], input_sha256=None)


def aggregate(scores, labels, ends):
    decided = labels >= 0
    positive = labels == 1
    return dict(status='ok' if decided.any() else 'abstained',
                prediction=('fall' if positive.any() else 'non_fall') if decided.any() else None,
                max_fall_probability=float(np.nanmax(scores)) if decided.any() else None,
                total_windows=len(labels), scored_windows=int(decided.sum()),
                abstained_windows=int((~decided).sum()), positive_windows=int(positive.sum()),
                first_detection_seconds=float(ends[np.flatnonzero(positive)[0]] / FPS) if positive.any() else None)


def evaluate_frames(frames, detector, pose_model, integrated, adapters, details, cache_path):
    """No class/sequence labels are accepted by the inference function."""
    records = {key: empty_row(key, details[key]) for key in MODEL_IDS}
    if integrated is not None:
        integrated.reset(camera_id='comparison')
        reset_tracking(detector)
    pixels, source_shape, baseline_scores, events = [], None, [], []
    try:
        if not frames:
            raise ValueError('Empty sequence')
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Unreadable image: {path.name}')
            if source_shape is not None and frame.shape != source_shape:
                raise ValueError('Inconsistent frame dimensions')
            source_shape = frame.shape
            result = pose_model.predict(frame, conf=.10, imgsz=640, verbose=False)[0]
            pixels.append(select_native_pose(result))
            if integrated is not None and records['souad_integrated']['error'] is None:
                try:
                    if result.boxes is None or not len(result.boxes):
                        pose_result = (np.zeros((0,4)), np.zeros((0,17,2)), np.zeros((0,17)), np.zeros(0))
                    else:
                        pose_result = (result.boxes.xyxy.cpu().numpy(), result.keypoints.xy.cpu().numpy(),
                                       result.keypoints.conf.cpu().numpy(), result.boxes.conf.cpu().numpy())
                    # Inject the same actual pose result; no second pose model is run.
                    integrated.detector._poses = lambda frame, values=pose_result: values
                    out = integrated.process(frame, index/FPS, extract_tracks(detector, frame))
                    baseline_scores.extend(s['fall_probability'] for s in out['tracks'].values() if s['fall_probability'] is not None)
                    events.extend(out['events'])
                except Exception as error:
                    records['souad_integrated'].update(status='error', error=f'{type(error).__name__}: {error}')
        raw = np.asarray(pixels, np.float32)
        h, w = source_shape[:2]
        starts = np.arange(0, len(raw)-30+1, STRIDE, dtype=int)
        clips = np.stack([raw[i:i+30] for i in starts]) if len(starts) else np.empty((0,30,17,3),np.float32)
        np.savez_compressed(cache_path, pixel_poses=raw, width=w, height=h, starts=starts,
                            filenames=np.array([p.name for p in frames]), assumed_source_fps=FPS)
        input_hash = digest(cache_path)
        for key, adapter in adapters.items():
            row = records[key]
            row['input_sha256'] = input_hash
            try:
                scores, labels = adapter.predict(clips, w, h)
                row.update(aggregate(scores, labels, starts+29))
                # Scores are explicitly P(fall), not confidence of the winning class.
                row['window_predictions'] = [dict(start_frame=int(start), end_frame=int(start+29),
                    fall_probability=float(p) if np.isfinite(p) else None,
                    prediction=None if label < 0 else ('fall' if label == 1 else 'non_fall'))
                    for start,p,label in zip(starts,scores,labels)]
            except Exception as error:
                row.update(status='error', error=f'{type(error).__name__}: {error}', prediction=None)
        row = records['souad_integrated']
        if row['available'] and row['error'] is None:
            row.update(status='ok' if baseline_scores else 'abstained',
                       prediction=('fall' if events else 'non_fall') if baseline_scores else None,
                       max_fall_probability=max(baseline_scores) if baseline_scores else None,
                       scored_windows=len(baseline_scores), total_windows=None, abstained_windows=None,
                       positive_windows=None, fall_events=len(events),
                       first_detection_seconds=events[0]['time'] if events else None)
    except Exception as error:
        for row in records.values():
            if row['available']:
                row.update(status='error', error=f'{type(error).__name__}: {error}', prediction=None)
    return list(records.values())


def metrics(rows):
    valid = [r for r in rows if r['status']=='ok' and r['prediction'] is not None]
    tp = sum(r['truth']=='fall' and r['prediction']=='fall' for r in valid)
    fn = sum(r['truth']=='fall' and r['prediction']=='non_fall' for r in valid)
    fp = sum(r['truth']=='non_fall' and r['prediction']=='fall' for r in valid)
    tn = sum(r['truth']=='non_fall' and r['prediction']=='non_fall' for r in valid)
    def ratio(n,d): return n/d if d else None
    return dict(TP=tp,FP=fp,TN=tn,FN=fn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
        f1=ratio(2*tp,2*tp+fp+fn), specificity=ratio(tn,tn+fp), false_positive_rate=ratio(fp,fp+tn),
        false_alarm_sequences=fp, false_positive_windows=sum(r['positive_windows'] or 0 for r in valid if r['truth']=='non_fall'),
        operational_false_events=sum(r['fall_events'] or 0 for r in valid if r['truth']=='non_fall'),
        evaluated=len(valid),abstained=sum(r['status']=='abstained' for r in rows),
        errors=sum(r['status']=='error' for r in rows),unavailable=sum(r['status']=='unavailable' for r in rows),
        coverage=ratio(len(valid),len(rows)))


def provenance(items):
    ids = np.load(RANEEM/'results/training_data/event_ids.npy', allow_pickle=False)
    event = lambda name: re.sub(r'-cam\d+-rgb$', '', name)
    current = {event(folder.name) for _,folder,_ in items}
    original = {event(str(i)) for i in ids}
    notebook = ROOT.parent/'Souad_Work/model2_fall_detection/training/Themam_STGCN_Clean.ipynb'
    nb = json.loads(notebook.read_text())
    outputs = '\n'.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]))
    return dict(status='provisional', raneem_available_event_ids=sorted(original),
        overlap_with_available_raneem_data=sorted(current & original),
        souad_notebook=str(notebook), souad_notebook_sha256=digest(notebook),
        souad_saved_split_check='train / test: None' in outputs,
        limitations=[
            'Available Raneem event_ids and preparation scripts show no overlap with these test action identities, but checkpoints lack signed/version-linked split manifests.',
            'Souad notebook contains saved split checks and train/validation lists with no apparent test-action overlap; this does not bind the exported checkpoint to a verified dataset manifest.',
            'Original recording FPS remains unverified. Notebook fallback mentions 30 FPS; prior evaluation assumed 15 FPS. This comparison retains 15 provisionally for baseline comparability.',
            'No claim of an independently verified held-out test set; no fall-onset or latency ground truth is supplied.',
        ])


def save(rows, details, metadata, output):
    simple = [{k:v for k,v in row.items() if k!='window_predictions'} for row in rows]
    for row in simple:
        row['disagreements'] = [r['model'] for r in simple if r['sequence']==row['sequence'] and r['model']!=row['model']
                               and row['prediction'] is not None and r['prediction'] is not None and row['prediction']!=r['prediction']]
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(simple[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in simple)
    by_model={key:dict(overall=metrics([r for r in rows if r['model']==key]),
        by_camera={c:metrics([r for r in rows if r['model']==key and r['camera']==c]) for c in ('cam0','cam1')}) for key in MODEL_IDS}
    disagreements=[]
    for reference in ('souad_integrated','souad_shared_clips'):
        for model in MODEL_IDS[2:]:
            paired=[]
            for baseline in rows:
                if baseline['model']!=reference: continue
                other=next(r for r in rows if r['sequence']==baseline['sequence'] and r['model']==model)
                if baseline['prediction'] is None or other['prediction'] is None: continue
                if baseline['prediction']!=other['prediction']:
                    paired.append(dict(sequence=baseline['sequence'],camera=baseline['camera'],truth=baseline['truth'],
                        reference_prediction=baseline['prediction'],model_prediction=other['prediction']))
            disagreements.append(dict(reference=reference,model=model,cases=paired))
    summary=dict(metadata,models=details,results=by_model,disagreements=disagreements,
                 completed_sequences=len({r['sequence'] for r in rows}))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (output/'window_predictions.json').write_text(json.dumps([{k:r[k] for k in ('sequence','model','window_predictions') if k in r} for r in rows],indent=2,allow_nan=False))
    write_report(summary, output)
    draw_matrices(summary, output)
    return summary


def write_report(summary, output):
    lines=['# Provisional classifier comparison','',
        f"Completed {summary['completed_sequences']} / {summary['selected_sequences']} selected sequences. Source FPS assumed 15; not verified.",'',
        '## Protocol and original-model fidelity','',
        'All classifiers use the same PNG sequences. A single shared YOLOv8n-pose pass (640 px, confidence 0.10) supplies real poses. Pose weight SHA256 matches Raneem’s weight file. Shared clips select the highest-confidence person above 0.25 per frame, exactly as Raneem’s preparation script; missing detections are zero frames. This original top-person policy is not identity-safe in crowds.',
        'Shared classifiers consume identical 30 consecutive source-frame windows, stride 10, from Raneem’s training preparation script. At the provisional 15 FPS this also matches Souad’s 30-frame/15 FPS sampling. Raneem keeps x/width,y/height normalization; Souad keeps its own original per-clip normalization. No artificial occlusion is applied at evaluation.',
        'Raneem ST-GCN models keep their documented >=15 reliable frames gate; LSTM keeps the visibility_aware_model.py gate. No additional calibrated probability threshold exists in their inspected code: original argmax is retained (ties select non_fall). P(fall), not winning-class confidence, is reported.',
        'Recording aggregation for shared classifiers is an explicitly experimental rule: any positive decided window => fall; at least one decided window with none positive => non_fall; all abstentions => unavailable decision, never non_fall. A recording-level alert hold/debounce policy is not documented for Raneem and is not invented here. Her LSTM demo uses stride 1; this controlled comparison uses the documented training stride 10 for all shared classifiers.',
        'souad_integrated is the separate current operational baseline: YOLO person tracking, current handover safeguards, original temporal sampling, confirmation and actual fall events at 0.70. souad_shared_clips removes tracking/confirmation/alarm hold to compare classifier responses to identical raw clips. These are distinct experiments; do not attribute their difference solely to classifier weights.',
        'Class mappings: 0=non_fall, 1=fall for all adapters, verified against Raneem preparation code/LSTM checkpoint metadata and Souad’s notebook export softmax[:,1]. Architecture definitions are extracted without executing training scripts; state_dict loading is strict. Raneem’s class name STGCNPlusPlus does not make its weights interchangeable with Souad’s MMAction2 model.', '',
        '## Availability','', '| Model | Available | Input | Decision rule / error |','|---|---|---|---|']
    for key,d in summary['models'].items():
        lines.append(f"| {key} | {d['available']} | {d.get('input_shape','operational pipeline')} | {d.get('model_error') or d.get('decision_rule')} |")
    for group in ('overall','cam0','cam1'):
        lines+=['',f'## {group}','', '| Model | Evaluated | Abstained | Errors | TP | FP | TN | FN | Precision | Recall | F1 |','|---|---|---|---|---|---|---|---|---|---|---|---|']
        for key,result in summary['results'].items():
            m=result['overall'] if group=='overall' else result['by_camera'][group]
            vals=[str(m[k]) for k in ('evaluated','abstained','errors','TP','FP','TN','FN')]
            vals+=['N/A' if m[k] is None else f'{m[k]:.2%}' for k in ('precision','recall','f1')]
            lines.append('| '+' | '.join([key]+vals)+' |')
    lines+=['','## Disagreements','', '| Reference | Model | Sequence | Truth | Reference predicts | Model predicts |','|---|---|---|---|---|---|']
    for pair in summary['disagreements']:
        for case in pair['cases']:
            lines.append('| '+' | '.join([pair['reference'],pair['model'],case['sequence'],case['truth'],case['reference_prediction'],case['model_prediction']])+' |')
    lines+=['','## Recovered falls and additional misses','']
    for pair in summary['disagreements']:
        recovered=[c['sequence'] for c in pair['cases'] if c['truth']=='fall' and c['model_prediction']=='fall']
        missed=[c['sequence'] for c in pair['cases'] if c['truth']=='fall' and c['reference_prediction']=='fall']
        false=[c['sequence'] for c in pair['cases'] if c['truth']=='non_fall' and c['model_prediction']=='fall']
        lines.append(f"- {pair['model']} vs {pair['reference']}: recovered falls: {', '.join(recovered) or 'none'}; falls caught only by the reference: {', '.join(missed) or 'none'}; additional false-alarm recordings: {', '.join(false) or 'none'}.")
    lines+=['','## Interpretation and limits','',
        'A Raneem-only fall prediction on a fall recording is a potential recovered miss; the same disagreement on a non-fall recording is a false alarm. Read both directions and the camera-specific FP counts before judging robustness. Softmax scores across different networks are not calibrated or directly comparable.',
        'False alarms are counted primarily as non-fall recordings predicted fall (FP). Positive-window counts are saved separately, not called independent events. Only the operational baseline has original fall-event counts. Rates exclude unavailable/error/all-abstained rows and explicitly report decision coverage.',
        'There are five cam1 fall recordings and no cam1 non-fall recordings: overhead specificity is unmeasured. Both views of an action are correlated. The visibility-augmented ST-GCN checkpoint reports epoch 2 and heavily weighted fall loss; these metadata do not establish viewpoint robustness.',
        *summary['provenance']['limitations'],
        'Successfully reproduced comparisons and any failed adapters are listed above and in summary.json. Training-time augmentation, retraining, calibration, MMAction2 training, full Colab actor-selection/ReID/interpolation, and classifier fusion were not executed. Raw pose tensors, window predictions, file hashes and availability errors are retained for audit.','']
    (output/'comparison_report.md').write_text('\n'.join(lines))


def draw_matrices(summary, output):
    canvas=np.full((3*230,2*520,3),255,np.uint8)
    for index,key in enumerate(MODEL_IDS):
        x=(index%2)*520;y=(index//2)*230;m=summary['results'][key]['overall']
        cv2.putText(canvas,key,(x+12,y+24),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(canvas,'Columns: predicted fall / non-fall',(x+12,y+48),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
        for j,(name,count) in enumerate([('TP',m['TP']),('FN',m['FN']),('FP',m['FP']),('TN',m['TN'])]):
            xx=x+145+(j%2)*170;yy=y+65+(j//2)*62
            cv2.rectangle(canvas,(xx,yy),(xx+160,yy+55),(220,240,220) if name in ('TP','TN') else (220,220,250),-1)
            cv2.putText(canvas,f'{name}: {count}',(xx+25,yy+35),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
        cv2.putText(canvas,'Actual fall',(x+10,y+100),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
        cv2.putText(canvas,'Actual non-fall',(x+10,y+160),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
        cv2.putText(canvas,f"N={m['evaluated']} abstained={m['abstained']} errors={m['errors']} unavailable={m['unavailable']}",(x+12,y+215),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    if not cv2.imwrite(str(output/'confusion_matrices.png'),canvas): raise RuntimeError('Cannot write matrix image')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT.parent/'test')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1: parser.error('--limit must be positive')
    items=discover(args.dataset)
    if not items: parser.error('No sequences found')
    all_items=items;items=items[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    cache=args.output/'input_cache';cache.mkdir(exist_ok=True)
    # Fingerprint read-only teammate source/weights and existing baseline artifacts.
    protected=[p for base in (RANEEM, ROOT.parent/'Souad_Work') for p in base.rglob('*')
        if p.is_file() and p.suffix in ('.py','.pth','.pt','.ipynb','.json','.npy')]
    protected += [ROOT/name for name in ('integrated_model2.py','test_video.py','batch_evaluate.py','model2_fall_detection/fall_core.py','model2_fall_detection/fall_module.py','model2_fall_detection/tracking.py','visibility/visibility_detection.py')]
    protected += [ROOT/'evaluation'/name for name in ('batch_results.csv','summary.json','tracking_comparison.md')]
    hashes={str(p):digest(p) for p in protected}
    try:
        integrated=IntegratedModel2(use_fast=False)
        souad_error=None
    except Exception as error:
        integrated=None;souad_error=f'{type(error).__name__}: {error}'
    adapters,details=inventory(integrated)
    if souad_error:
        for key in ('souad_integrated','souad_shared_clips'):details[key]['model_error']=souad_error
    if digest(ROOT/'yolov8n-pose.pt') != digest(RANEEM/'yolov8n-pose.pt'):
        raise RuntimeError('Pose checkpoints differ; cannot silently substitute')
    detector=YOLO(str(ROOT/'yolov8n.pt')) if integrated is not None else None
    pose=YOLO(str(ROOT/'yolov8n-pose.pt'))
    metadata=dict(status='provisional',assumed_fps=FPS,window_frames=30,stride_frames=STRIDE,
        selected_sequences=len(items),discovered_sequences=len(all_items),provenance=provenance(all_items),
        protected_file_sha256=hashes,original_files_unchanged=None)
    rows=[]
    for i,(truth,folder,frames) in enumerate(items,1):
        print(f'[{i}/{len(items)}] {folder.name} ({len(frames)} frames)',flush=True)
        start=time.perf_counter()
        predictions=evaluate_frames(frames,detector,pose,integrated,adapters,details,cache/(folder.name+'.npz'))
        elapsed=time.perf_counter()-start
        camera=re.search(r'cam[01]',folder.name)
        # Ground truth is attached ONLY AFTER every model has produced predictions.
        for row in predictions:
            row.update(sequence=folder.name,camera=camera.group() if camera else 'unknown',truth=truth,
                frames=len(frames),processing_seconds=elapsed)
            print(f"  {row['model']}: {row['status']} {row['prediction']} maxP={row['max_fall_probability']} error={row['error']}",flush=True)
        rows.extend(predictions)
        save(rows,details,metadata,args.output)
    changed=[p for p,h in hashes.items() if digest(p)!=h]
    metadata['original_files_unchanged']=not changed
    metadata['changed_protected_files']=changed
    summary=save(rows,details,metadata,args.output)
    print(json.dumps(summary['results'],indent=2),flush=True)
    if changed:raise RuntimeError(f'Protected files changed: {changed}')
    if any(r['status']=='error' for r in rows):raise SystemExit('Sequence/model errors recorded; inspect reports before full evaluation')


if __name__=='__main__':main()
