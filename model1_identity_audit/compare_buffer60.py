"""Verify frozen CV artifacts, prepare new crops without inference, compare identity windows."""
import argparse
from collections import Counter, defaultdict
import configparser
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import audit
from audit import PROJECT, digest, read_csv, write_csv, write_json
from model1_cross_validation import run_sequence_cv as cv
from run_model1_test import validate_frozen

CV = PROJECT/'model1_cross_validation/runs/cv_xv920xpl'
FROZEN = PROJECT/'model1_cross_validation/final_model1_config.json'
OLD = PROJECT/'model1_identity_audit/run_gbg5837h'


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError('Provenance mismatch: '+label)


def require_prediction(path):
    if not path.is_file():
        raise FileNotFoundError('Required buffer60 prediction missing; no buffer30 substitution: '+str(path))
    return path


def verify_artifacts():
    frozen = json.loads(FROZEN.read_text())
    validate_frozen(frozen)
    plan = json.loads((CV/'plan.json').read_text())
    require_equal(digest(PROJECT/frozen['detector']['weights']), frozen['detector']['checkpoint_sha256'], 'checkpoint')
    require_equal(digest(PROJECT/'model1_cross_validation/run_sequence_cv.py'), plan['runner_sha256'], 'CV runner')
    require_equal(digest(PROJECT/'model1_cross_validation/evaluation/mot20_protocol.py'), plan['evaluator_sha256'], 'evaluator')
    require_equal(plan['prediction_origin'], frozen['coordinate_convention'], 'coordinates')
    params = {k: frozen['tracker'][k] for k in cv.CANDIDATES['buffer60']}
    require_equal(plan['candidates']['buffer60'], params, 'tracker parameters')
    records = {}
    for fold, name in enumerate(cv.NAMES, 1):
        selection_path = CV/f'fold_{fold}/selection.json'
        selection = json.loads(selection_path.read_text())
        require_equal(selection['selected'], 'buffer60', name+' selection')
        require_equal(selection['held_out'], name, name+' fold')
        require_equal(selection['parameters'], params, name+' tracker')
        root = Path(json.loads((PROJECT/audit.RUNS[name]/'config.json').read_text())['sequence_dir'])
        ini = configparser.ConfigParser()
        with (root/'seqinfo.ini').open() as f:
            ini.read_file(f)
        require_equal(ini.get('Sequence','name'), name, name+' source')
        total, fps = ini.getint('Sequence','seqLength'), ini.getfloat('Sequence','frameRate')
        signature = plan['cache_signatures'][name]
        for field, value in dict(sequence=name,frames=total,fps=fps,
                weights_sha256=frozen['detector']['checkpoint_sha256'],image_size=frozen['detector']['image_size'],
                confidence=frozen['detector']['confidence_threshold'],nms_iou=frozen['detector']['nms_iou_threshold'],
                person_class=frozen['detector']['person_class']).items():
            require_equal(signature[field], value, name+' '+field)
        for package, version in signature['versions'].items():
            require_equal(importlib.metadata.version(package), version, package+' version')
        for path, checksum in signature['detector_sources'].items():
            require_equal(digest(PROJECT/path), checksum, path)
        frames = sorted((int(p.stem),p) for p in (root/'img1').glob('*.jpg'))
        require_equal([n for n,_ in frames], list(range(1,total+1)), name+' complete frame inventory')
        import hashlib
        inventory = hashlib.sha256(json.dumps([(n,str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
                                              for n,p in frames]).encode()).hexdigest()
        require_equal(inventory, signature['source_inventory'], name+' original frame provenance')
        require_equal(digest(root/'gt/gt.txt'), plan['inputs'][name]['gt_sha256'], name+' GT')
        cache = cv.validate_cache(CV/'cache'/name, signature)
        pred = require_prediction(CV/f'fold_{fold}/held_out_{name}.csv')
        data = audit.load_predictions(pred, total, offset=0)
        require_equal(sorted(data.frame.unique()), list(range(1,total+1)), name+' prediction timeline')
        copies = sorted(CV.glob(f'fold_*/development_{name}_buffer60.csv'))
        require_equal(len(copies), 3, name+' development copies')
        for copy in copies:
            require_equal(digest(copy), digest(pred), name+' duplicate CV tracking output')
        old_prov = json.loads((OLD/name/'provenance.json').read_text())
        for path, checksum in old_prov['file_sha256'].items():
            require_equal(digest(path), checksum, 'archival input '+path)
        sampling = json.loads((PROJECT/audit.RUNS[name]/'config.json').read_text())
        for key,value in dict(target_fps=10,sequence_length=30,stride=30,padding=.1,max_gap_seconds=1/fps).items():
            require_equal(sampling[key], value, name+' sampling '+key)
        protected = [FROZEN, CV/'plan.json', selection_path, pred, root/'gt/gt.txt', root/'seqinfo.ini',
                     CV/f'fold_{fold}/held_out_metrics.csv', CV/'cache'/name/'metadata.json', cache,
                     PROJECT/'prepare_person_sequences.py', OLD/name/'observations.csv',
                     OLD/name/'windows.csv', OLD/name/'summary.json', OLD/name/'provenance.json'] + copies
        records[name] = dict(sequence=name, fold=fold, source=str(root), prediction=str(pred),
                             cache=str(cache), rows=len(data), frames=total, fps=fps,
                             parameters=params, sampling={k:sampling[k] for k in
                                 ['target_fps','sequence_length','stride','padding','max_gap_seconds']},
                             protected_sha256={str(p):digest(p) for p in protected},
                             source_inventory=inventory, checkpoint_sha256=signature['weights_sha256'])
        print(name, 'verified:', len(data), 'rows, complete cache and original frames', flush=True)
    return records


def window_signatures(observations):
    """Pair identical ordered frame/box/confidence observations, not tracker-local IDs or GT outcomes."""
    groups = defaultdict(list)
    for o in observations:
        groups[o['sequence_id']].append(o)
    signatures = defaultdict(list)
    for sid, entries in groups.items():
        entries.sort(key=lambda o:int(o['sample_index']))
        if len(entries) != 30 or [int(o['sample_index']) for o in entries] != list(range(30)):
            raise ValueError('Invalid paired window inventory')
        signature = tuple((int(o['frame']), *(float(o[k]) for k in ['x1','y1','x2','y2','confidence'])) for o in entries)
        signatures[signature].append(sid)
    return signatures


def pair_windows(old_obs, new_obs, old_windows, new_windows):
    old, new = window_signatures(old_obs), window_signatures(new_obs)
    a = {w['sequence_id']:w for w in old_windows}
    b = {w['sequence_id']:w for w in new_windows}
    pairs = []
    for signature in sorted(old.keys() & new.keys()):
        # Do not arbitrarily pair duplicate geometry windows.
        if len(old[signature]) == len(new[signature]) == 1:
            x, y = old[signature][0], new[signature][0]
            pairs.append(dict(buffer30_sequence_id=x,buffer60_sequence_id=y,
                              buffer30_classification=a[x]['classification'],buffer60_classification=b[y]['classification']))
    matrix = Counter((p['buffer30_classification'],p['buffer60_classification']) for p in pairs)
    return pairs, dict(exact_pairs=len(pairs),buffer30_unpaired=len(a)-len(pairs),buffer60_unpaired=len(b)-len(pairs),
                      outcomes=[dict(buffer30=x,buffer60=y,count=n) for (x,y),n in sorted(matrix.items())],
                      definition='Exact ordered source frames, full-body boxes and confidence; Track IDs and GT outcomes are not pairing keys.')


def metric_provider(records):
    def lookup(name, pred_hash, gt_hash):
        record = records[name]
        require_equal(pred_hash, record['protected_sha256'][record['prediction']], 'new crop prediction')
        gt_path = str(Path(record['source'])/'gt/gt.txt')
        require_equal(gt_hash, record['protected_sha256'][gt_path], 'new crop GT')
        path = CV/f"fold_{record['fold']}/held_out_metrics.csv"
        rows = read_csv(path)
        require_equal(len(rows), 1, 'held-out metrics row count')
        require_equal((rows[0]['sequence'],rows[0]['selected']), (name,'buffer60'), 'held-out metrics identity')
        return [dict(metadata=str(CV/'plan.json'),metrics=str(path),metrics_sha256=digest(path),
                     idf1=float(rows[0]['idf1']),mota=float(rows[0]['mota']),id_switches=int(float(rows[0]['num_switches'])))]
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    records = verify_artifacts()
    if not args.execute:
        print('Read-only checks passed. No crops or outputs written.'); return
    out = Path(tempfile.mkdtemp(prefix='buffer60_', dir=Path(__file__).resolve().parent))
    print('New comparison folder:',out,flush=True)
    write_json(out/'artifact_provenance.json',records)
    write_json(out/'method.json',dict(
        protocol='Unchanged audit.match_frame and audit.classify; as-stored coordinates, IoU >= .5, all frame predictions compete.',
        code_sha256={str(p):digest(p) for p in [Path(__file__),Path(audit.__file__),PROJECT/'prepare_person_sequences.py']},
        command=sys.argv,python=sys.version,scope='Frozen buffer60 training predictions versus separately identified archival buffer30 windows',
        limitations='Diagnostic association evidence; not official accuracy. Retrospectively exposed training data. '
                    'Frame inventory uses recorded paths, sizes and mtimes, not original image content hashes.'))
    lookup = metric_provider(records)
    summaries = []
    try:
        for name, record in records.items():
            # Deterministically replay tracker ONLY from verified detection cache.
            replay = out/f'{name}_replayed_buffer60.csv'
            print(name,'replaying ByteTrack from cache (no YOLO)',flush=True)
            cv.track(Path(record['cache']), {'fps':record['fps']}, record['parameters'], replay)
            require_equal(digest(replay), record['protected_sha256'][record['prediction']], name+' cache-to-prediction replay')
            command = [sys.executable,'-B',str(PROJECT/'prepare_person_sequences.py'),
                       '--sequence-name',name,'--sequence-dir',record['source'],'--tracking-csv',record['prediction'],
                       '--output-root',str(out/'prepared'),'--target-fps','10','--sequence-length','30',
                       '--stride','30','--padding','0.1','--max-gap-seconds',str(1/record['fps'])]
            print(name,'generating RGB crops with unchanged prepare_person_sequences.py',flush=True)
            with (out/f'{name}_preparation.log').open('x') as log:
                subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            generated = list((out/'prepared'/name).glob('run_*'))
            require_equal(len(generated), 1, 'new prepared run count')
            summary = audit.evaluate_run(name,generated[0],out/name,2,metric_lookup=lookup,
                                         scope='Verified frozen buffer60 CV held-out predictions; cache replay byte-identical.')
            # Validate new crop run's provider and detailed CV provenance remain separately identifiable.
            old_summary = json.loads((OLD/name/'summary.json').read_text())
            for variant, s in [('buffer30',old_summary),('buffer60',summary)]:
                summaries.append(dict(variant=variant,**s,inconclusive_fraction=s['inconclusive']/s['windows']))
            pairs, paired_summary = pair_windows(read_csv(OLD/name/'observations.csv'),read_csv(out/name/'observations.csv'),
                                                read_csv(OLD/name/'windows.csv'),read_csv(out/name/'windows.csv'))
            write_csv(out/name/'paired_windows.csv',pairs)
            write_json(out/name/'paired_summary.json',paired_summary)
            for path, checksum in record['protected_sha256'].items():
                require_equal(digest(path),checksum,'unchanged input '+path)
            write_csv(out/f'{name}_comparison.csv',summaries[-2:])
            print(name,'complete:',json.dumps(summary),flush=True)
        write_csv(out/'comparison.csv',summaries)
        write_json(out/'completion.json',dict(status='complete',sequences=list(records),inference_performed=False))
    except Exception as exc:
        write_json(out/'failure.json',dict(error=str(exc),status='incomplete'))
        raise
    print('Completed:',out,flush=True)


if __name__ == '__main__':
    main()
