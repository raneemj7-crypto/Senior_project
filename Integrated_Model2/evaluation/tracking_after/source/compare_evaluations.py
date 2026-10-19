"""Compare complete sequence evaluations without treating throughput as accuracy."""
import argparse
import csv
import json
from pathlib import Path


def compare(baseline, after, destination):
    before_summary = json.loads((baseline / 'summary.json').read_text())
    after_summary = json.loads((after / 'summary.json').read_text())
    def rows(path):
        with (path / 'batch_results.csv').open() as file:
            return {r['sequence_id']: r for r in csv.DictReader(file)}
    before, current = rows(baseline), rows(after)
    for summary, records in ((before_summary, before), (after_summary, current)):
        if len(records) != summary['selected_sequences'] or summary['completed_sequences'] != len(records):
            raise ValueError('Comparison requires completed evaluations, not partial progress reports')
    if set(before) != set(current):
        raise ValueError('Comparisons require exactly the same sequence IDs')
    for sequence in before:
        for key in ('ground_truth', 'number_of_frames', 'assumed_fps'):
            if before[sequence][key] != current[sequence][key]:
                raise ValueError(f'Non-comparable {key} for {sequence}')
    for key in ('checkpoint_sha256', 'model_settings', 'assumed_fps'):
        if before_summary[key] != after_summary[key]:
            raise ValueError(f'Non-comparable {key}')
    if any(r['error'] for r in list(before.values())+list(current.values())):
        raise ValueError('Resolve sequence processing errors before comparing complete runs')
    lines = ['# Tracking continuity: before and after', '',
             f"PROVISIONAL: same {len(before)}-sequence dataset, assumed source FPS {after_summary['assumed_fps']}; split provenance and original recording FPS are still unverified. No latency claims without onset annotations.", '',
             'Sequence predictions use generated fall events at threshold 0.70. Checkpoint, model settings, sequence IDs, frame counts and source FPS match between runs.', '',
             '| Group | Run | TP | FP | TN | FN | Recall | Precision | F1 |',
             '|---|---|---|---|---|---|---|---|---|']
    for group in ('overall', 'cam0', 'cam1'):
        for label, summary in [('Before', before_summary), ('After', after_summary)]:
            m = summary['overall'] if group == 'overall' else summary['by_camera'][group]
            values = [str(m[k]) for k in ('TP','FP','TN','FN')]
            values += ['N/A' if m[k] is None else f'{100*m[k]:.2f}%' for k in ('recall','precision','f1')]
            lines.append('| ' + ' | '.join([group,label]+values) + ' |')
    old, new = before_summary['overall'], after_summary['overall']
    improved = new['recall'] > old['recall']
    lines += ['', f"Recall {'increased' if improved else 'did not improve'}: {old['recall']:.1%} → {new['recall']:.1%}. False-positive sequences: {old['FP']} → {new['FP']}.", '',
              '## Per-sequence comparison', '',
              '| Sequence | Before | After | Max P before | Max P after | Transfers before / after |',
              '|---|---|---|---|---|---|']
    for sequence, row in current.items():
        old_row = before[sequence]
        lines.append(f"| {sequence} | {old_row['predicted_class']} | {row['predicted_class']} | {old_row['max_fall_probability']} | {row['max_fall_probability']} | {old_row['track_id_transfers']} / {row['track_id_transfers']} |")
    lines += ['', '## Missed-sequence evidence', '',
              'Counts below describe actual returned person-frame observations. Confidence is not anatomical correctness; available predictions do not prove the pose is accurate.', '',
              '| Sequence | No-track frames / total | Missing poses | Low-confidence poses | History resets | Prediction-ready observations | Max P |',
              '|---|---|---|---|---|---|---|']
    for sequence, row in current.items():
        if row['ground_truth'] != 'fall' or row['predicted_class'] == 'fall':
            continue
        d = json.loads(row['diagnostics'])
        lines.append(f"| {sequence} | {row['frames_without_tracks']} / {row['number_of_frames']} | {d.get('missing_skeleton',0)} | {d.get('low_confidence_joints',0)} | {d.get('history_reset',0)} | {d.get('ready',0)+d.get('prediction_cached',0)} | {row['max_fall_probability']} |")
    lines += ['', 'cam1 has no non-fall examples; its specificity and false-positive rate cannot be estimated. Camera views of the same action are correlated.',
              'See the JSONL traces for exact tracking interruptions, candidate rejection evidence, confidence changes, history resets and handovers. No fabricated joints or pose interpolation were supplied to the classifier.', '']
    destination.write_text('\n'.join(lines))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('evaluation/baseline_tracking'))
    parser.add_argument('--after', type=Path, default=Path('evaluation/tracking_after'))
    parser.add_argument('--output', type=Path, default=Path('evaluation/tracking_comparison.md'))
    args = parser.parse_args()
    print(compare(args.baseline, args.after, args.output))
