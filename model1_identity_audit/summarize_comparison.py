"""Read back completed comparison artifacts, independently verify totals, write a new report."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd
import audit
from compare_buffer60 import OLD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    root = args.run.resolve()
    assert json.loads((root/'completion.json').read_text())['status'] == 'complete'
    table = pd.read_csv(root/'comparison.csv')
    checks, pairs = [], []
    for name in table.sequence.unique():
        exact_observations = {}
        for variant, folder in [('buffer30', OLD/name), ('buffer60', root/name)]:
            obs = pd.read_csv(folder/'observations.csv')
            windows = pd.read_csv(folder/'windows.csv').set_index('sequence_id')
            counts = Counter()
            exact_observations[variant] = {}
            for sid, group in obs.groupby('sequence_id', sort=False):
                assert len(group) == 30
                assert list(group.sample_index) == list(range(30))
                assert group.track_id.nunique() == 1
                assert (group.frame.diff().dropna() > 0).all()
                identities = set(group.gt_id.dropna())
                classification = (audit.CONFLICT if len(identities) > 1 else
                                  audit.CONSISTENT if len(identities) == 1 and group.gt_id.notna().all() else
                                  audit.INCONCLUSIVE)
                assert windows.loc[sid,'classification'] == classification
                counts[classification] += 1
                exact_observations[variant][sid] = tuple(group[
                    ['frame','x1','y1','x2','y2','confidence']].itertuples(index=False,name=None))
            matched = obs.dropna(subset=['gt_id']).drop_duplicates(['frame','track_id'])
            assert not matched.duplicated(['frame','gt_id']).any()
            summary = table[(table.sequence == name) & (table.variant == variant)].iloc[0]
            assert (summary.windows, summary.sampled_observations, summary.matched_observations) == (
                len(windows), len(obs), int(obs.gt_id.notna().sum()))
            assert (summary.consistent,summary.conflicts,summary.inconclusive) == (
                counts[audit.CONSISTENT],counts[audit.CONFLICT],counts[audit.INCONCLUSIVE])
            checks.append(dict(sequence=name,variant=variant,windows=len(windows),status='passed'))
        paired = json.loads((root/name/'paired_summary.json').read_text())
        pair_path = root/name/'paired_windows.csv'
        pair_rows = audit.read_csv(pair_path) if pair_path.exists() else []
        assert len(pair_rows) == paired['exact_pairs']
        assert len({r['buffer30_sequence_id'] for r in pair_rows}) == len(pair_rows)
        assert len({r['buffer60_sequence_id'] for r in pair_rows}) == len(pair_rows)
        for row in pair_rows:
            assert exact_observations['buffer30'][row['buffer30_sequence_id']] == exact_observations['buffer60'][row['buffer60_sequence_id']]
        matrix = Counter((r['buffer30_classification'],r['buffer60_classification']) for r in pair_rows)
        assert matrix == Counter({(r['buffer30'],r['buffer60']):r['count'] for r in paired['outcomes']})
        pairs.append((name,paired))
    artifacts = json.loads((root/'artifact_provenance.json').read_text())
    for record in artifacts.values():
        for path, checksum in record['protected_sha256'].items():
            assert audit.digest(path) == checksum, 'Changed protected input: '+path
    method = json.loads((root/'method.json').read_text())
    for path, checksum in method['code_sha256'].items():
        assert audit.digest(path) == checksum, 'Changed evaluated code: '+path
    audit.write_json(root/'independent_verification.json',dict(status='passed',checks=checks,
                     protected_hashes_unchanged=True,code_hashes_match=True))
    lines = ['# Frozen buffer60 temporal identity comparison', '',
             'All four frozen buffer60 CV held-out predictions were available and reproduced byte-for-byte by '
             'ByteTrack replay from verified detection caches. No YOLO inference was performed.', '',
             '## Counts and rates', '',
             '| Dataset | Buffer | Windows | Consistent | Conflicts | Inconclusive | Matched observations | Coverage |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table.itertuples():
        lines.append(f'| {r.sequence} | {r.variant} | {r.windows:,} | {r.consistent:,} ({100*r.consistent/r.windows:.2f}%) '
                     f'| {r.conflicts:,} ({100*r.conflicts/r.windows:.2f}%) | {r.inconclusive:,} ({100*r.inconclusive/r.windows:.2f}%) '
                     f'| {r.matched_observations:,}/{r.sampled_observations:,} | {100*r.match_coverage:.2f}% |')
    for variant,g in table.groupby('variant'):
        n=int(g.windows.sum()); c=int(g.consistent.sum()); x=int(g.conflicts.sum()); u=int(g.inconclusive.sum())
        m=int(g.matched_observations.sum()); total=int(g.sampled_observations.sum())
        lines.append(f'| **Total** | **{variant}** | **{n:,}** | **{c:,} ({100*c/n:.2f}%)** | **{x:,} ({100*x/n:.2f}%)** '
                     f'| **{u:,} ({100*u/n:.2f}%)** | **{m:,}/{total:,}** | **{100*m/total:.2f}%** |')
    lines += ['', '## Exact paired windows', '',
              'Pairs require identical ordered source frame IDs, original full-body boxes and confidence values '
              'in all 30 observations. Tracker-local IDs and GT labels are not pairing keys. Duplicate signatures '
              'are not arbitrarily paired. The unpaired sets are reported rather than treated as failures.', '',
              '| Dataset | Exact pairs | Buffer30 unpaired | Buffer60 unpaired | Classification changed |',
              '|---|---:|---:|---:|---:|']
    for name,s in pairs:
        changed=sum(r['count'] for r in s['outcomes'] if r['buffer30'] != r['buffer60'])
        lines.append(f"| {name} | {s['exact_pairs']:,} | {s['buffer30_unpaired']:,} | {s['buffer60_unpaired']:,} | {changed:,} |")
    lines += ['', '| Dataset | Buffer30 classification | Buffer60 classification | Count |', '|---|---|---|---:|']
    for name,s in pairs:
        for r in s['outcomes']:
            lines.append(f"| {name} | {r['buffer30']} | {r['buffer60']} | {r['count']} |")
    lines += ['', '## Interpretation', '',
              'A window is consistent only if every sample matches the same eligible GT person. Two or more '
              'matched identities establish an observed conflict. Missing matches alone are inconclusive, not '
              'identity switches. This uses unchanged framewise one-to-one IoU >= 0.5 matching, the existing '
              'distractor preprocessing and as-stored coordinates. It is diagnostic association evidence, not '
              'official benchmark accuracy or visual proof of physical identity.', '',
              'All-window percentages have different denominators: changing track retention can change '
              'reconnections, segment boundaries and window starting points. The strict 0.04-second crop gap '
              'policy still splits interruptions even when ByteTrack retains an ID longer. New windows and '
              'lost windows can change aggregate rates while exact paired windows remain unchanged. '
              'These archival buffer30 predictions were not rerun from the CV caches, so this comparison '
              'is not a pure causal estimate of changing only lost_track_buffer.', '',
              'Neither variant guarantees that every emitted window contains one physical person. Findings '
              'apply to complete windows surviving the generator, not rejected tails, short tracks or unsampled '
              'frames. Training sequences were previously exposed; no independent generalization claim follows.', '',
              '## Historical whole-sequence metrics (separate measurement)', '',
              '| Dataset | Buffer | Diagnostic IDF1 | Diagnostic MOTA | ID switches |', '|---|---|---:|---:|---:|']
    for r in table.itertuples():
        lines.append(f'| {r.sequence} | {r.variant} | {100*r.historical_idf1:.2f}% | {100*r.historical_mota:.2f}% | {r.historical_id_switches:,} |')
    lines += ['', 'IDF1 measures global identity association, and MOTA combines misses, false positives and '
              'switches. Those metrics use whole source timelines and motmetrics temporal associations; '
              'this diagnostic evaluates selected sampled windows with framewise assignments. A switch may '
              'occur between two internally consistent windows. Metrics above were read from verified '
              'historical evidence, not recomputed or mixed across variants.', '',
              '## Artifacts and validation', '',
              '- `artifact_provenance.json`: frozen/CV settings, frame inventory, cache and protected file hashes.',
              '- `method.json`: implementation hashes and method; `completion.json`: successful completion.',
              '- `prepared/`: fresh crops and manifests from the unchanged preparation script, length 30, '
              '10 FPS, stride 30, padding 0.1, gap 1/source FPS.',
              '- `MOT20-*/observations.csv`: per-sample GT identity, match status and original crop metadata.',
              '- `MOT20-*/paired_windows.csv`: individual exact-pair outcomes.',
              '- `MOT20-*/contact_01.png` and `contact_02.png`: deterministic suspicious examples with frame '
              'and Track ID labels. Displayed crops were pixel-compared with original JPEG regions. '
              'Visual inspection is illustrative and is not used as GT or as a random accuracy estimate.',
              '- `independent_verification.json`: all window classifications, counts, sample order, one-to-one '
              'GT assignments, paired transition totals and protected/code hashes checked on output readback.', '',
              '19 identity/comparison unit tests passed, including provenance mismatches, missing predictions, '
              'different inventories and tracker-local ID pairing. See `../BUFFER60_README.md` for exact commands.', '',
              'The original-image inventory matches recorded paths, sizes and mtimes; the historical cache '
              'does not provide per-image content hashes. Full historical outputs and frozen settings remain unchanged.']
    with (root/'REPORT.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    print('Verified and reported:',root)


if __name__ == '__main__':
    main()
