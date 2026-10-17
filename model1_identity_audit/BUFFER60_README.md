# Frozen buffer60 comparison

This entry point verifies the existing CV artifacts and compares newly prepared
buffer60 windows against the separately labeled archival buffer30 audit. It never
runs YOLO, trains a model, changes frozen settings, or overwrites historical runs.

From `Senior_project`:

```bash
../.venv/bin/python -B -m unittest discover -s model1_identity_audit -p 'test_*.py' -v
# Read-only provenance/cache checks:
../.venv/bin/python -B model1_identity_audit/compare_buffer60.py
# New crops, identity scoring, visual evidence and paired comparison:
../.venv/bin/python -B model1_identity_audit/compare_buffer60.py --execute
```

Execution creates a unique `model1_identity_audit/buffer60_*` directory. It uses
the held-out buffer60 prediction in fold 1/2/3/4 for MOT20-01/02/03/05 respectively.
There is no fallback to buffer30 or automatic inference when a file is missing.

## Verification

- Frozen checkpoint SHA-256, detector settings, coordinate convention and tracker
  parameters must match the recorded CV plan and each fold's selection.
- Current CV runner, evaluator, detector source hashes and recorded package
  versions must match. Original JPEG IDs must cover the full source sequence;
  their path/size/mtime inventory must match the recorded detector-cache inventory.
  This inventory is not a generation-time pixel-content hash.
- Source sequence name/FPS/length and GT checksum must match; cache signature,
  checksum and complete frame inventory are checked by the existing CV validator.
- Each held-out prediction must match all three development buffer60 copies and
  pass the diagnostic prediction validator. These particular runs have predictions
  in every frame, which is checked explicitly; this is not a universal tracker
  requirement for other datasets.
- Before crop generation, ByteTrack is replayed from the validated detection cache
  with the frozen settings and source FPS. Its CSV must be **byte-identical** to
  the chosen existing prediction. This is tracker-only work, not detector inference.
- The old audit input hashes are checked. Old observation/window/summary results
  and new inputs are hashed into `artifact_provenance.json`; protected hashes are
  checked again after each sequence completes.

## Preparation and scoring

The unchanged `prepare_person_sequences.py` is executed as a subprocess with
explicit source/prediction paths, target 10 FPS, length 30, stride 30, padding 0.1,
and maximum gap 1/source FPS. This writes fresh RGB crops and manifests. It takes
image-processing time and disk space even though it performs no detector inference.

`audit.py` accepts an optional metric-provenance lookup and scope label. Its original
default behavior remains buffer30. Both variants use the same unchanged
`match_frame` and `classify` functions and the same structural window checks.
Buffer60 metric references are its corresponding held-out CV metrics, verified
against the validated prediction/GT hashes; buffer30 metrics remain separate.

## Paired comparisons

Track IDs are local to a tracker run. Equal numeric Track IDs alone cannot pair
windows. We pair only unique, exact ordered sequences of
`(original_frame, x1, y1, x2, y2, confidence)` across the two variants. Pairing never
uses GT assignment or outcome classification, avoiding selection on the answer.
Duplicate signatures are left unpaired. Changes in Track ID or segment number do
not prevent an otherwise exact observation-level pair.

This conservative pairing excludes windows with even one changed box or shifted
sample grid. For exact pairs, competing predictions elsewhere in the frame may
still alter the one-to-one GT association. Rates on all generated windows have
different denominators whenever tracker lifetimes, reconnections, interruptions
or sampling start points change. Higher consistency percentage alone does not
establish tracker superiority. The historical buffer30 run is also not a new
controlled rerun from the CV detection caches; comparisons describe the actual
specified artifacts, not a pure causal estimate of changing one parameter.

## Files

- `artifact_provenance.json`, `method.json`, `completion.json`: verification,
  method, code hashes and completion state; `failure.json` on an interrupted failure.
- `prepared/MOT20-*/run_*`: generator outputs; original historical outputs unchanged.
- `MOT20-*/summary.json`, `windows.csv`, `observations.csv`: identity evidence.
- `MOT20-*/paired_windows.csv`, `paired_summary.json`: exact pairs, transition counts,
  and unpaired inventories (the CSV is absent when there are no pairs).
- `comparison.csv`: both variants' counts, rates, coverage and diagnostic metrics.
- `MOT20-*/contact_*.png`: two deterministic suspicious examples, conflict-first;
  original frames, Track IDs and GT assignments are labeled. Displayed crop pixels
  are verified against original JPEG regions. Visual inspection is illustrative,
  not ground truth and not a random accuracy estimate.

These remain diagnostic identity-association measurements on retrospectively
exposed MOT20 training data. They do not establish an official benchmark score,
physical-person certainty in ambiguous overlapping crops, or identity accuracy
for the unlabeled official-test windows. Unmatched observations are inconclusive
unless other matched observations already establish a multi-identity conflict.
