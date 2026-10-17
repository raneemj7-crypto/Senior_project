# Retrospective leave-one-MOT20-sequence-out protocol

Outer folds hold out MOT20-01, 02, 03 and 05 once each. All other listed sequences are development sequences. MOT20-04/06/07/08 are never read. Do not call sequence IDs independent scenes without checking camera/site provenance. All four development sequences have been explored previously: these are retrospective held-out folds, not pristine final tests.

## Frozen detector and predefined selection

The detector is never trained. Use the unchanged CrowdHuman checkpoint on original JPEGs, CPU, size 640, confidence .25, NMS .45 and person class 0. Candidates are baseline (.25 activation, 30 buffer, .8 matching), buffer60 (.25,60,.8) and match07 (.25,30,.7). Consecutive frames=1. Actual seqinfo FPS is supplied for every fresh tracker.

Select by unweighted mean IDF1 across the three development sequences. Report all per-sequence ID switches, recall and MOTA as supporting metrics, not post-hoc selection criteria. Exact ties prefer baseline, buffer60, then match07. No extra nested fitting is needed here because no detector/tracker model is learned; these three sequences are the parameter-selection set. If detector training is introduced, redesign with inner held-out selection. Candidate choices were informed by prior exploration; disclose this.

Selection code only accepts the fold's development rows. Save its selected parameters before held-out scoring. Only that configuration is tracked and scored on the held-out sequence for that fold. The same sequence necessarily serves as development in other folds; candidate information from those folds is not used by this fold's selection. Each held-out fold is scored once. No outer-result-driven changes or reruns for improvement.

## Detection caches

Tracking CSVs cannot reconstruct all detections (unconfirmed, unmatched and low-confidence detections may be absent). Legacy YOLO26 labels cannot stand in for YOLOv5. No existing raw YOLOv5 detection cache with sufficient provenance has been established.

Create JSONL caches containing every original frame ID, including empty lists, and raw post-NMS [x1,y1,x2,y2,confidence,class_id] detections. Do not cache tracks as detections. Cache once per sequence and reuse across configurations/folds. Reuse a previous runner cache via --cache-dir pointing to its cache folder; mismatches abort and old caches are never overwritten. Missing sequence caches are generated in the new run only.

Metadata records checkpoint hash, detector code hashes, versions, detector settings and source file path/size/mtime fingerprint. A checksum protects JSONL contents and complete frame numbering is checked. Source fingerprint is metadata-based, not a byte-level image integrity proof; invalidate caches if source content is changed with preserved file metadata. Current version changes conservatively invalidate reuse.

## Evaluation and outputs

Use evaluation/mot20_protocol.py without modifications, including its documented distractor preprocessing and diagnostic motmetrics status. Require explicit --prediction-origin as-stored or zero-to-one, established independently of held-out scores. This choice applies identically to all folds. Coordinate checks, original IDs, empty-frame updates and no head observations are retained.

Each new folder contains plan/provenance, caches, all development metrics/predictions, frozen selections and held-out metrics/predictions. Summary is macro mean and sample standard deviation (ddof=1) over four folds, not pooled MOT metrics. Identity switches are also reported as raw per-fold counts. Four scenes do not justify frame-wise confidence intervals. Each output includes protocol/plan-level detector and tracker identities and diagnostic status.

Planned work without reusable caches: 8,931 detector frames once; 36 development sequence/configuration evaluations and 4 outer evaluations; 89,310 tracker frame updates. Development tracking is intentionally repeated with fresh instances per fold. Runtime is not estimated without benchmarking. No full run is performed by default.

```bash
../.venv/bin/python model1_cross_validation/run_sequence_cv.py --prediction-origin as-stored
# Add --execute only when intentionally starting the full workload.
```

Input preflight inspects all four GT files for schema validity, not performance or parameter selection. It imports dependencies and tests empty tracker updates but never loads model weights in plan mode. All original scripts, results, weights and datasets remain read-only.
