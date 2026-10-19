# Provisional batch evaluation

Completed 2 / 2 selected sequences. Assumed source FPS: 15.0.

Souad predictions use original alert events; Raneem predictions use original argmax on ready windows. No fusion. Raneem event counts are positive windows.
Checkpoint, normalization, 30-frame / 15 FPS model sampling and visibility classification are unchanged.

## Provenance and interpretation

PROVISIONAL: original recording FPS is unverified; --fps is an assumption, not proof of source timing.
Held-out status is unverified. model2_fall_detection/README.md describes 47 train/validation videos and 10 reserved cam0 test videos, but provides no sequence IDs or split manifest. Its referenced training directory/notebook is absent from this project.
No dataset metadata or onset annotations were found locally. Directory names alone do not prove exclusion from training, including alternate-camera views of training actions.
Required before final reporting: recover the checkpoint training split manifest, compare action identities across cameras, and verify source FPS and fall onset annotations.
Latency omitted without supplied ground-truth onset timestamps.
Souad uses its original alert events; Raneem uses class argmax on ready windows. Raneem fall_events counts positive windows, not alerts. No fusion is performed.

Errors are excluded from classification metrics; partial results remain in the CSV. Undefined ratios are N/A/null.
Visibility counts are person-frame observations, not independent samples. Grouping uses each sequence’s dominant category (ties: mixed; no detections: no_tracks). These groups are descriptive, not causal visibility-performance estimates.
Camera views of the same action are correlated. cam1 has no non-fall examples in the current dataset; its specificity and false-positive rate are undefined.
Processing FPS includes image reading, tracking, pose, classifier, and optional rendering/encoding. Model loading is excluded; first inference warm-up is included. These are observed wall-clock rates, not an isolated hardware benchmark; other machine activity can affect them.
Timestamps use sorted frame position / assumed FPS. Numbering gaps are not reconstructed. Latency is only recorded with supplied onset annotations; negative latency means the first alert preceded the annotated onset.

## Overall metrics

| Metric | Value |
|---|---|
| evaluated | 2 |
| errors | 0 |
| TP | 1 |
| FP | 1 |
| TN | 0 |
| FN | 0 |
| precision | 0.5000 |
| recall | 1.0000 |
| f1 | 0.6667 |
| specificity | 0.0000 |
| false_positive_rate | 1.0000 |
| false_alerts | 2 |
| false_alerts_per_non_fall_video | 2.0000 |
| average_processing_fps | 18.5573 |
| median_processing_fps | 18.5573 |

## Camera comparison

| Group | N | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | FPR | False alerts / non-fall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cam0 | 2 | 1 | 1 | 0 | 0 | 0.5000 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 2.0000 |
| cam1 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |

## Dominant visibility comparison

| Group | N | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | FPR | False alerts / non-fall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_body | 2 | 1 | 1 | 0 | 0 | 0.5000 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 2.0000 |

## Failure cases

| Sequence | Truth | Prediction | Max P | Events | Error |
|---|---|---|---|---|---|
| non_fall/adl-02-cam0-rgb | non_fall | fall | 0.9744908213615417 | 2 | N/A |

Inspect per-sequence diagnostics for missing skeletons, insufficient history, person confirmation, temporal gaps and tracking transfers. A ready classifier can still return a probability below threshold.
