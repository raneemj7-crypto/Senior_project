# Provisional batch evaluation

Completed 15 / 15 selected sequences. Assumed source FPS: 15.

Predictions use actual fall events at the unchanged 0.70 threshold, not maximum probabilities.
Checkpoint, normalization, 30-frame / 15 FPS model sampling and visibility classification are unchanged.

## Provenance and interpretation

PROVISIONAL: original recording FPS is unverified; --fps is an assumption, not proof of source timing.
Held-out status is unverified. model2_fall_detection/README.md describes 47 train/validation videos and 10 reserved cam0 test videos, but provides no sequence IDs or split manifest. Its referenced training directory/notebook is absent from this project.
No dataset metadata or onset annotations were found locally. Directory names alone do not prove exclusion from training, including alternate-camera views of training actions.
Required before final reporting: recover the checkpoint training split manifest, compare action identities across cameras, and verify source FPS and fall onset annotations.
Latency omitted without supplied ground-truth onset timestamps.

Errors are excluded from classification metrics; partial results remain in the CSV. Undefined ratios are N/A/null.
Visibility counts are person-frame observations, not independent samples. Grouping uses each sequence’s dominant category (ties: mixed; no detections: no_tracks). These groups are descriptive, not causal visibility-performance estimates.
Camera views of the same action are correlated. cam1 has no non-fall examples in the current dataset; its specificity and false-positive rate are undefined.
Processing FPS includes image reading, tracking, pose, classifier, and optional rendering/encoding. Model loading is excluded; first inference warm-up is included. These are observed wall-clock rates, not an isolated hardware benchmark; other machine activity can affect them.
Timestamps use sorted frame position / assumed FPS. Numbering gaps are not reconstructed. Latency is only recorded with supplied onset annotations; negative latency means the first alert preceded the annotated onset.

## Overall metrics

| Metric | Value |
|---|---|
| evaluated | 15 |
| errors | 0 |
| TP | 5 |
| FP | 0 |
| TN | 5 |
| FN | 5 |
| precision | 1.0000 |
| recall | 0.5000 |
| f1 | 0.6667 |
| specificity | 1.0000 |
| false_positive_rate | 0.0000 |
| false_alerts | 0 |
| false_alerts_per_non_fall_video | 0.0000 |
| average_processing_fps | 19.7916 |
| median_processing_fps | 19.4288 |

## Camera comparison

| Group | N | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | FPR | False alerts / non-fall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cam0 | 10 | 4 | 0 | 5 | 1 | 1.0000 | 0.8000 | 0.8889 | 1.0000 | 0.0000 | 0.0000 |
| cam1 | 5 | 1 | 0 | 0 | 4 | 1.0000 | 0.2000 | 0.3333 | N/A | N/A | N/A |

## Dominant visibility comparison

| Group | N | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | FPR | False alerts / non-fall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_body | 15 | 5 | 0 | 5 | 5 | 1.0000 | 0.5000 | 0.6667 | 1.0000 | 0.0000 | 0.0000 |

## Failure cases

| Sequence | Truth | Prediction | Max P | Events | Error |
|---|---|---|---|---|---|
| fall/fall-05-cam1-rgb | fall | non_fall | 0.172 | 0 | N/A |
| fall/fall-06-cam1-rgb | fall | non_fall | 0.0981 | 0 | N/A |
| fall/fall-10-cam0-rgb | fall | non_fall | 0.0753 | 0 | N/A |
| fall/fall-10-cam1-rgb | fall | non_fall | 0.2062 | 0 | N/A |
| fall/fall-16-cam1-rgb | fall | non_fall | 0.0237 | 0 | N/A |

Inspect per-sequence diagnostics for missing skeletons, insufficient history, person confirmation, temporal gaps and tracking transfers. A ready classifier can still return a probability below threshold.
