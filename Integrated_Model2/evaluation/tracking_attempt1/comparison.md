# Tracking continuity: before and after

PROVISIONAL: same 15-sequence dataset, assumed source FPS 15; split provenance and original recording FPS are still unverified. No latency claims without onset annotations.

Sequence predictions use generated fall events at threshold 0.70. Checkpoint, model settings, sequence IDs, frame counts and source FPS match between runs.

| Group | Run | TP | FP | TN | FN | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| overall | Before | 5 | 0 | 5 | 5 | 50.00% | 100.00% | 66.67% |
| overall | After | 4 | 0 | 5 | 6 | 40.00% | 100.00% | 57.14% |
| cam0 | Before | 4 | 0 | 5 | 1 | 80.00% | 100.00% | 88.89% |
| cam0 | After | 3 | 0 | 5 | 2 | 60.00% | 100.00% | 75.00% |
| cam1 | Before | 1 | 0 | 0 | 4 | 20.00% | 100.00% | 33.33% |
| cam1 | After | 1 | 0 | 0 | 4 | 20.00% | 100.00% | 33.33% |

Recall did not improve: 50.0% → 40.0%. False-positive sequences: 0 → 0.

## Per-sequence comparison

| Sequence | Before | After | Max P before | Max P after | Transfers before / after |
|---|---|---|---|---|---|
| fall/fall-05-cam0-rgb | fall | fall | 0.9957 | 0.9957 | 1 / 1 |
| non_fall/adl-02-cam0-rgb | non_fall | non_fall | 0.003 | 0.003 | 0 / 0 |
| fall/fall-05-cam1-rgb | non_fall | non_fall | 0.172 | 0.172 | 0 / 0 |
| non_fall/adl-05-cam0-rgb | non_fall | non_fall | 0.0037 | 0.0037 | 0 / 0 |
| fall/fall-06-cam0-rgb | fall | fall | 0.9963 | 0.9963 | 0 / 0 |
| non_fall/adl-17-cam0-rgb | non_fall | non_fall | 0.0053 | 0.0053 | 1 / 1 |
| fall/fall-06-cam1-rgb | non_fall | non_fall | 0.0981 | 0.0981 | 0 / 0 |
| non_fall/adl-20-cam0-rgb | non_fall | non_fall | 0.0035 | 0.0035 | 0 / 0 |
| fall/fall-09-cam0-rgb | fall | non_fall | 0.9097 | 0.0099 | 1 / 0 |
| non_fall/adl-25-cam0-rgb | non_fall | non_fall | 0.0006 | 0.0006 | 0 / 0 |
| fall/fall-09-cam1-rgb | fall | fall | 0.7669 | 0.7669 | 0 / 0 |
| fall/fall-10-cam0-rgb | non_fall | non_fall | 0.0753 | 0.0753 | 1 / 1 |
| fall/fall-10-cam1-rgb | non_fall | non_fall | 0.2062 | 0.2062 | 0 / 0 |
| fall/fall-16-cam0-rgb | fall | fall | 0.9949 | 0.9949 | 2 / 0 |
| fall/fall-16-cam1-rgb | non_fall | non_fall | 0.0237 | 0.0237 | 0 / 0 |

## Missed-sequence evidence

Counts below describe actual returned person-frame observations. Confidence is not anatomical correctness; available predictions do not prove the pose is accurate.

| Sequence | No-track frames / total | Missing poses | Low-confidence poses | History resets | Prediction-ready observations | Max P |
|---|---|---|---|---|---|---|
| fall/fall-05-cam1-rgb | 106 / 151 | 9 | 18 | 1 | 10 | 0.172 |
| fall/fall-06-cam1-rgb | 5 / 100 | 0 | 0 | 0 | 69 | 0.0981 |
| fall/fall-09-cam0-rgb | 11 / 185 | 9 | 21 | 1 | 120 | 0.0099 |
| fall/fall-10-cam0-rgb | 22 / 130 | 1 | 16 | 0 | 80 | 0.0753 |
| fall/fall-10-cam1-rgb | 19 / 130 | 30 | 30 | 1 | 54 | 0.2062 |
| fall/fall-16-cam1-rgb | 8 / 55 | 19 | 19 | 0 | 14 | 0.0237 |

cam1 has no non-fall examples; its specificity and false-positive rate cannot be estimated. Camera views of the same action are correlated.
See the JSONL traces for exact tracking interruptions, candidate rejection evidence, confidence changes, history resets and handovers. No fabricated joints or pose interpolation were supplied to the classifier.
