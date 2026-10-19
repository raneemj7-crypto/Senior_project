# Tracking and skeleton diagnosis

## Result

The final complete evaluation is unchanged: TP=5, FP=0, TN=5, FN=5; recall=50%, precision=100%, F1=66.67%. cam0 recall=80%, cam1 recall=20%. There is **no demonstrated recall improvement**. Results remain provisional because source FPS and checkpoint split provenance are unverified. All timestamps below assume 15 FPS.

## fall-10-cam0-rgb: what actually happened

The original pipeline was replayed before modifying the core. Its full per-frame trace is preserved in `baseline_tracking/fall10_trace.jsonl` and its transition log in `baseline_tracking/fall10_trace.log`.

- The sequence contains 130 frames, including 22 frames without a tracking detection.
- ID 1 disappears and returns several times: gaps of 0.60 s (return at 4.733 s), 0.333 s (return at 5.067 s), and 0.20 s (return at 5.333 s).
- Last observation of ID 1: zero-based frame 85, 5.667 s. ID 12 first appears at frame 94 (PNG 095), 6.267 s, a 0.60 s gap.
- The **baseline successfully transferred ID 1 → ID 12**. Immediately afterward it had 15 buffered observations spanning 2.4 s, retained person confirmation (59 clear observations), and status `ready`; probability was 0.0002. There was no history restart at this handover.
- The final implementation also accepts this handover: five common confident joints, motion residual 0.4726 box-size units, keypoint residual 0.2755, rotation-aligned shape residual 0.1051. It retains confirmation and scores immediately, with 14 buffered observations after pruning. One frame elsewhere was withheld by conservative pose association; no new joint values were generated.
- Baseline and final maximum probability are both **0.0753**, far below 0.70. No fall event occurs in either run.
- At frames 71, 80 and 85, the baseline has only 7, 8 and 6 confident joints respectively. Tracking continuity alone cannot recover missing anatomical information during this motion.

The missed fall is therefore **not explained by a failed ID 1 → ID 12 history transfer**. Detection/pose interruptions and the resulting skeleton evidence remain relevant; a classifier limitation on the observed poses is also possible.

## Remaining pose versus viewpoint limitation

| Missed sequence | Frames with at least one usable returned pose / all frames | Prediction-bearing person observations | Peak P | Interpretation |
|---|---|---|---|---|
| fall-05-cam1 | 36 / 151 | 10 | 0.1720 | Severe detection/pose coverage limitation; 106 frames have no tracks. |
| fall-06-cam1 | 95 / 100 | 69 | 0.0981 | Strongest evidence consistent with classifier viewpoint sensitivity: 91 observations have at least 13 confident joints, yet scores remain low. |
| fall-10-cam0 | 107 / 130 | 80 | 0.0753 | Successful history handover, interrupted/weak poses during motion; tracking transfer is not the cause of the miss. |
| fall-10-cam1 | 107 / 130 | 54 | 0.2062 | Mixed: missing poses, tracking interruptions and a history reset, despite many confident returned poses. |
| fall-16-cam1 | 28 / 55 | 14 | 0.0237 | Major pose-coverage loss and a short sequence leave little scored temporal evidence. |

Pose frames count frames with any returned pose; prediction observations count per-person scores, including cached scores. Neither count is an accuracy label. The final traces are the evidence source for this table.

The cam0/cam1 comparison for fall-06 is informative: cam0 peaks at 0.9963, cam1 at 0.0981. Visual inspection of baseline frames 45, 50 and 60 also shows a different camera projection and a brief missing track during the motion in cam1. Thus **viewpoint generalization is a credible classifier limitation once pose coverage is adequate**, but high keypoint confidence and a `full_body` label do not prove anatomical correctness.

The evidence does not justify assigning one universal root cause across all five misses. Detection/pose coverage is the clearest measured upstream limitation in fall-05-cam1 and fall-16-cam1; fall-06-cam1 is the clearest candidate for classifier viewpoint sensitivity. Tracking handover is not the main demonstrated bottleneck. To separate pose error from classifier behavior conclusively, verify joint correctness, actor identities and fall onset on a stratified sample from both cameras; no oracle poses or onset timestamps were fabricated here.

## Changes and identity safeguards

- `model2_fall_detection/tracking.py`: motion-aware, short-gap association with common-joint checks and rigid rotation-aligned shape comparison. Mutual-best matching and margins on both sides reject competing identities. Alignment affects association only.
- `model2_fall_detection/fall_core.py`: protects all currently visible IDs, forbids transfers between previously coexisting IDs, expires stale state before append, checks interrupted same-ID continuity, and records quality/transfer/reset evidence. Low-confidence poses are not filled in.
- `model2_fall_detection/fall_module.py`: passes all tracking IDs even when poses are absent, rejects near-tied box/pose assignments, rejects invalid numeric poses, and exposes tracking diagnostics.
- `integrated_model2.py`: clears visibility history on explicit core history resets; visibility classification itself is unchanged.
- `test_video.py`: prints continuity decisions alongside existing pose/probability diagnostics; existing shared tracking/rendering helpers remain in use. YOLO/ByteTrack thresholds were not changed.
- `batch_evaluate.py`: optional `--trace` JSONL output plus aggregate interruption, confidence, missing-joint and history counters.
- `compare_evaluations.py`: validates matching completed datasets/settings/checkpoint and generates the before/after tables.
- `tests/test_tracking_continuity.py`: identity continuity and separation regressions, including rotation, coexisting people, absent poses, ambiguous matches and stale-ID reuse.
- `README.md`: usage, limitations and reproducibility commands.

No retraining, extra pose model, changed model weights, synthesized classifier joints or edits to original teammate repositories were made. Normalization was compared structurally with the baseline AST; checkpoint, settings JSON and Raneem's visibility file hashes were verified unchanged. The model's 30-frame window, 15 FPS sampling, person confirmation rules and 0.70 threshold are preserved.

Geometric association cannot guarantee identity with indistinguishable or continuously swapped tracks. Conservative abstention may lose useful poses in crowded/duplicate-box scenes. Synthetic multi-person regressions exercise known hazards but do not establish real-world identity accuracy.

## Tests and experiment record

- 22 unit/regression tests passed; output: `tracking_tests.log`.
- Original fall-10-cam0 replay recorded before code changes.
- First complete 15-sequence candidate run: TP=4, FP=0, TN=5, FN=6. Its shape test incorrectly penalized body rotation during a legitimate handover in fall-09-cam0. Preserved in `tracking_attempt1/`, including its comparison, traces, videos and association source.
- Added a general rotation regression and corrected association using rigid rotation alignment; no detection threshold was tuned and no sequence-specific rule was added.
- Final complete 15-sequence run: zero processing errors; TP=5, FP=0, TN=5, FN=5. All 15 saved videos decode completely and all JSONL traces have the expected total of 2,212 frames.
- The project `.venv` ran successfully with existing dependencies. The existing Python 3.14 TorchScript compatibility warning remains; it did not prevent inference.
- Observed final average/median processing rates were about 18.02/17.92 FPS with video and trace writing. These are not controlled performance benchmarks.

Baseline reports and source snapshots remain in `baseline_tracking/`; original baseline videos remain in `annotated_videos/`. Final results and source snapshots are in `tracking_after/`. The original `evaluation/batch_results.csv` and `evaluation/summary.json` were not overwritten.
