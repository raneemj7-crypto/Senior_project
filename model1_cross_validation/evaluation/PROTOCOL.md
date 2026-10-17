# Standardized MOT20 person evaluation — diagnostic

This new evaluator leaves original inputs and evaluators unchanged. All four sequences use the same preprocessing and metrics. It reads complete seqinfo timelines, including frames with no predictions. No detector/tracker inference occurs.

## Evidence and rules

Reference inspected 2026-10-12: [TrackEval MOTChallenge dataset implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py), `get_preprocessed_seq_data`. The upstream URL is not pinned; before formal results, pin an exact TrackEval commit and verify parity. TrackEval was not installed in the project environment; motmetrics 1.4.0 is available.

1. Use all GT classes for one-to-one Hungarian matching, maximizing IoU after setting similarities below 0.5 minus machine epsilon to zero.
2. Remove predictions matched to distractor classes 2, 6, 7, 8, 12 (MOT20 includes non-MOT vehicles).
3. Keep marked pedestrian GT (class 1, mark nonzero). This input validator requires binary marks.
4. There is no additional generic crowd-region suppression in the referenced MOTChallenge loader. Classes 11 and 13 are not blanket ignore masks. A zero-marked pedestrian does not by itself suppress its matched prediction in this implementation.
5. Do not filter on visibility. Preserve it for inspection. Zero visibility is not equivalent to an ignored annotation.
6. Use motmetrics IoU distance <= 0.5 for tracking association and compute IDF1/IDP/IDR, MOTA, switches, fragmentations, precision/recall, FN/FP. No scores are recomputed during check-only mode.

## Coordinates and provenance

Input CSV boxes are continuous pixel xyxy with width=x2-x1 and height=y2-y1, without +1 extent. GT is xywh. Require explicit `--prediction-origin as-stored` or `zero-to-one`; the latter translates prediction x/y by +1 without changing extents. Do not select the offset by whichever produces better held-out scores. Establish the source convention first. The old code applied no offset; as-stored reproduces that choice, not proof that it is benchmark-correct. Bounds are reported, not silently clipped: full-body GT may extend beyond the visible image. Reject nonfinite values, duplicate frame/ID keys, nonpositive extents, and frames outside 1..seqLength. Head-only rows are not full-body predictions.

Detector and tracker labels are explicit CLI metadata, not inferred from filenames. Defaults describe only the supplied YOLOv5 CrowdHuman + ByteTrack presets; callers overriding predictions must supply matching names. Source hashes, dependencies, settings and GT class/visibility summaries are recorded in each unique output folder. Historical checkpoint provenance remains unverified.

## Remaining limitations

This is a documented fallback, not TrackEval and not certified official scoring. motmetrics association/identity/fragmentation semantics and tie handling have not been checked for parity with a pinned TrackEval version. Synthetic tests establish selected behavior, not official equivalence. No HOTA or official submission is implemented. Existing exploratory exposure remains; better evaluation does not retroactively make results held out. No pooled metric is obtained by averaging sequence scores.

The corrected detection-only script replaces the mislabeled `evaluate_mot20.py` use case: it reads the legacy YOLO26 label directory and requires the actual detector name. It reports detection F1, not IDF1 or AP, and records tracker=none. Missing label files are treated as empty detections, which must be corroborated by generation logs.

## Usage from Senior_project

```bash
../.venv/bin/python model1_cross_validation/evaluation/evaluate_person_tracking.py --check-only --prediction-origin as-stored
../.venv/bin/python -m unittest discover -s model1_cross_validation/evaluation -p 'test_*.py' -v
```

Remove `--check-only` only when ready to score. Optional overrides: `--sequences MOT20-02`, `--prediction MOT20-02=/path/tracking.csv`, `--detector-name`, `--tracker-name`, `--data-root`, `--output-root`.

Corrected detection evaluator (not run automatically):

```bash
../.venv/bin/python model1_cross_validation/evaluation/evaluate_yolo26_detections_corrected.py --detector-name YOLO26 --prediction-origin as-stored
```
