# Model 1 temporal identity diagnostic

This evaluates existing 30-observation RGB windows against MOT20 training person
IDs. It performs no detector, tracker, pose or classification inference. All writes
are confined to new `model1_identity_audit/run_*` directories. Historical predictions,
annotations, crops, reports and the frozen configuration remain unchanged.

## Commands

Run from `Senior_project`, using the existing environment:

```bash
../.venv/bin/python -B -m unittest discover -s model1_identity_audit -p 'test_*.py' -v
../.venv/bin/python -B model1_identity_audit/audit.py --check-only
../.venv/bin/python -B model1_identity_audit/audit.py --visual-per-sequence 2
```

For a smaller repeat: append `--sequences MOT20-01`. Omit the visual flag for CSV/JSON
evaluation only. Each evaluation creates a new directory. `--check-only` writes
nothing. Missing inputs or failed provenance checks fail closed, without inventing
scores; evaluation exits nonzero and lists limitations in `run_metadata.json`.

## Selected provenance

| MOT20 sequence | Crop/manifests run | Configured prediction CSV |
|---|---|---|
| 01 | `module1_sequences/MOT20-01/run_nb3r3kz2` | `confidence_experiments/tracking_conf_0.25.csv` |
| 02 | `module1_sequences/MOT20-02/run_kr6ki56d` | `validation_mot20_02/tracking_A.csv` |
| 03 | `module1_sequences/MOT20-03/run_7dy_fhf1` | `validation_mot20_additional/MOT20-03/tracking.csv` |
| 05 | `module1_sequences/MOT20-05/run_lemmoul4` | `validation_mot20_additional/MOT20-05/tracking.csv` |

These are the archival **buffer30** runs, not frozen buffer60 CV/test predictions.
Configuration evidence: `experiment_confidence_mot20.py:46`,
`validate_mot20_02.py:62` (A=30), `validate_mot20_sequences.py:27`, and
`validation_mot20_additional/configuration.json:9`.

Every crop metadata row must exactly match the configured prediction's frame,
Track ID, full-body box and confidence. Each manifest must exactly match that
metadata, and the complete window inventory must match independent replay of the
stored sampling/gap policy. All referenced crop files must exist. Prediction and
GT SHA-256 must also match a saved diagnostic evaluation using the same protocol
hash and as-stored coordinate policy. Input and manifest hashes are saved and
checked again after evaluation. The historical crop configurations did not record
generation-time checkpoint/environment hashes; current correspondence does not
retroactively supply those missing records.

## Matching and interpretation

1. Use original **full-body xyxy boxes**, not padded crop rectangles. Convert to
   xywh without an origin offset or inclusive `+1` extent.
2. Include **all predictions in each sampled source frame**, even predictions
   absent from the selected windows. Match once per frame and reuse the result.
3. Reuse `model1_cross_validation/evaluation/mot20_protocol.py:55` preprocessing:
   Hungarian matching against all GT first removes predictions matched to
   distractor classes 2, 6, 7, 8, 12. Retain marked class-1 pedestrians, without
   a visibility cutoff or generic crowd-region masking.
4. On remaining predictions and eligible GT, maximize total IoU with one-to-one
   Hungarian assignment; edges below 0.5 are zeroed and rejected. Sort by ID for
   reproducible tie ordering. This is **framewise geometric assignment** with no
   prior-frame identity carry-over, not motmetrics' tracking association algorithm.
5. Preserve GT ID, IoU, GT visibility, candidate count and match status per sample.
   Removed distractor observations have no pedestrian identity and count as
   unmatchable, not switches. Multiple eligible candidates are recorded for review;
   the assignment is deterministic, but overlap/ties can still be ambiguous evidence.
6. Classify each complete window:
   - **Consistent matched identity:** all 30 observations match one GT ID.
   - **Observed identity conflict:** at least two distinct matched GT IDs, even if
     other observations are unmatched.
   - **Inconclusive:** any unmatched observation without an established conflict.

Coverage is matched sampled observations / all sampled observations. Window
fractions use all selected windows, including inconclusive windows. Unique
observation counts are also reported so overlapping windows cannot silently
inflate a unique-sample denominator (the selected runs have stride 30).

This is conditional evidence for windows that survived the generator's strict
gap and length rules. It excludes rejected windows, short tracks and unsampled
source frames. A person can switch between windows while both windows remain
internally consistent. Conversely, an IoU association conflict can arise from
crowded overlapping boxes without establishing a visual identity switch. A crop
may contain other people; matching evaluates the intended tracked full-body box.

IDF1 assesses global identity association; MOTA combines false negatives, false
positives and identity switches over the full source timeline. Historical ID-switch
counts use motmetrics' temporal association rules. None is equivalent to this
30-sample classification. Comparisons use **only historical metrics whose exact
prediction, GT and protocol hashes match**, never the frozen buffer60 CV scores.
These are diagnostic results, not official MOTChallenge accuracy or proof of
physical identity. No GT identity scores are assigned to the official test crops.

## Outputs and visual review

- `summary.csv`: per-dataset-sequence counts, coverage and matching historical metrics.
- `run_metadata.json`: method, versions, code hashes and any unavailable inputs.
- `MOT20-*/observations.csv`: every sample with frame, timestamp, Track ID, box,
  crop path, GT ID (blank if unmatched), IoU, visibility and match status.
- `MOT20-*/windows.csv`: classifications, matched counts and distinct GT IDs.
- `MOT20-*/provenance.json`: exact source hashes and historical metric provenance.
- `MOT20-*/contact_*.png`: optional 30-crop sheets, ordered left-to-right then
  top-to-bottom, labeled with original frame, Track ID, GT assignment and IoU.

Visual selection is deterministic: conflicts first, then inconclusive windows;
within each category sort by Track ID, start frame and sequence ID. This is a
suspicious-example selection, **not a random estimate of visual accuracy**. Each
exported crop is pixel-compared with its original JPEG region before rendering;
the entire crop archive is not pixel-revalidated. Contact sheets preserve aspect
ratio and are thumbnails; CSV retains original geometry and paths.
