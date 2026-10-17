# Model 1 temporal identity findings — 2026-10-15

**A persistent Track ID does not guarantee the same physical person across a
window.** Under the documented diagnostic box-association convention, 6,493 of
8,494 selected training windows (76.44%) match one GT identity throughout all 30
samples. There are 772 observed identity conflicts (9.09%) and 1,229 inconclusive
windows (14.47%). These are GT association observations, not official accuracy or
a visually verified physical-identity success rate.

## Scope and provenance

Results: `run_gbg5837h/summary.csv:1`; method and code hashes:
`run_gbg5837h/run_metadata.json:1`; per-source exact hashes and matching historical
metric references: `run_gbg5837h/MOT20-*/provenance.json:1`.

The read-only audit preceding implementation established that all four training
crop runs correspond exactly to their configured tracking CSVs. All 983,579 crop
metadata rows agree with their prediction rows; every one of the 254,820 sampled
manifest observations agrees with those boxes and Track IDs. Prediction and GT
hashes match the previously saved standardized diagnostic evaluations. The
implementation repeats those checks and independently reconstructs window
inventory before matching (`audit.py:131`).

**These are archival buffer30 runs, not the frozen buffer60 runs.** Their settings
are evidenced by `experiment_confidence_mot20.py:46`, `validate_mot20_02.py:62`
(A=30), and `validation_mot20_additional/configuration.json:9`. The exact selected
paths appear in `README.md` and `audit.py:30`. No new windows were substituted and
no scores from another tracker configuration were mixed into the comparison.
Generation-time checkpoint/environment hashes were not recorded in these older
crop configurations; exact current correspondence and historical evaluation
hashes cannot fill that historical provenance gap.

## Sequence-level results

| Dataset | Windows | Consistent matched identity | Observed conflict | Inconclusive | Matched samples / all samples | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 106 | 79 (74.53%) | 16 (15.09%) | 11 | 2,908 / 3,180 | 91.45% |
| MOT20-02 | 1,015 | 738 (72.71%) | 161 (15.86%) | 116 | 27,801 / 30,450 | 91.30% |
| MOT20-03 | 2,748 | 2,305 (83.88%) | 127 (4.62%) | 316 | 76,641 / 82,440 | 92.97% |
| MOT20-05 | 4,625 | 3,371 (72.89%) | 468 (10.12%) | 786 | 119,992 / 138,750 | 86.48% |
| **Total** | **8,494** | **6,493 (76.44%)** | **772 (9.09%)** | **1,229** | **227,342 / 254,820** | **89.22%** |

Of 27,478 observations without pedestrian identity matches, 24,910 were removed
by the documented distractor preprocessing and 2,568 remained unmatched against
eligible pedestrian GT. Neither category alone is an identity switch. The 17,226
observations with more than one eligible IoU candidate are flagged in the CSV;
one-to-one assignment resolves competition deterministically but cannot eliminate
geometric ambiguity in crowded scenes.

All observations and GT identities are preserved in each dataset's
`observations.csv`; classifications and distinct GT IDs are in `windows.csv`.
Classification prioritizes an established conflict over unmatched observations
(`audit.py:65`). Matching uses all predictions in a sampled frame, not merely
window members (`audit.py:77`), and original full-body boxes rather than padded
crop rectangles. It reuses the existing distractor preprocessing and as-stored
coordinate policy (`model1_cross_validation/evaluation/mot20_protocol.py:55`).

## Comparison with existing metrics for the exact same predictions

| Dataset | Historical diagnostic IDF1 | Historical diagnostic MOTA | Historical ID switches | Windows with observed conflict |
|---|---:|---:|---:|---:|
| MOT20-01 | 63.22% | 70.86% | 203 | 16 |
| MOT20-02 | 56.84% | 69.96% | 1,259 | 161 |
| MOT20-03 | 76.55% | 77.71% | 1,074 | 127 |
| MOT20-05 | 58.50% | 70.03% | 3,565 | 468 |

Historical metrics were read, not recomputed. Exact prediction/GT/protocol hashes
and as-stored coordinates gate comparison (`audit.py:108`). MOT20-03 has both the
highest existing IDF1 and the strongest observed window consistency here; this
agreement does not make the measurements interchangeable.

IDF1 measures global identity association over the evaluated timeline. MOTA
includes misses, false positives and switches. Historical ID-switch counts use
motmetrics' temporal association. This diagnostic instead asks whether **each
selected 30-sample window** has one framewise matched GT identity, multiple
identities or insufficient matches. One window can contain several changes;
changes between windows can leave both windows internally consistent. Short,
interrupted or rejected tracks are excluded. Sampling also omits source frames.
Therefore 76.44% is conditional window consistency evidence, not overall tracker
identity accuracy and not a replacement for IDF1.

## Visual audit and validation

Eight contact sheets were exported, two suspicious windows per dataset, chosen
deterministically with conflicts first. All 240 displayed crops exactly matched
the corresponding original JPEG crop pixels. The images preserve sample order,
frame numbers, Track IDs, GT labels and IoU. Start with
`run_gbg5837h/MOT20-01/contact_01.png`: Track 16's window contains associations to
GT 33, 49 and 53, plus an unmatched observation. The crowded crops contain more
than one person, illustrating why association evidence is not a claim that every
person visible in a crop has the assigned identity. This example selection is not
a random visual accuracy estimate. Only these displayed crops were pixel-checked;
all sampled crop paths were checked for existence.

Validation performed:

- 10 lightweight identity tests passed before evaluation, including all requested
  synthetic scenarios, competing predictions, distractors, IoU threshold, empty
  input and deterministic tie ordering (`test_audit.py:18`).
- 12 existing MOT20 protocol tests passed.
- Independent readback of output CSVs verified all 8,494 classifications, exactly
  30 ordered samples per window and one-to-one GT assignments within each frame.
- Source hashes were checked again after scoring. The frozen configuration hash
  remains `5b7b0613078fb5d4137f3ce92ab2359a29f02a0f15eea447bda2a5778280ceb1`.

No inference, downloads, training, changes to pipelines or historical outputs,
or Module 2 implementation were performed. Commands are in `README.md`.

## What this establishes and what remains unknown

Model 1's strict temporal construction preserves chronological Track ID
continuity, but the evaluated archival windows do not universally preserve a
single GT person identity. Report the three categories explicitly rather than
describing all valid windows as identity-clean.

This evaluation does **not** establish the frozen buffer60 configuration's window
identity rate. The existing training manifests reference different predictions;
the four official-test crop runs have no local identity GT used here. A future
buffer60-specific diagnostic would need new manifests generated from the exact
saved buffer60 training predictions, in a separate output directory. It must not
reuse these archival manifests or attach these scores to frozen test results.
