# MOT20 Dataset Cleaning and Validation

## Purpose

MOT20 is used to evaluate multi-object pedestrian tracking in highly crowded scenes for the collapse-detection project. The dataset supports person detection, identity tracking, occlusion testing, and crowd-stress evaluation. It is not a fall or collapse classification dataset.

## Dataset ownership

- Dataset: MOTChallenge MOT20
- Project role: `C` — crowded multi-object tracking
- Modality: RGB
- Target project label: `person`
- Official split preserved: train/test by sequence
- Full raw and processed sequences remain in shared storage and are not committed to GitHub.

## Storage structure

```text
senior-project-storage/
├── raw/mot20/
│   ├── train/
│   │   ├── MOT20-01/
│   │   ├── MOT20-02/
│   │   ├── MOT20-03/
│   │   └── MOT20-05/
│   └── test/
│       ├── MOT20-04/
│       ├── MOT20-06/
│       ├── MOT20-07/
│       └── MOT20-08/
└── processed/mot20/
    ├── images/{train,test}/<sequence>/
    └── annotations/train/<sequence>/gt.txt
```

Repository files:

```text
Senior_project_github/
├── scripts/clean_mot20.py
└── reports/mot20/
    ├── README.md
    ├── dataset_summary.md
    ├── sequence_manifest.csv
    ├── clean_manifest.csv
    ├── integrity_report.csv
    ├── cleaning_statistics.csv
    └── validation_statistics.json
```

## Cleaning rules

1. The raw dataset is read-only and is never deleted or modified.
2. Every image is decoded with Pillow and checked against the declared resolution.
3. Declared frame counts, frame names, missing frames, and extra frames are checked.
4. Ground-truth rows are checked for nine fields, numeric values, valid frame references, positive track IDs, positive box sizes, valid visibility, mark values, and class IDs.
5. Annotation-to-image references and duplicate frame/track-ID pairs are checked.
6. SHA-256 hashes are used to identify exact duplicate image files.
7. Only active pedestrian annotations with `mark = 1` and `class_id = 1` are written to processed ground truth.
8. The original MOT20 label `pedestrian` is normalized to the project label `person` using issue code `F-LABEL-NORM`.
9. Known non-target classes remain unchanged in raw storage but are excluded from processed pedestrian annotations.
10. Boxes crossing image boundaries are clipped and recorded with `F-BOX-CLIP` when present.
11. Official train and test sequence assignments are preserved to avoid leakage.
12. MOT20 test sequences have no local ground truth; this is expected and is not treated as an error.

## MOT20 class mapping

| Class ID | Original class | Project use |
| ---: | --- | --- |
| 1 | pedestrian | Normalize to `person` and keep |
| 2 | person on vehicle | Exclude from target annotations |
| 3 | car | Exclude from target annotations |
| 4 | bicycle | Exclude from target annotations |
| 5 | motorbike | Exclude from target annotations |
| 6 | non-motorized vehicle | Exclude from target annotations |
| 7 | static person | Exclude from target annotations |
| 8 | distractor | Exclude from target annotations |
| 9 | occluder | Exclude from target annotations |
| 10 | occluder on ground | Exclude from target annotations |
| 11 | full occluder | Exclude from target annotations |
| 12 | reflection | Exclude from target annotations |
| 13 | crowd | Exclude from target annotations |

## Reproduction

Install Pillow in the active Python environment:

```bash
python3 -m pip install Pillow
```

Run the complete validation and cleaning pipeline from the repository root:

```bash
caffeinate -i python3 scripts/clean_mot20.py \
  --dataset-root "$HOME/Desktop/senior-project-storage/raw/mot20" \
  --output-dir "$HOME/Desktop/Senior_project_github/reports/mot20" \
  --processed-root "$HOME/Desktop/senior-project-storage/processed/mot20"
```

The command is safe to rerun. Existing processed images with matching file sizes are not recopied.

## Verified results

| Metric | Result |
| --- | ---: |
| Sequences | 8 |
| Valid sequences | 8 |
| Invalid sequences | 0 |
| Frames/images | 13,410 |
| Ground-truth rows | 1,336,920 |
| Active pedestrian boxes | 1,134,614 |
| Removed non-target annotation rows | 202,306 |
| Clean-manifest samples | 13,410 |
| Kept test samples | 4,479 |
| Fixed/normalized training samples | 8,931 |
| Excluded samples | 0 |
| Corrupt images | 0 |
| Dimension mismatches | 0 |
| Exact duplicate images | 0 |
| Clipped boxes | 0 |

## Clean manifest

`clean_manifest.csv` contains one row per frame and exactly these 15 columns:

```text
sample_id,dataset,role,modality,path_rgb,path_thermal,sequence_id,subject_id,frame_index,original_label,project_label,status,issue_code,split,notes
```

Missing values use the literal value `none`. Status values are `keep`, `fix`, or `exclude`.

## Intended sequence use

| Sequence | Intended use |
| --- | --- |
| MOT20-01 | Initial smoke test |
| MOT20-02 | Long indoor crowd test |
| MOT20-03 | High-density night test |
| MOT20-05 | Maximum-crowd stress test |
| MOT20-04, MOT20-06, MOT20-07, MOT20-08 | Tracker testing without local ground truth |

## GitHub policy

Commit the cleaning script, documentation, manifests, and statistical reports. Do not commit the 4.8 GB raw dataset or the processed image sequences. Any small sample must comply with the MOTChallenge dataset terms.
