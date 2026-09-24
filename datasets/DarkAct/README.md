# DarkAct

RGB–Thermal action recognition dataset ([source](https://github.com/darkact-creator/DarkAct)), used in this project for two purposes:
1. Training data for the RGB–thermal fusion backbone (alongside KAIST and LLVIP)
2. Hard-negative examples for the temporal collapse classifier (alongside URFD and UP-Fall)

**DarkAct has no "fall" or "collapse" class.** It contains 27 general action classes (walking, sitting, squatting, etc.) and is *not* a source of positive fall examples — only URFD and UP-Fall provide those. Verified directly against the dataset's own annotation files, not assumed from the paper.

## Dataset stats (verified, not estimated)

- 12,778 total RGB–Thermal clip pairs, 27 action classes, 30fps, clips 4.5–19.1s long
- Official split: 9,040 train / 3,738 test
- RGB/Thermal pairing: 0 mismatches, 0 missing files (checked across all 12,778 pairs)
- File integrity: 0 corrupted/unreadable files (checked across all 25,556 RGB + Thermal video files)

### A note on the annotation files
The dataset ships 14 annotation files. Two versions of each split exist:
- `*_train1_*` / `*_test1_*` — matches the officially published 9,040/3,738 split. **These are the ones used here.**
- Plain-named versions (no "1") — an alternate split (8,769/4,009 clips). Not used.

### Hard-negative classes
Five action classes were flagged as "hard negatives" — activities that could visually resemble a fall in a single frame, meant to train the collapse classifier not to false-alarm on them:

| Class | Train clips | Test clips |
|---|---|---|
| sit | 311 | 121 |
| squat | 233 | 124 |
| crouch | 192 | 77 |
| pick | 273 | 121 |
| lift | 293 | 99 |

1,844 clips total, flagged with `is_hard_negative=True` in the label table.

## Folder structure

```
DarkAct/
├── README.md
├── requirements.txt
├── scripts/
│   ├── explore_annotations.py       # inspects the raw annotation files
│   ├── check_multimodal_pairs.py    # verifies RGB/Thermal pairing + file existence
│   ├── check_video_integrity.py     # verifies every video actually opens (corruption check)
│   ├── build_label_table.py         # builds the standardized label table
│   └── darkact_dataset.py           # PyTorch Dataset for loading clips during training
└── cleaned/
    └── darkact_label_table.csv      # video_id, rgb_path, thermal_path, class_id,
                                      # class_name, split, is_hard_negative
```

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Download the raw dataset separately (not included in this repo — too large for git). Source: [DarkAct on GitHub](https://github.com/darkact-creator/DarkAct).
3. Each script has one variable to edit at the top before running:
   ```python
   ARIM_ROOT = r"C:\path\to\your\ARIM_v1"
   ```

## Running the pipeline (already done once — see stats above)

1. `explore_annotations.py` — sanity-checks the raw annotation files
2. `check_multimodal_pairs.py` — verifies RGB/Thermal pairing
3. `build_label_table.py` — builds `cleaned/darkact_label_table.csv`
4. `check_video_integrity.py` — confirms no corrupted files
5. `darkact_dataset.py` — example PyTorch `Dataset` for loading the clips

## Open decisions — needs team input, not resolved yet

These affect how DarkAct combines with URFD/UP-Fall for the temporal classifier:

- **Label scheme**: all 27 classes (down-sampled) plus the 5 hard negatives, or hard negatives only — and whether labels end up binary (`fall`/`not_fall`) or 3-way (`fall`/`not_fall`/`hard_negative`).
- **Modality mismatch**: URFD/UP-Fall are RGB-only; DarkAct is RGB+Thermal. Needs a decision on whether the classifier uses thermal at all, or whether thermal fusion stays restricted to the detection stage of the pipeline.