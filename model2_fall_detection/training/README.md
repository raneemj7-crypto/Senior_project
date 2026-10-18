# Training (Google Colab)

`Themam_STGCN_Clean.ipynb` is the training notebook. After changing it in Colab, download it again
(File → Download → Download .ipynb) and replace this copy. Sections 10 (training) and 12 (final test)
are locked: set `RETRAIN = True` / `RUN_FINAL_TEST = True` only on purpose.

The data is **not** in this repository: URFD and all caches stay in Google Drive
(`MyDrive/UDFD` and `MyDrive/Themam_STGCN`). Personal recordings must never be uploaded here.

## Notebook sections

| Section | What it does |
|---|---|
| 1–4 | GPU check, Drive paths, installs (fixed versions), pretrained ST-GCN++ (NTU60) |
| 5–6 | Split check (no event in two splits), URFD manifest with posture labels |
| 7–8 | Frozen tracker (BoT-SORT + ReID) on all videos, automatic actor choice + visual review |
| 9 | Clean skeletons (gaps ≤ 5 frames filled) → 2-s clips at 15 FPS with labels |
| 10 | Train ST-GCN++ (fall / non-fall) on train, pick the epoch on validation |
| 10b | 5-fold cross-validation: 2 s vs 3 s clips |
| 10c | Videos showing how the model judged a video (box, skeleton, probability) |
| 10d | Fair comparison with the LSTM (same clips, labels, folds) |
| 10e | Alarm threshold check → 0.7 |
| 11 | Export for the laptop / integration (`models/` files) |
| 12 | Final test on the 10 URFD test videos — run once, at the very end |

## Main settings
- Pose: YOLOv8n-pose (Ultralytics 8.4.175), detector confidence 0.10, image size 640
- Clips: 30 frames, 2 s at 15 FPS, a new clip every 0.2 s; per-clip normalization (centre and scale)
- Labels: a clip is "fall" if it covers ≥ 80% of URFD's falling frames; clips before the fall are "non-fall"
- Model: MMAction2 ST-GCN++ (NTU60 pretrained), 2 classes, AdamW, cosine schedule, 60 epochs, class weights, flip/scale/shift augmentation, seed 42
