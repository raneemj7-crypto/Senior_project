# Themam — Model 2: Fall / Collapse Detection

Model 2 of the Themam crowd-safety system. For every tracked person it watches the body
skeleton over the last 2 seconds and decides whether the person is **falling**.

```
camera frame ──► Model 1: detect + track people ──► track ID + box per person
                                                         │
                         Model 2 (this repo)             ▼
             YOLOv8n-pose: 17 body points per person, matched to the Model 1 boxes
                         │
             last 2 s of skeletons per track ID (30 frames at 15 FPS, normalized)
                         │
             ST-GCN++ ─► fall probability per person ─► ≥ 0.7 ─► FALL EVENT
```

## Results (URFD dataset, cam0)

5-fold cross-validation on the 47 train + validation videos (25 falls, 22 normal); every video
was judged by a model that never saw it. The 10 test videos are kept for one final test.

| | F1 (clips) | Falls caught (videos) | Normal videos with a false alarm |
|---|---|---|---|
| **ST-GCN++ (this model)**, threshold 0.5 | **0.96** | **25/25** | 2/22 |
| ST-GCN++, threshold **0.7** (used) | — | **25/25** | **0/22** |
| LSTM (same clips, folds and labels) | 0.90 | 24/25 | 3/22 |

- 2-second clips beat 3-second clips (F1 0.96 vs 0.86).
- The threshold 0.7 was chosen from cross-validation (0.6–0.8 all caught every fall with no
  false-alarm video), before any test.

**Known limitation — be aware during integration:** in a live test in a new room (home webcam)
the model caught 2 of 4 falls and gave 5 false alarms in ~3.5 minutes (e.g. raising the arms).
It was trained only on URFD (one room, one camera). Own recordings from the real camera setup
are planned (v2). Swapping in a v2 model needs **no code change** — only the two files in `models/`.

## Files

| File | What it is |
|---|---|
| `fall_module.py` | **For integration.** `FallDetectionModule.process(frame, time, tracks)` |
| `fall_core.py` | Shared logic: skeleton history per person, normalization, ID hand-over, alarm |
| `live_fall_demo.py` | Stand-alone laptop demo (webcam or video, its own tracker) |
| `integration_example.py` | Example: a Model 1 stand-in (YOLOv8n + ByteTrack) feeding Model 2 |
| `models/` | The trained model: `stgcnpp_fall_v1_laptop.pt` + `stgcnpp_fall_v1_laptop.json` |
| `training/` | The Colab training notebook `Themam_STGCN_Clean.ipynb` (data is not in the repo) |
| `tests/test_module.py` | Quick check of the integration module, no models needed: `python tests/test_module.py` |

## Setup

Python 3.10 or newer (the demo has been run with Python 3.11 on Windows).

Nothing from Google Drive is needed to **run** Model 2: only the two files in `models/`
(Drive is only used for training). Use the PyTorch version written in
`models/stgcnpp_fall_v1_laptop.json` (`torch_version`) or newer.

```bash
pip install -r requirements.txt
```

The pose model (`yolov8n-pose.pt`) downloads automatically the first time. On a CPU, OpenVINO
makes it about 3× faster; the first run converts it (about 1 minute).

## Run the demo

```bash
python live_fall_demo.py                       # webcam
python live_fall_demo.py --video test.mp4      # a video file
python live_fall_demo.py --video test.mp4 --save checked.mp4
```
Q = quit, C = clear an alarm. The fall probability of every moment is saved in `fall_log.csv`.

Best camera setup: about 1 m high, 3–4 m away, whole body and floor visible, good light.

## Integration (Model 1 → Model 2)

```python
from model2_fall_detection import FallDetectionModule     # from the repo root
# (inside this folder:  from fall_module import FallDetectionModule)

fall = FallDetectionModule(camera_id="cam1")          # load once

# for every frame:
tracks = [                                            # from Model 1
    {"track_id": 3, "box": [x1, y1, x2, y2], "confidence": 0.91},
    {"track_id": 7, "box": [x1, y1, x2, y2], "confidence": 0.84},
]
out = fall.process(frame, t, tracks)

out["tracks"]   # {3: {"fall_probability": 0.03, "falling": False}, 7: {...}}
out["events"]   # [] or [{"event": "fall_detected", "camera_id": "cam1", "track_id": 7,
                #        "time": 12.4, "fall_probability": 0.93, "box": [...], ...}]
```

### What Model 1 must give, every frame
| Field | Meaning |
|---|---|
| `frame` | The image (BGR, as read by OpenCV) |
| `t` | Capture time in **seconds**, always increasing (not the frame number) |
| `track_id` | Same number for the same person over time |
| `box` | `[x1, y1, x2, y2]` in pixels **of this frame** |
| `confidence` *(optional)* | Detector score of the box |
| `keypoints`, `keypoint_conf` *(optional)* | If Model 1 already runs a COCO-17 pose model: 17×[x, y] and 17 scores. Then Model 2 does not run a second pose model (faster). |

### What Model 2 returns
- `tracks`: fall probability per person (`None` for the first ~2 s of a new person) and
  `falling` (True for 3 s after a high probability).
- `events`: one `fall_detected` event **when a fall starts** — send these to the dashboard.
  The event is JSON-ready.

### Notes
- **Frame rate:** any; the module resamples by time to 15 FPS. At least ~12 FPS is recommended,
  otherwise people get lost between frames during fast falls.
- **ID switches:** a person often gets a new track ID while falling. Model 2 handles short switches
  (a new ID appearing where an old one was lost < 1 s ago keeps that person's history), but stable
  IDs from Model 1 mean fewer missed falls.
- **Who counts as a person:** a track is only analysed after it looked clearly like a person in
  5 frames (box ≥ 0.5 and ≥ 10 clear joints) — this ignores pillows, clothes and shadows.
- **Crowds:** all people who need a new prediction are scored together in one batch.
- **Same settings as training:** do not change `normalize_clip()` or the clip settings in the json.

## Training

See `training/README.md`. Trained in Google Colab (GPU) on URFD skeletons: YOLOv8n-pose →
BoT-SORT + ReID tracking → actor selection with visual review → 2-s clips at 15 FPS →
ST-GCN++ fine-tuned from NTU60 (MMAction2).
