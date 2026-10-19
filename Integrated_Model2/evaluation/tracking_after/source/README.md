# Themam — Integrated Model 2 v0.1 (visibility + fall detection)

This initial integration **does not fuse two fall classifiers**. It wraps Souad's pretrained ST-GCN++ fall detector with Raneem's COCO-17 joint-visibility classification (`full_body`, `upper_body`, `head_shoulders`, `uncertain`). Original detection, threshold, and event behavior are preserved; visibility is added to per-track diagnostics and fall events.

## Included components
- `model2_fall_detection/`: Souad's inference code, original `.pt` checkpoint and `.json` settings (unchanged).
- `visibility/visibility_detection.py`: Raneem's original visibility classification code (unchanged).
- `integrated_model2.py`: new adapter.
- `tests/test_integration.py`: mocked interface checks (not accuracy tests).

## Install (from this folder)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r model2_fall_detection/requirements.txt
python -m unittest discover -s tests -v
```

## Usage
```python
from integrated_model2 import IntegratedModel2
model = IntegratedModel2(camera_id='cam1', use_fast=False)
# For each BGR frame with monotonically increasing capture time in seconds:
result = model.process(frame, timestamp_seconds, tracks_from_model1)
print(result['tracks'], result['events'])
```

The track schema is identical to Souad's original `FallDetectionModule`: each track has `track_id`, pixel `box`, optionally `confidence`, `keypoints` (17x2 xy pixels), and `keypoint_conf` (17). If the keypoints are missing, Souad's pose detector will be used. Visibility scores do **not** suppress alarms, because that could silently increase false negatives.

## Limitations / evaluation
- This wrapper adds visibility classification; it does not yet establish a performance gain, a calibrated confidence fusion, or an end-to-end test with Model 1.
- Runtime requires torch/numpy/ultralytics and possibly the YOLO pose weights (downloaded by Ultralytics). PyTorch version/architecture compatibility with the supplied TorchScript checkpoint must be checked on the target machine.
- A single pose pass supplies both fall inference and visibility. Unmatched tracks report missing skeletons; no previous-frame pose is drawn. Greedy one-to-one IoU matching prevents pose reuse between tracks, but overlapping people can still create association ambiguity.
- Classification is per frame; the 30-frame category is a rolling mode of the last 30 observed classifications, not Souad's time-resampled window.
- Run baseline Souad vs this wrapper on the **same held-out videos**; record event-level fall recall, precision, false alarms/minute, detection latency, throughput/FPS, and visibility-specific recall. Separate videos by actor/scene where possible to avoid leakage. Do not report cross-validation from one model as end-to-end performance.
- To compare RGB-only vs RGB+thermal, integrate the camera pipeline later; these source modules are RGB/pose-based.

## Skeleton video

```bash
source .venv/bin/activate
python test_video.py --video ../test/fall-06-cam0-test.mp4 --save evaluation/fall_06_cam0_skeleton.mp4 --show-skeleton
```

Skeleton drawing defaults to enabled; use `--no-show-skeleton` to disable it.
The returned per-track `keypoints` (17x2) and `keypoint_conf` (17) are copies
of the original image-space pose passed into Souad's temporal buffer. Missing
poses are `None`. Tracks are included even before person confirmation or a full
window, with `fall_probability=None` while unavailable.

Green denotes normal tracking; orange denotes a display-only possible-fall band
(0.50 <= probability < 0.70); red follows the model's existing confirmed alarm,
including its three-second hold. Joints below the configured `joint_ok=0.30`,
invalid coordinates, and joints outside the image are hidden. Bones require both
endpoints to be visible. Visibility classification still uses Raneem's unchanged
0.35 confidence threshold. The checkpoint, normalization, 30-frame / 15 FPS
sampling and 0.70 fall threshold are unchanged.

Console diagnostics distinguish missing poses, low joint confidence, person
confirmation, insufficient history, temporal gaps, invalid normalized clips,
below-threshold predictions, and track appearances/losses/history transfers.
The final maximum probability per ID helps review missed events; the script
cannot determine whether a human-visible fall occurred without ground truth.
Tests use synthetic fixtures for association and history behavior, not accuracy.

## Batch evaluation of PNG sequences (provisional)

Run from the `Integrated_Model2` directory using the existing macOS environment:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
# First evaluate one fall and one non-fall sequence:
python batch_evaluate.py --dataset ../test --fps 15 --limit 2 --save-videos --output evaluation/smoke
# Evaluate all sequences (omit --save-videos for metrics only):
python batch_evaluate.py --dataset ../test --fps 15 --save-videos
```

The batch runner discovers PNG sequence folders recursively below `fall/` and
`non_fall/`, sorts frame names numerically, and interleaves the two classes so a
small smoke test covers both. It loads networks once and resets ByteTrack,
Souad's temporal/confirmation/alarm state, and visibility history between
sequences. Timestamp = sorted frame index / assumed source FPS. `--fps` changes
source timing only; the model still samples 30 frames at 15 FPS with threshold
0.70. Missing numbering intervals are not reconstructed.

Outputs are `evaluation/batch_results.csv`, `summary.json`,
`confusion_matrix.png`, `performance_report.md`, and optional
`annotated_videos/`. Reports are saved after each sequence. Sequence errors are
recorded, partial predictions are excluded from metrics, and processing continues.
A sequence is predicted fall only when the existing model emits a fall event.
Visibility distributions count person-frame observations; camera and dominant
sequence-visibility groups are included. Undefined metric denominators are null
in JSON and N/A in the report. Model loading is excluded from processing FPS,
but inference warm-up, image decoding, and optional video encoding are included.

Results remain **provisional**: no sequence-level training split manifest or
original recording FPS metadata is available locally. The model README's claim
of a reserved test set does not establish that these specific sequences were
held out. Alternate-camera recordings of the same action must also be checked
for training overlap. cam1 currently has only fall examples.

Latency is omitted unless real onset annotations are supplied via `--onsets`:
provide a JSON object mapping dataset-relative sequence paths to onset seconds,
e.g. keys in the format `fall/fall-06-cam0-rgb`. Values must come from verified
annotations, never an assumed zero onset. Latency uses the first event minus the
annotated onset; negative values flag pre-onset alerts. Source FPS must also be
verified before interpreting latency.

The default Anaconda Python previously lacked `lap`; use `.venv/bin/python` or
activate `.venv` as above. Python 3.14 currently emits a TorchScript compatibility
warning; the existing checkpoint executed successfully in this environment.

## Tracking continuity safeguards and comparison

The fall core receives **all visible tracking IDs**, including people without a
matched pose, so a missing skeleton cannot make a present person a history donor.
New IDs can inherit a recently lost history only within one second, with compatible
box motion and at least four common confident joints (including two body joints).
Joint-shape comparison uses rigid rotation alignment so a falling body is not
rejected simply for changing orientation; reflection and scale fitting are excluded.
Motion prediction is used for association only: no estimated or interpolated
joints are passed to ST-GCN++. Matching is mutual, one-to-one and rejects close
competitors on both sides. IDs observed together cannot exchange histories.
Near-tied pose-box associations are also left unmatched. Malformed non-finite
poses are rejected rather than inserted into the classifier buffer.

After an interruption longer than 0.2 seconds, the same ID must pass a continuity
check; incompatible or stale histories and alarms are cleared. Expiration happens
before new observations are appended, preventing an old ID from reviving a stale
history. These conservative checks may sacrifice continuity during heavy occlusion.
They are geometric safeguards, not appearance-based person re-identification;
undetected ID swaps between similar people remain possible. No cross-person
identity guarantee is claimed without annotated multi-person evaluation.

Per-track diagnostics include missing-joint indices, mean confidence, confidence
changes, pose gaps, confirmation and temporal-window state. Tracking events record
appearances, interruptions, candidate evidence, accepted/rejected transfers and
history resets. `--trace` writes per-frame tracks, exact poses and all diagnostics
into the batch output's `traces/` directory. Reset counts include expiration of
inactive tracks and should not be interpreted as confirmed identity swaps.

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python batch_evaluate.py --fps 15 --save-videos --trace --output evaluation/tracking_after
python compare_evaluations.py --baseline evaluation/baseline_tracking \
  --after evaluation/tracking_after --output evaluation/tracking_comparison.md
```

The previous results and source snapshots are preserved in
`evaluation/baseline_tracking/`; original annotated baseline videos remain in
`evaluation/annotated_videos/`. The comparison refuses mismatched sequence IDs,
labels, frame counts, FPS, checkpoint hash or model settings, and excludes no
failed sequences silently. It reports the entire dataset, including non-falls.
