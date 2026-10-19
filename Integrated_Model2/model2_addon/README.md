# Tracked Raneem ST-GCN integration add-on (experimental)

This package is **additive**: copy `raneem_live/`, `checkpoints/`, and `tests/test_raneem_live.py` into your existing `Integrated_Model2` without replacing your Codex-modified code.

Both Raneem model checkpoints are loaded with strict weights. Input: *tracked* real 17-joint COCO pose in pixels, per-frame ID, camera frame/time. Each ID gets an independent 30-frame sequence and Raneem-style image-width/height normalization. No model retraining or tuned probability threshold. Original 0/1 argmax classification and 15-reliable-frame gate are preserved.

```python
from raneem_live import RaneemTrackedPipeline
model = RaneemTrackedPipeline(mode='parallel', camera_id='cam0')
# For each frame pass already tracked skeletons:
output = model.process(frame, seconds, [{
  'track_id':1, 'box':[x1,y1,x2,y2],
  'keypoints':xy_17_by_2, 'keypoint_conf':confidence_17
}])
print(output['tracks'])
```

Modes: `visibility`, `fullbody`, `parallel`. In `parallel`, both predictions are returned but **no fusion or alert** is applied; decisions aren't calibrated. In single mode, events are **candidate_fall** only (not emergency-confirmed). `reset()` before a new recording. For a full video runner, connect to your **existing** single-pass YOLO-pose and tracked-person pipeline, not an additional pose model.

Test locally from `Integrated_Model2`:
```
source .venv/bin/activate
python -m unittest discover -s tests -p test_raneem_live.py -v
```

**Limitations:** The original comparison used top-confidence-person windows; this module uses per-ID input instead, so its results MUST be re-evaluated. It does not yet implement the ID-handover of your latest Codex build, and expects pose keypoints to be supplied by caller. It does not automatically integrate into the dashboard or existing batch runner. Source FPS and held-out status remain unverified; the 15 diagnostic sequences are not an independent final test. Do not treat these events as validated collapse diagnoses.
