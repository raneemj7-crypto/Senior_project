"""Checks the integration module without real models or videos:  python tests/test_module.py
A) Model 1 gives keypoints: one person falls (ID changes mid-fall), one stands -> exactly one event, one batch.
B) Model 1 gives only boxes: the module matches skeletons to boxes by overlap."""
import json, sys, types
from pathlib import Path
import numpy as np, torch
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import fall_core, fall_module

class FakeNet:                                    # "fall" = body turned from upright to lying in the clip
    def eval(self): return self
    def __call__(self, x):
        out = []
        for c in x.numpy():
            v = c[:, 0, 2] > 0
            tilt = lambda f: np.ptp(f[:, 0]) / (np.ptp(f[:, 1]) + 1e-6)
            out.append(float(v.sum() >= 3 and tilt(c[v][-1]) > 2 and tilt(c[v][0]) < 0.5))
        return torch.tensor(out)
torch.jit.load = lambda *a, **k: FakeNet()
fall_core.find_model_dir = lambda d=None: d
import tempfile; d = Path(tempfile.mkdtemp())
(d / "stgcnpp_fall_v1_laptop.json").write_text(json.dumps(dict(threshold=0.7, clip_frames=30, clip_fps=15,
    joint_ok=0.3, max_missing_frac=0.3, pose_model="yolov8n-pose.pt", detector_conf=0.1)))
(d / "stgcnpp_fall_v1_laptop.pt").write_bytes(b"x")

def body(a, cx, cy, L=300):
    s = np.linspace(-.5, .5, 17); return np.stack([cx + L*s*np.sin(a), cy + L*s*np.cos(a)], 1)
def box(k): return [float(k[:, 0].min()) - 40, float(k[:, 1].min()) - 10, float(k[:, 0].max()) + 40, float(k[:, 1].max()) + 10]   # stick bodies have no width: pad the box

# A) Model 1 gives keypoints
m = fall_module.FallDetectionModule(model_dir=d, camera_id="camA")
events, batches = [], []
real_predict = m.core.predict
m.core.predict = lambda c: (batches.append(len(c)), real_predict(c))[1]
for i in range(15 * 8):
    t = i / 15
    a = 0 if t < 3 else min(1, (t - 3) / .7) * np.pi / 2
    me = body(a, 300, 240 + 100 * a / (np.pi / 2)); other = body(0, 900, 240)
    tracks = [{"track_id": 1 if t < 3.3 else 5, "box": box(me), "confidence": 0.9, "keypoints": me, "keypoint_conf": np.full(17, .9)},
              {"track_id": 2, "box": box(other), "confidence": 0.9, "keypoints": other, "keypoint_conf": np.full(17, .9)}]
    out = m.process(None, t, tracks)
    events += out["events"]
json.dumps(out); json.dumps(events)
assert len(events) == 1 and events[0]["track_id"] == 5 and events[0]["camera_id"] == "camA", events
assert out["tracks"][2]["falling"] is False
assert max(batches) == 2, "both people should be scored in one batch"
print("A passed:", events[0])

# B) Model 1 gives only boxes -> module runs pose and matches by overlap
class FakePose:
    def __init__(self): self.t = 0
    def predict(self, frame, **k):
        a = 0 if self.t < 3 else min(1, (self.t - 3) / .7) * np.pi / 2; self.t += 1 / 15
        kp = [body(a, 300, 240 + 100 * a / (np.pi / 2)), body(0, 900, 240)]
        boxes = torch.tensor([box(k) for k in kp]); kxy = torch.tensor(np.stack(kp), dtype=torch.float32)
        class B:
            xyxy, conf = boxes, torch.tensor([.9, .9])
            def __len__(self): return 2
        return [types.SimpleNamespace(boxes=B(),
                                      keypoints=types.SimpleNamespace(xy=kxy, conf=torch.full((2, 17), .9)))]
m2 = fall_module.FallDetectionModule(model_dir=d)
m2.pose, m2.pose_engine = FakePose(), "fake"
ev2 = []
for i in range(15 * 8):
    t = i / 15
    a = 0 if t < 3 else min(1, (t - 3) / .7) * np.pi / 2
    me = body(a, 300, 240 + 100 * a / (np.pi / 2)); jitter = np.array([3, -2, 4, 1])
    out = m2.process(np.zeros((480, 1280, 3), np.uint8), t,
                     [{"track_id": 9, "box": list(np.array(box(body(0, 900, 240))) + jitter)},
                      {"track_id": 4, "box": list(np.array(box(me)) + jitter)}])
    ev2 += out["events"]
assert [e["track_id"] for e in ev2] == [4], ev2
print("B passed:", ev2[0]["track_id"], ev2[0]["time"])
print("MODULE TESTS PASSED")
