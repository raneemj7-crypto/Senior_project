"""
Themam - Model 2 core: fall detection from body skeletons (YOLOv8n-pose + ST-GCN++).

Shared by the laptop demo (live_fall_demo.py) and the integration module (fall_module.py).
It keeps each person's last 2 seconds of skeletons, asks the ST-GCN++ model for a fall
probability, and decides when a fall alarm starts.

Do not change normalize_clip(): it must match the training exactly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

ALARM_HOLD_S = 3.0          # a person counts as "falling" for 3 s after a high probability
PREDICT_EVERY_S = 0.1       # ask the model at most 10 times per second per person
PERSON_CONF = 0.5           # a track counts as a real person only after it was seen clearly
PERSON_JOINTS = 10          #   in PERSON_FRAMES frames: box confidence >= 0.5 and
PERSON_FRAMES = 5           #   >= 10 joints with confidence >= 0.5 (pillows, clothes, shadows rarely manage that)
HANDOVER_S = 1.0            # a new track ID that appears where another one was lost less than
HANDOVER_DIST = 1.0         #   1 s ago (within 1 box size) is treated as the same person
FORGET_S = 2.0              # forget a person not seen for 2 s

COCO_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
              (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


# ---------------- model files ----------------
def find_model_dir(model_dir=None):
    """The folder with stgcnpp_fall_v1_laptop.pt/.json: the given one, ./models, or this folder."""
    for d in ([Path(model_dir)] if model_dir else []) + [HERE / "models", HERE]:
        if (d / "stgcnpp_fall_v1_laptop.pt").exists() and (d / "stgcnpp_fall_v1_laptop.json").exists():
            return d
    raise SystemExit("Model files not found. Put stgcnpp_fall_v1_laptop.pt and stgcnpp_fall_v1_laptop.json "
                     "in the 'models' folder (download them from Drive -> Themam_STGCN/checkpoints).")


def load_fall_model(model_dir=None):
    """Returns (settings, predict). predict(clips) takes an array (N, 30, 17, 3) -> N fall probabilities."""
    d = find_model_dir(model_dir)
    settings = json.loads((d / "stgcnpp_fall_v1_laptop.json").read_text())
    net = torch.jit.load(str(d / "stgcnpp_fall_v1_laptop.pt"), map_location="cpu").eval()

    @torch.no_grad()
    def predict(clips):
        clips = np.asarray(clips, dtype=np.float32)
        if len(clips) == 0:
            return np.zeros(0, np.float32)
        return net(torch.from_numpy(clips)).numpy()

    return settings, predict


def load_pose(weights="yolov8n-pose.pt", imgsz=640, use_fast=True):
    """YOLOv8n-pose: GPU if available; otherwise the OpenVINO version (about 3x faster on a CPU)."""
    from ultralytics import YOLO
    if use_fast and not torch.cuda.is_available():
        try:
            import openvino  # noqa: F401
            fast_dir = HERE / f"{Path(weights).stem}_{imgsz}_openvino_model"   # name must end with _openvino_model
            if not fast_dir.exists():
                print("Preparing the fast pose model (one time only, about 1 minute)...")
                exported = Path(YOLO(weights).export(format="openvino", imgsz=imgsz, verbose=False))
                exported.rename(fast_dir)
            model = YOLO(str(fast_dir), task="pose")
            model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=imgsz, verbose=False)   # warm-up check
            return model, "OpenVINO"
        except ImportError:
            print("\n*** SLOW MODE: OpenVINO is not installed for this Python. ***")
            print("To make the program about 3x faster, close it and run this once:")
            print(f'    "{sys.executable}" -m pip install openvino\n')
        except Exception as error:
            print("Fast mode not available, using normal mode:", error)
    return YOLO(weights), "PyTorch"


# ---------------- same normalization as training (do not change) ----------------
def normalize_clip(xy, kc, valid, joint_ok):
    good = (kc >= joint_ok) & valid[:, None]
    centres, sizes = [], []
    for t in range(len(xy)):
        pts = xy[t][good[t]]
        if len(pts) >= 2:
            centres.append(pts.mean(axis=0))
            sizes.append(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    if len(sizes) < 3:
        return None
    scale = float(np.median(sizes))
    if scale < 5:
        return None
    out = np.zeros((len(xy), 17, 3), dtype=np.float32)
    out[..., :2] = np.clip((xy - np.mean(centres, axis=0)) / scale, -5, 5)
    out[..., 2] = kc
    out[~valid] = 0
    return out


def box_centre_size(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2]), max(x2 - x1, y2 - y1, 1e-6)


class LiveFallDetector:
    """Keeps each person's last 2 seconds of skeletons and asks ST-GCN++ for a fall probability.

    update() is called once per frame with every tracked person in that frame. All people
    who need a new prediction are scored together in one batch (cheaper in crowds).
    If a person gets a NEW track ID (common during a fall), the new ID takes over the old
    person's history when it appears where the old one was just lost.

    After update():
      self.new_falls : track IDs whose fall alarm STARTED in this frame (send an event for these)
      self.renamed   : {new ID: old ID} hand-overs that happened in this frame
    """

    def __init__(self, predict, settings):
        self.predict, self.s = predict, settings
        self.window = (settings["clip_frames"] - 1) / settings["clip_fps"]
        self.buffers, self.last_seen, self.alarm_until = {}, {}, {}
        self.last_pred = {}                        # id -> (time, probability)
        self.confirmed = set()                     # ids that were seen clearly as a person
        self.clear_frames = {}                     # id -> number of frames it looked clearly like a person
        self.alarm_prev = {}                       # id -> was this person "falling" in the previous frame
        self.renamed, self.new_falls = {}, []

    def _parent(self, t, box, current):
        best, best_d = None, HANDOVER_DIST
        c_new, _ = box_centre_size(box)
        for old, buf in self.buffers.items():
            if old in current or not buf or t - self.last_seen[old] > HANDOVER_S:
                continue
            c_old, s_old = box_centre_size(buf[-1][3])
            d = np.linalg.norm(c_new - c_old) / s_old
            if d < best_d:
                best, best_d = old, d
        return best

    def _clip(self, t, buf):
        if not buf or t - buf[0][0] < 0.9 * self.window:
            return None
        times = np.array([b[0] for b in buf])
        want = t - self.window + np.arange(self.s["clip_frames"]) / self.s["clip_fps"]
        idx = np.abs(times[None, :] - want[:, None]).argmin(axis=1)
        valid = np.abs(times[idx] - want) <= 0.2
        if (~valid).mean() > self.s["max_missing_frac"]:
            return None
        xy = np.stack([buf[i][1] for i in idx]).astype(np.float32)
        kc = np.stack([buf[i][2] for i in idx]).astype(np.float32)
        return normalize_clip(xy, kc, valid, self.s["joint_ok"])

    def update(self, t, ids, boxes, kxy, kconf, confs):
        """t: time in seconds. For each person: track id, box [x1,y1,x2,y2], 17 keypoints (x, y)
        in pixels, 17 keypoint confidences, box confidence.
        Returns {track id: (fall probability or None while waiting, falling now True/False)}."""
        current = set(int(i) for i in ids)
        self.renamed, self.new_falls = {}, []
        for tid, box, xy, kc, conf in zip(ids, boxes, kxy, kconf, confs):
            tid = int(tid)
            if tid not in self.buffers:
                parent = self._parent(t, box, current)
                self.buffers[tid] = self.buffers.pop(parent) if parent is not None else []
                if parent is not None:
                    self.renamed[tid] = parent
                    self.alarm_until[tid] = self.alarm_until.pop(parent, -1)
                    self.alarm_prev[tid] = self.alarm_prev.pop(parent, False)
                    self.last_seen.pop(parent, None)
                    self.last_pred.pop(parent, None)
                    self.clear_frames[tid] = self.clear_frames.pop(parent, 0)
                    if parent in self.confirmed:
                        self.confirmed.discard(parent)
                        self.confirmed.add(tid)
            if conf >= PERSON_CONF and (np.asarray(kc) >= 0.5).sum() >= PERSON_JOINTS:
                self.clear_frames[tid] = self.clear_frames.get(tid, 0) + 1
                if self.clear_frames[tid] >= PERSON_FRAMES:
                    self.confirmed.add(tid)
            self.buffers[tid].append((t, np.asarray(xy, np.float32), np.asarray(kc, np.float32), np.asarray(box)))
            self.last_seen[tid] = t
        for tid in list(self.buffers):
            self.buffers[tid] = [b for b in self.buffers[tid] if t - b[0] <= self.window + 0.5]
            if t - self.last_seen.get(tid, t) > FORGET_S:
                for d in (self.buffers, self.last_seen, self.alarm_until, self.last_pred,
                          self.clear_frames, self.alarm_prev):
                    d.pop(tid, None)
                self.confirmed.discard(tid)

        # people who need a new prediction -> one batch
        people = sorted(tid for tid in current if tid in self.confirmed)   # others: not (yet) a clear person
        ask, clips = [], []
        for tid in people:
            last = self.last_pred.get(tid)
            if last is None or t - last[0] >= PREDICT_EVERY_S:
                clip = self._clip(t, self.buffers[tid])
                if clip is not None:
                    ask.append(tid)
                    clips.append(clip)
        if clips:
            for tid, p in zip(ask, self.predict(np.stack(clips))):
                self.last_pred[tid] = (t, float(p))

        results = {}
        for tid in people:
            last = self.last_pred.get(tid)
            prob = last[1] if last is not None and t - last[0] < PREDICT_EVERY_S else None
            if prob is not None and prob >= self.s["threshold"]:
                self.alarm_until[tid] = max(self.alarm_until.get(tid, -1), t + ALARM_HOLD_S)
            falling = t <= self.alarm_until.get(tid, -1)
            if falling and not self.alarm_prev.get(tid, False):
                self.new_falls.append(tid)
            self.alarm_prev[tid] = falling
            results[tid] = (prob, falling)
        return results
