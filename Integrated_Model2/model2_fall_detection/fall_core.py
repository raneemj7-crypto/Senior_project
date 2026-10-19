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

try:
    from .tracking import continuity_score, unique_handover, HANDOVER_MAX_GAP
except ImportError:
    from tracking import continuity_score, unique_handover, HANDOVER_MAX_GAP

HERE = Path(__file__).resolve().parent

ALARM_HOLD_S = 3.0          # a person counts as "falling" for 3 s after a high probability
PREDICT_EVERY_S = 0.1       # ask the model at most 10 times per second per person
PERSON_CONF = 0.5           # a track counts as a real person only after it was seen clearly
PERSON_JOINTS = 10          #   in PERSON_FRAMES frames: box confidence >= 0.5 and
PERSON_FRAMES = 5           #   >= 10 joints with confidence >= 0.5 (pillows, clothes, shadows rarely manage that)
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
        self.coexisted = {}
        self.previous_present = set()
        self.tracking_events = []

    def _forget(self, tid):
        for data in (self.buffers, self.last_seen, self.alarm_until, self.last_pred,
                     self.clear_frames, self.alarm_prev):
            data.pop(tid, None)
        self.confirmed.discard(tid)

    def _handover(self, t, observations, present):
        candidates, evidence = [], {}
        for tid, box, xy, kc, _ in observations:
            if tid in self.buffers:
                continue
            for old, buf in self.buffers.items():
                if old in present or not buf or t - self.last_seen[old] > HANDOVER_MAX_GAP:
                    continue
                if frozenset((tid, old)) in self.coexisted:
                    evidence[tid, old] = {'reason': 'previously_coexisting_ids'}
                    continue
                score, detail = continuity_score(t, buf, box, xy, kc)
                evidence[tid, old] = detail
                if score is not None:
                    candidates.append((score, tid, old))
        selected = unique_handover(candidates)
        for (tid, old), detail in evidence.items():
            self.tracking_events.append(dict(type='handover_candidate', track_id=tid, old_id=old,
                accepted=selected.get(tid) == old, **detail))
        return selected

    def _clip(self, t, buf):
        self.clip_status = "insufficient_temporal_history"
        if not buf or t - buf[0][0] < 0.9 * self.window:
            return None
        times = np.array([b[0] for b in buf])
        want = t - self.window + np.arange(self.s["clip_frames"]) / self.s["clip_fps"]
        idx = np.abs(times[None, :] - want[:, None]).argmin(axis=1)
        valid = np.abs(times[idx] - want) <= 0.2
        self.clip_status = "missing_temporal_samples"
        if (~valid).mean() > self.s["max_missing_frac"]:
            return None
        xy = np.stack([buf[i][1] for i in idx]).astype(np.float32)
        kc = np.stack([buf[i][2] for i in idx]).astype(np.float32)
        clip = normalize_clip(xy, kc, valid, self.s["joint_ok"])
        self.clip_status = "ready" if clip is not None else "low_confidence_or_degenerate_skeleton"
        return clip

    def update(self, t, ids, boxes, kxy, kconf, confs, present_ids=None):
        """t: time in seconds. For each person: track id, box [x1,y1,x2,y2], 17 keypoints (x, y)
        in pixels, 17 keypoint confidences, box confidence.
        Returns {track id: (fall probability or None while waiting, falling now True/False)}."""
        current = set(int(i) for i in ids)
        present = current if present_ids is None else set(map(int, present_ids))
        observations = list(zip(ids, boxes, kxy, kconf, confs))
        self.renamed, self.new_falls, self.tracking_events = {}, [], []
        for tid in sorted(self.previous_present - present):
            self.tracking_events.append(dict(type='tracking_interruption', track_id=tid))
        for tid in sorted(present - self.previous_present):
            self.tracking_events.append(dict(type='track_appeared', track_id=tid))
        self.previous_present = present
        # Expire BEFORE append: a stale reused ID must not revive old alarms/history.
        for tid in list(self.buffers):
            if t - self.last_seen[tid] > FORGET_S:
                self._forget(tid)
                self.tracking_events.append(dict(type='history_reset', track_id=tid, reason='expired'))
        self.coexisted = {pair: seen for pair, seen in self.coexisted.items() if t-seen <= FORGET_S}
        for i in present:
            for j in present:
                if i < j:
                    self.coexisted[frozenset((i, j))] = t
        reset_ids = set()
        for tid, box, xy, kc, _ in observations:
            if tid in self.buffers and t - self.last_seen[tid] > .2 + 1e-6:
                score, detail = continuity_score(t, self.buffers[tid], box, xy, kc)
                self.tracking_events.append(dict(type='pose_resumed', track_id=tid, **detail))
                if score is None:
                    self._forget(tid)
                    reset_ids.add(tid)
                    self.tracking_events.append(dict(type='history_reset', track_id=tid, reason=detail['reason']))
        selected = self._handover(t, [o for o in observations if o[0] not in reset_ids], present)
        quality = {}
        for tid, box, xy, kc, conf in observations:
            tid = int(tid)
            if tid not in self.buffers:
                parent = selected.get(tid)
                self.buffers[tid] = self.buffers.pop(parent) if parent is not None else []
                if parent is not None:
                    self.renamed[tid] = parent
                    self.tracking_events.append(dict(type='history_transferred', track_id=tid, old_id=parent,
                        history_frames=len(self.buffers[tid]), gap_seconds=round(t-self.last_seen[parent], 4)))
                    self.alarm_until[tid] = self.alarm_until.pop(parent, -1)
                    self.alarm_prev[tid] = self.alarm_prev.pop(parent, False)
                    self.last_seen.pop(parent, None)
                    self.last_pred.pop(parent, None)
                    self.clear_frames[tid] = self.clear_frames.pop(parent, 0)
                    if parent in self.confirmed:
                        self.confirmed.discard(parent)
                        self.confirmed.add(tid)
                    # Carry evidence that this identity coexisted with other people.
                    for pair, seen in list(self.coexisted.items()):
                        if parent in pair:
                            self.coexisted[frozenset(tid if i == parent else i for i in pair)] = seen
                else:
                    self.tracking_events.append(dict(type='history_started', track_id=tid))
            previous = self.buffers[tid][-1] if self.buffers[tid] else None
            kc_array = np.asarray(kc)
            mean_conf = float(np.mean(kc_array))
            quality[tid] = dict(mean_joint_confidence=round(mean_conf, 4),
                confidence_change=None if previous is None else round(mean_conf-float(np.mean(previous[2])), 4),
                missing_joint_indices=np.flatnonzero(kc_array < self.s['joint_ok']).tolist(),
                pose_gap_seconds=None if previous is None else round(t-previous[0], 4))
            if conf >= PERSON_CONF and (np.asarray(kc) >= 0.5).sum() >= PERSON_JOINTS:
                self.clear_frames[tid] = self.clear_frames.get(tid, 0) + 1
                if self.clear_frames[tid] >= PERSON_FRAMES:
                    self.confirmed.add(tid)
            self.buffers[tid].append((t, np.asarray(xy, np.float32), np.asarray(kc, np.float32), np.asarray(box)))
            self.last_seen[tid] = t
        for tid in list(self.buffers):
            self.buffers[tid] = [b for b in self.buffers[tid] if t - b[0] <= self.window + 0.5]

        self.diagnostics = {
            tid: {"status": "person_not_confirmed" if tid not in self.confirmed else "prediction_cached",
                  "history_seconds": round(t - self.buffers[tid][0][0], 3),
                  "history_frames": len(self.buffers[tid]),
                  "clear_frames": self.clear_frames.get(tid, 0), **quality[tid]} for tid in current
        }
        # people who need a new prediction -> one batch
        people = sorted(tid for tid in current if tid in self.confirmed and self.predict is not None)   # others: not (yet) a clear person
        ask, clips = [], []
        for tid in people:
            last = self.last_pred.get(tid)
            if last is None or t - last[0] >= PREDICT_EVERY_S:
                clip = self._clip(t, self.buffers[tid])
                self.diagnostics[tid]["status"] = self.clip_status
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
