"""
Themam - Model 2 integration module (fall / collapse detection).

Model 1 (detection + tracking) gives, for every frame: the frame, its time, and the tracked
people (track ID + box). This module adds the skeletons, keeps each person's last 2 seconds,
and returns the fall probability of every person plus a FALL EVENT when a fall starts.

    from fall_module import FallDetectionModule

    fall = FallDetectionModule(camera_id="cam1")              # loads the models once
    out = fall.process(frame, t, tracks)                      # call once per frame
    for event in out["events"]:
        send_to_dashboard(event)                              # a JSON-ready dict

tracks = list of dicts, one per person, from Model 1:
    {"track_id": 3, "box": [x1, y1, x2, y2]}                  # pixels of THIS frame
    optional: "confidence": 0.9                               # detector score of the box
    optional: "keypoints": 17 x [x, y], "keypoint_conf": 17   # if Model 1 already runs a pose model
                                                              #   (then no second pose model is run)
t = time of the frame in seconds (must increase; the real capture time, not the frame number)

Returned:
    {"time": t,
     "tracks": {3: {"fall_probability": 0.04, "falling": False}, ...},   # None = not enough history yet
     "events": [{"event": "fall_detected", "camera_id": "cam1", "track_id": 3, "time": 12.4,
                 "fall_probability": 0.93, "box": [x1, y1, x2, y2], "threshold": 0.7, "model": "..."}]}
"""

import numpy as np

try:                                  # works both as a package (from the repo root) and as plain files
    from .fall_core import LiveFallDetector, load_fall_model, load_pose
except ImportError:
    from fall_core import LiveFallDetector, load_fall_model, load_pose

MATCH_IOU = 0.3             # a skeleton belongs to a Model 1 box if their boxes overlap at least this much


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_skeletons(track_boxes, pose_boxes):
    """Greedy best-overlap matching. Returns {track index: pose index}."""
    pairs = sorted(((iou(tb, pb), i, j) for i, tb in enumerate(track_boxes) for j, pb in enumerate(pose_boxes)),
                   reverse=True)
    used_t, used_p, match = set(), set(), {}
    for score, i, j in pairs:
        if score < MATCH_IOU:
            break
        if i not in used_t and j not in used_p:
            match[i] = j
            used_t.add(i)
            used_p.add(j)
    return match


class FallDetectionModule:
    def __init__(self, model_dir=None, camera_id="cam0", pose_weights=None, imgsz=640, use_fast=True):
        self.settings, predict = load_fall_model(model_dir)
        self.camera_id = camera_id
        self.pose_weights = pose_weights or self.settings["pose_model"]
        self.imgsz, self.use_fast = imgsz, use_fast
        self.pose, self.pose_engine = None, None             # loaded only if Model 1 gives no keypoints
        self.core = LiveFallDetector(predict, self.settings)

    def _poses(self, frame):
        if self.pose is None:
            self.pose, self.pose_engine = load_pose(self.pose_weights, self.imgsz, self.use_fast)
        device = "cpu" if self.pose_engine == "OpenVINO" else None
        r = self.pose.predict(frame, imgsz=self.imgsz, conf=self.settings["detector_conf"],
                              device=device, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return np.zeros((0, 4)), np.zeros((0, 17, 2)), np.zeros((0, 17)), np.zeros(0)
        return (r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(),
                r.keypoints.conf.cpu().numpy(), r.boxes.conf.cpu().numpy())

    def process(self, frame, t, tracks):
        ids, boxes, kxy, kconf, confs = [], [], [], [], []
        need_pose = [k for k, tr in enumerate(tracks) if "keypoints" not in tr]
        if need_pose:
            p_boxes, p_xy, p_kc, p_conf = self._poses(frame)
            match = match_skeletons([tracks[k]["box"] for k in need_pose], p_boxes)
        for k, tr in enumerate(tracks):
            if "keypoints" in tr:
                xy = np.asarray(tr["keypoints"], np.float32).reshape(17, 2)
                kc = np.asarray(tr.get("keypoint_conf", np.ones(17)), np.float32).reshape(17)
                conf = tr.get("confidence", 1.0)
            else:
                j = match.get(need_pose.index(k))
                if j is None:
                    continue                                 # no skeleton for this person in this frame
                xy, kc = p_xy[j], p_kc[j]
                conf = tr.get("confidence", float(p_conf[j]))
            ids.append(int(tr["track_id"]))
            boxes.append(np.asarray(tr["box"], np.float32))
            kxy.append(xy)
            kconf.append(kc)
            confs.append(conf)
        results = self.core.update(t, ids, boxes, kxy, kconf, confs)
        box_of = dict(zip(ids, boxes))
        events = [{
            "event": "fall_detected",
            "camera_id": self.camera_id,
            "track_id": tid,
            "time": round(float(t), 3),
            "fall_probability": round(float(results[tid][0] or 0.0), 4),
            "box": [round(float(v), 1) for v in box_of[tid]],
            "threshold": self.settings["threshold"],
            "model": self.settings.get("trained_from", "stgcnpp_fall_v1"),
        } for tid in self.core.new_falls]
        return {
            "time": float(t),
            "tracks": {tid: {"fall_probability": None if p is None else round(float(p), 4), "falling": bool(f)}
                       for tid, (p, f) in results.items()},
            "events": events,
        }
