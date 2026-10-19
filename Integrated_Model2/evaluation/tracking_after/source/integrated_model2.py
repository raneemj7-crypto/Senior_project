"""Souad's live fall detector augmented with Raneem's visibility assessment.

Visibility is a diagnostic signal, not a fall/non-fall decision. This wrapper
never changes Souad's fall probability, threshold, or alert logic.
"""
from collections import Counter, defaultdict, deque
import numpy as np
from model2_fall_detection.fall_module import FallDetectionModule
from visibility.visibility_detection import detect_visibility


class IntegratedModel2:
    def __init__(self, camera_id="cam0", model_dir=None, pose_weights=None, use_fast=True):
        self.detector = FallDetectionModule(camera_id=camera_id, model_dir=model_dir,
                                            pose_weights=pose_weights, use_fast=use_fast)
        self.visibility_history = defaultdict(lambda: deque(maxlen=30))

    def reset(self, camera_id=None):
        """Clear sequence state while retaining loaded pose and fall networks."""
        from model2_fall_detection.fall_core import LiveFallDetector
        self.detector.core = LiveFallDetector(self.detector.core.predict, self.detector.settings)
        self.visibility_history.clear()
        if camera_id is not None:
            self.detector.camera_id = camera_id

    def process(self, frame, t, tracks):
        """Accept Souad's Model 1 track format. Returns original fields + visibility.

        Visibility consumes the exact current-frame pose returned by Souad's
        detector, so unmatched tracks never cause a second pose inference.
        """
        out = self.detector.process(frame, t, tracks)
        for event in out.get('tracking_events', []):
            if event['type'] == 'history_reset':
                self.visibility_history.pop(event['track_id'], None)
        visibility = {}
        for tid, state in out['tracks'].items():
            if state.get('keypoints') is None:
                visibility[tid] = 'uncertain'
                continue
            xy = np.asarray(state['keypoints'], dtype=np.float32)
            kc = np.asarray(state['keypoint_conf'], dtype=np.float32)
            cat = detect_visibility(np.column_stack((xy, kc)))
            self.visibility_history[tid].append(cat)
            visibility[tid] = cat
        # Souad's core may transfer history after an ID switch. Mirror visibility.
        for new, old in self.detector.core.renamed.items():
            if old in self.visibility_history:
                previous = list(self.visibility_history.pop(old))
                current = list(self.visibility_history[new])
                self.visibility_history[new] = deque((previous + current)[-30:], maxlen=30)
        for tid, state in out['tracks'].items():
            categories = self.visibility_history.get(int(tid), ())
            counts = Counter(categories)
            state['visibility'] = visibility.get(int(tid), 'uncertain')
            state['visibility_30frame'] = counts.most_common(1)[0][0] if counts else 'uncertain'
        for event in out['events']:
            event['visibility'] = out['tracks'].get(event['track_id'], {}).get('visibility', 'uncertain')
            event['visibility_30frame'] = out['tracks'].get(event['track_id'], {}).get('visibility_30frame', 'uncertain')
        return out
