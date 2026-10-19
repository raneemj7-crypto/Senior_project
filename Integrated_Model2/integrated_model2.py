"""Shared tracked poses with independent Souad and Raneem classifiers."""
from collections import Counter, defaultdict, deque
import numpy as np
from model2_fall_detection.fall_module import FallDetectionModule
from visibility.visibility_detection import detect_visibility


INFERENCE_MODES = ('souad', 'raneem_visibility', 'raneem_fullbody', 'parallel')


class IntegratedModel2:
    def __init__(self, camera_id="cam0", model_dir=None, pose_weights=None, use_fast=True,
                 mode='souad', checkpoint_dir=None):
        if mode not in INFERENCE_MODES:
            raise ValueError(f'mode must be one of {INFERENCE_MODES}')
        self.mode = mode
        self.raneem = None
        if mode != 'souad':
            from raneem_live.pipeline import RaneemTrackedPipeline
            self.raneem = RaneemTrackedPipeline(checkpoint_dir=checkpoint_dir,
                mode='parallel' if mode == 'parallel' else mode.removeprefix('raneem_'),
                camera_id=camera_id)
        self.detector = FallDetectionModule(camera_id=camera_id, model_dir=model_dir,
                                            pose_weights=pose_weights, use_fast=use_fast,
                                            enable_classifier=mode in ('souad', 'parallel'))
        self.visibility_history = defaultdict(lambda: deque(maxlen=30))

    def reset(self, camera_id=None):
        """Clear sequence state while retaining loaded pose and fall networks."""
        from model2_fall_detection.fall_core import LiveFallDetector
        self.detector.core = LiveFallDetector(self.detector.core.predict, self.detector.settings)
        self.visibility_history.clear()
        if getattr(self, 'raneem', None) is not None:
            self.raneem.reset()
            if camera_id is not None:
                self.raneem.camera_id = camera_id
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
        # Souad's core may transfer history after an ID switch. Mirror visibility.
        for new, old in self.detector.core.renamed.items():
            if old in self.visibility_history:
                previous = list(self.visibility_history.pop(old))
                self.visibility_history[new] = deque(previous[-30:], maxlen=30)
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
        for tid, state in out['tracks'].items():
            categories = self.visibility_history.get(int(tid), ())
            counts = Counter(categories)
            state['visibility'] = visibility.get(int(tid), 'uncertain')
            state['visibility_30frame'] = counts.most_common(1)[0][0] if counts else 'uncertain'
        for event in out['events']:
            event['visibility'] = out['tracks'].get(event['track_id'], {}).get('visibility', 'uncertain')
            event['visibility_30frame'] = out['tracks'].get(event['track_id'], {}).get('visibility_30frame', 'uncertain')
        mode = getattr(self, 'mode', 'souad')
        out['mode'] = mode
        out['primary_model'] = 'souad' if mode == 'parallel' else mode
        for event in out['events']:
            event['classifier'] = 'souad'
        for state in out['tracks'].values():
            state['predictions'] = {}
            if mode in ('souad', 'parallel'):
                state['predictions']['souad'] = {
                    'fall_probability': state['fall_probability'],
                    'falling': state['falling'],
                    'decision': ('fall' if state['falling'] else 'non_fall')
                                if state['fall_probability'] is not None else 'abstain',
                    'status': state.get('diagnostics', {}).get('status', 'unknown')}
        if getattr(self, 'raneem', None) is not None:
            self.raneem.apply_tracking_events(out.get('tracking_events', []))
            shared = []
            for tr in tracks:
                item = {'track_id': tr['track_id'], 'box': tr['box']}
                state = out['tracks'][int(tr['track_id'])]
                if state['keypoints'] is not None:
                    item.update(keypoints=state['keypoints'], keypoint_conf=state['keypoint_conf'])
                shared.append(item)
            result = self.raneem.process(frame, t, shared)
            for tid, rs in result['tracks'].items():
                state = out['tracks'][tid]
                for name in self.raneem.models:
                    pred = dict(rs['predictions'].get(name, {
                        'decision': 'abstain', 'fall_probability': None}))
                    pred['status'] = rs['status']
                    state['predictions']['raneem_' + name] = pred
                if mode != 'parallel':
                    pred = state['predictions'][mode]
                    state['fall_probability'] = pred['fall_probability']
                    state['falling'] = pred['decision'] == 'fall'
            for event in result['events']:
                event['classifier'] = event['model']
            out['events'].extend(result['events'])
        # In parallel, legacy scalar fields refer only to Souad, never fusion.
        return out
