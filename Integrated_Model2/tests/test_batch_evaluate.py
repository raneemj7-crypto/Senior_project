import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from collections import defaultdict, deque
import cv2
import numpy as np
from batch_evaluate import discover, evaluate_sequence, metrics, reset_tracking
from integrated_model2 import IntegratedModel2
from model2_fall_detection.fall_core import LiveFallDetector


class BatchTest(unittest.TestCase):
    def test_natural_sort_and_class_interleaving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ('fall', 'non_fall'):
                seq = root / label / 'seq-cam0'
                seq.mkdir(parents=True)
                for name in ('frame10.png', 'frame2.png', 'frame1.png'):
                    (seq / name).touch()
            items = discover(root)
            self.assertEqual([i[0] for i in items], ['fall', 'non_fall'])
            self.assertEqual([p.name for p in items[0][2]], ['frame1.png','frame2.png','frame10.png'])

    def test_metrics_errors_and_undefined_camera_specificity(self):
        def row(truth, prediction, events=0, error=None):
            return dict(ground_truth=truth, predicted_class=prediction, fall_events=events,
                        error=error, processing_fps=10)
        rows = [row('fall','fall',1), row('fall','non_fall'), row('non_fall','fall',3),
                row('non_fall','non_fall'), row('non_fall',None,error='bad frame')]
        out = metrics(rows)
        self.assertEqual([out[k] for k in ('TP','FP','TN','FN','errors')], [1]*5)
        self.assertEqual(out['false_alerts_per_non_fall_video'], 1.5)
        self.assertEqual(out['f1'], .5)
        self.assertIsNone(metrics(rows[:1])['specificity'])

    def test_reset_retains_networks_and_clears_state(self):
        model = IntegratedModel2.__new__(IntegratedModel2)
        predict = Mock()
        pose = object()
        settings = dict(clip_frames=30, clip_fps=15)
        core = LiveFallDetector(predict, settings)
        core.buffers[7] = ['old']
        core.alarm_until[7] = 100
        model.detector = SimpleNamespace(core=core, pose=pose, settings=settings)
        model.visibility_history = defaultdict(lambda: deque(maxlen=30), {7: deque(['full_body'])})
        model.reset('cam1')
        self.assertIs(model.detector.core.predict, predict)
        self.assertIs(model.detector.pose, pose)
        self.assertFalse(model.detector.core.buffers)
        self.assertFalse(model.detector.core.alarm_until)
        self.assertFalse(model.visibility_history)
        tracker = Mock()
        detector = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker], vid_path=['old']))
        reset_tracking(detector)
        tracker.reset.assert_called_once()
        self.assertEqual(detector.predictor.vid_path, [None])

    def test_events_not_max_probability_and_corrupt_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'fall' / 'seq-cam0'
            folder.mkdir(parents=True)
            frame = folder / '001.png'
            cv2.imwrite(str(frame), np.zeros((20,20,3), np.uint8))
            model = Mock()
            model.process.return_value = dict(tracks={1: dict(visibility='full_body',
                fall_probability=.99, diagnostics={'status':'ready'})}, events=[])
            item = ('fall', folder, [frame])
            with patch('batch_evaluate.extract_tracks', return_value=[{'track_id':1}]):
                row = evaluate_sequence(item, root, 15, SimpleNamespace(), model)
                self.assertEqual(row['predicted_class'], 'non_fall')
                self.assertIsNone(row['detection_latency'])
                model.process.return_value['events'] = [{'time': 2.0}]
                row = evaluate_sequence(item, root, 15, SimpleNamespace(), model,
                                        onsets={'fall/seq-cam0': 1.5})
                self.assertEqual(row['detection_latency'], .5)
                frame.write_bytes(b'corrupt')
                row = evaluate_sequence(item, root, 15, SimpleNamespace(), model)
                self.assertIn('Unreadable PNG', row['error'])
                self.assertIsNone(row['predicted_class'])

if __name__ == '__main__':
    unittest.main()
