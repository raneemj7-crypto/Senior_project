import unittest
from unittest.mock import patch
import numpy as np
from integrated_model2 import IntegratedModel2
from raneem_live.pipeline import RaneemTrackedPipeline


class ClassifierModesTest(unittest.TestCase):
    def test_parallel_shared_pose_and_independent_results(self):
        model = IntegratedModel2(mode='parallel', use_fast=False)
        frame = np.zeros((240, 320, 3), np.uint8)
        xy = np.column_stack((np.linspace(30, 90, 17), np.linspace(30, 180, 17))).astype('float32')
        tracks = [dict(track_id=i, box=[20, 20, 100, 200], keypoints=xy + i,
                       keypoint_conf=np.ones(17)) for i in (1, 2)]
        with patch.object(model.detector, '_poses', side_effect=AssertionError('duplicate pose')):
            for k in range(30):
                out = model.process(frame, k / 15, tracks)
        self.assertEqual(set(out['tracks'][1]['predictions']),
                         {'souad', 'raneem_visibility', 'raneem_fullbody'})
        self.assertIsNot(model.raneem.history[1], model.raneem.history[2])
        np.testing.assert_allclose(model.raneem.history[1][-1][1][:, 0], (xy[:, 0] + 1) / 320)
        self.assertEqual(out['tracks'][1]['fall_probability'], out['tracks'][1]['predictions']['souad']['fall_probability'])
        model.reset('new')
        self.assertFalse(model.raneem.history)
        self.assertEqual(model.raneem.camera_id, 'new')

    def test_boxes_only_share_one_pose_pass(self):
        model = IntegratedModel2(mode='parallel')
        frame = np.zeros((240, 320, 3), np.uint8)
        box = np.array([[20, 20, 100, 200]], dtype='float32')
        xy = np.ones((1, 17, 2), dtype='float32') * 50
        kc = np.ones((1, 17), dtype='float32')
        with patch.object(model.detector, '_poses', return_value=(box, xy, kc, np.ones(1))) as poses:
            out = model.process(frame, 0, [dict(track_id=1, box=box[0].tolist())])
        poses.assert_called_once()
        np.testing.assert_array_equal(out['tracks'][1]['keypoints'], xy[0])
        np.testing.assert_allclose(model.raneem.history[1][-1][1][:, :2], xy[0] / [320, 240])

    def test_single_raneem_does_not_load_souad_network(self):
        with patch('model2_fall_detection.fall_module.load_fall_model', side_effect=AssertionError):
            for mode in ('raneem_visibility', 'raneem_fullbody'):
                model = IntegratedModel2(mode=mode)
                self.assertIsNone(model.detector.core.predict)
                self.assertEqual(list(model.raneem.models), [mode.removeprefix('raneem_')])

    def test_transfer_moves_and_reset_clears_all_state(self):
        model = RaneemTrackedPipeline()
        model.history[1] = ['donor']
        model.history[2] = ['unrelated']
        model.counts[1] = 25
        model.last_emitted[(1, 'visibility')] = 1
        model.apply_tracking_events([dict(type='history_transferred', track_id=2, old_id=1)])
        self.assertEqual(model.history, {2: ['donor']})
        self.assertEqual(model.counts, {2: 25})
        self.assertEqual(model.last_emitted, {(2, 'visibility'): 1})
        model.apply_tracking_events([dict(type='history_reset', track_id=2)])
        self.assertFalse(model.history)
        self.assertFalse(model.counts)
        self.assertFalse(model.last_emitted)

    def test_stale_reused_id_starts_fresh(self):
        model = RaneemTrackedPipeline()
        frame = np.zeros((100, 100, 3), np.uint8)
        track = dict(track_id=1, box=[0,0,80,80], keypoints=np.ones((17,2)), keypoint_conf=np.ones(17))
        model.process(frame, 0, [track])
        model.process(frame, 3, [track])
        self.assertEqual(len(model.history[1]), 1)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            IntegratedModel2(mode='fusion')

class IndependentEvaluationTest(unittest.TestCase):
    def test_disagreeing_classifiers_are_scored_separately(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock
        import cv2
        from batch_evaluate import evaluate_sequence
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'fall' / 'sample'
            folder.mkdir(parents=True)
            path = folder / '1.png'
            cv2.imwrite(str(path), np.zeros((20, 20, 3), np.uint8))
            model = Mock()
            model.process.return_value = dict(tracks={1: dict(
                visibility='full_body', fall_probability=.1,
                diagnostics={'status': 'ready'}, predictions={
                    'souad': dict(fall_probability=.1, decision='non_fall', status='ready'),
                    'raneem_visibility': dict(fall_probability=.9, decision='fall', status='ready'),
                    'raneem_fullbody': dict(fall_probability=None, decision='abstain', status='missing_pose')})}, events=[])
            with patch('batch_evaluate.extract_tracks', return_value=[{'track_id': 1}]):
                row = evaluate_sequence(('fall', folder, [path]), root, 15, SimpleNamespace(), model)
            self.assertIsNone(row['error'])
            results = row['model_results']
            self.assertEqual(results['souad']['predicted_class'], 'non_fall')
            self.assertEqual(results['raneem_visibility']['predicted_class'], 'fall')
            self.assertIsNone(results['raneem_fullbody']['predicted_class'])
