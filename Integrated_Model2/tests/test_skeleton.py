"""Synthetic fixtures verify data plumbing; real-video results are separate."""
import unittest
from unittest.mock import Mock
import numpy as np
from integrated_model2 import IntegratedModel2
from model2_fall_detection.fall_module import FallDetectionModule
from model2_fall_detection.fall_core import LiveFallDetector
from test_video import draw_skeleton, state_color
from collections import defaultdict, deque


class SkeletonTest(unittest.TestCase):
    def setUp(self):
        self.settings = dict(clip_frames=30, clip_fps=15, threshold=.7,
                             joint_ok=.3, max_missing_frac=.3)
        self.predict = Mock(side_effect=lambda clips: np.full(len(clips), .8))
        self.detector = FallDetectionModule.__new__(FallDetectionModule)
        self.detector.settings = self.settings
        self.detector.camera_id = 'test'
        self.detector.core = LiveFallDetector(self.predict, self.settings)
        self.xy = np.arange(34, dtype=np.float32).reshape(17, 2) + 10
        self.kc = np.ones(17, dtype=np.float32)

    def test_multiple_people_match_reversed_pose_order_and_missing(self):
        boxes = [[0, 0, 50, 100], [100, 0, 150, 100], [200, 0, 250, 100]]
        self.detector._poses = Mock(return_value=(np.array(boxes[1::-1]),
            np.array([self.xy + 100, self.xy]), np.array([self.kc, self.kc]), np.ones(2)))
        model = IntegratedModel2.__new__(IntegratedModel2)
        model.detector = self.detector
        model.visibility_history = defaultdict(lambda: deque(maxlen=30))
        out = model.process(None, 0, [dict(track_id=i, box=b) for i, b in enumerate(boxes)])
        self.detector._poses.assert_called_once()
        np.testing.assert_array_equal(out['tracks'][0]['keypoints'], self.xy)
        np.testing.assert_array_equal(out['tracks'][1]['keypoints'], self.xy + 100)
        self.assertIsNone(out['tracks'][2]['keypoints'])
        self.assertEqual(out['tracks'][2]['diagnostics']['status'], 'missing_skeleton')
        self.assertEqual(out['tracks'][0]['visibility'], 'full_body')
        out['tracks'][0]['keypoints'][0][0] = -999
        self.assertEqual(self.detector.core.buffers[0][0][1][0, 0], 10)

    def test_history_prediction_and_id_transfer(self):
        for frame in range(31):
            tid = 4 if frame < 30 else 8
            out = self.detector.process(None, frame / 15, [dict(track_id=tid,
                box=[0, 0, 50, 100], keypoints=self.xy, keypoint_conf=self.kc)])
            if frame == 4:
                self.assertEqual(out['tracks'][tid]['diagnostics']['status'], 'insufficient_temporal_history')
        self.assertEqual(out['track_id_changes'], {8: 4})
        self.assertTrue(out['tracks'][8]['falling'])
        self.assertEqual(self.predict.call_args[0][0].shape, (1, 30, 17, 3))

    def test_hidden_joints_and_colors(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        state = dict(keypoints=self.xy, keypoint_conf=np.zeros(17))
        draw_skeleton(frame, state, (0, 200, 0), .3)
        self.assertFalse(frame.any())
        state['keypoint_conf'] = self.kc
        draw_skeleton(frame, state, (0, 200, 0), .3)
        self.assertTrue(frame.any())
        self.assertEqual(state_color(dict(fall_probability=.6)), (0, 165, 255))
        self.assertEqual(state_color(dict(falling=True)), (0, 0, 255))

if __name__ == '__main__':
    unittest.main()
