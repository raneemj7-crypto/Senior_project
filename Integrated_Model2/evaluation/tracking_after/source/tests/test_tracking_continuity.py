"""Synthetic identity regressions independent of the evaluation sequence labels."""
import unittest
from unittest.mock import Mock
import numpy as np
from model2_fall_detection.fall_core import LiveFallDetector
from model2_fall_detection.fall_module import FallDetectionModule, match_skeletons


class ContinuityTest(unittest.TestCase):
    def setUp(self):
        self.settings = dict(clip_frames=30, clip_fps=15, threshold=.7, joint_ok=.3, max_missing_frac=.3)
        self.predict = Mock(side_effect=lambda clips: np.zeros(len(clips)))
        self.core = LiveFallDetector(self.predict, self.settings)
        self.xy = np.column_stack((np.linspace(40, 70, 17), np.linspace(30, 170, 17))).astype(np.float32)
        self.kc = np.ones(17, np.float32)
        self.box = np.array([20, 10, 100, 200], np.float32)

    def update(self, t, people, present=None):
        # Each person is (ID, horizontal offset, optional pose/conf override).
        self.core.update(t, [p[0] for p in people],
            [self.box + [p[1], 0, p[1], 0] for p in people],
            [p[2] if len(p) > 2 else self.xy + [p[1], 0] for p in people],
            [p[3] if len(p) > 3 else self.kc for p in people],
            [1.] * len(people), present_ids=present)

    def test_stable_tracking_never_resets_and_preserves_input(self):
        for i in range(35):
            self.update(i/15, [(1, i), (2, 250+i)])
            self.assertFalse(self.core.renamed)
            self.assertFalse(any(e['type']=='history_reset' for e in self.core.tracking_events))
        np.testing.assert_array_equal(self.core.buffers[1][-1][1], self.xy+[34,0])
        self.assertEqual(self.predict.call_args[0][0].shape, (2, 30, 17, 3))

    def test_short_gap_handover_preserves_confirmation_and_alarm(self):
        for i in range(10): self.update(i/15, [(1, i)])
        self.core.alarm_until[1] = 3
        self.core.alarm_prev[1] = True
        self.update(.7, [])
        self.update(.8, [(9, 12)])
        self.assertEqual(self.core.renamed, {9: 1})
        self.assertEqual(len(self.core.buffers[9]), 11)
        self.assertIn(9, self.core.confirmed)
        self.assertEqual(self.core.alarm_until[9], 3)
        self.assertNotIn(1, self.core.buffers)

    def test_legitimate_body_rotation_does_not_break_handover(self):
        # A 90-degree fall-like rigid rotation must not be mistaken for another
        # identity. This is an association transform, never classifier input.
        self.update(0, [(1, 0)])
        centre = self.xy.mean(0)
        rotated = (self.xy-centre) @ np.array([[0., -1.], [1., 0.]]) + centre
        self.update(.4, [(9, 0, rotated, self.kc)])
        self.assertEqual(self.core.renamed, {9: 1})
        np.testing.assert_array_equal(self.core.buffers[9][-1][1], rotated.astype(np.float32))

    def test_present_person_without_pose_cannot_be_donor(self):
        self.update(0, [(1, 0)])
        self.update(.1, [(2, 0)], present=[1, 2])
        self.assertFalse(self.core.renamed)
        self.assertEqual(len(self.core.buffers[2]), 1)

    def test_previously_coexisting_ids_cannot_merge(self):
        self.update(0, [(1, 0)], present=[1, 2])
        self.update(.1, [(2, 0)])
        self.assertFalse(self.core.renamed)
        self.assertTrue(any(e.get('reason')=='previously_coexisting_ids' for e in self.core.tracking_events))

    def test_ambiguous_two_donors_and_two_receivers_abstain(self):
        self.update(0, [(1, 0), (2, 10)])
        self.update(.1, [(3, 5)])
        self.assertFalse(self.core.renamed)
        self.setUp()
        self.update(0, [(1, 0)])
        self.update(.1, [(2, 0), (3, 5)])
        self.assertFalse(self.core.renamed)

    def test_disjoint_people_handover_is_order_independent(self):
        self.update(0, [(1, 0), (2, 300)])
        self.update(.1, [(20, 300), (10, 0)])
        self.assertEqual(self.core.renamed, {20: 2, 10: 1})

    def test_keypoint_disagreement_and_weak_pose_cannot_merge(self):
        for xy, kc in [(self.xy+[0,180], self.kc), (self.xy, np.zeros(17))]:
            self.setUp()
            self.update(0, [(1, 0)])
            self.update(.1, [(2, 0, xy, kc)])
            self.assertFalse(self.core.renamed)

    def test_long_gap_same_id_does_not_revive_history_or_alarm(self):
        self.update(0, [(1, 0)])
        self.core.alarm_until[1] = 9
        self.update(3, [(1, 0)])
        self.assertEqual(len(self.core.buffers[1]), 1)
        self.assertNotIn(1, self.core.alarm_until)
        self.assertTrue(any(e['type']=='history_reset' for e in self.core.tracking_events))

    def test_same_id_spatial_jump_after_interruption_resets(self):
        self.update(0, [(1, 0)])
        self.update(.5, [(1, 500)])
        self.assertEqual(len(self.core.buffers[1]), 1)
        self.assertFalse(self.core.renamed)

    def test_ambiguous_pose_boxes_not_assigned_to_either_person(self):
        self.assertEqual(match_skeletons([self.box, self.box+1], [self.box]), {})

    def test_invalid_pose_never_enters_buffer(self):
        detector = FallDetectionModule.__new__(FallDetectionModule)
        detector.core, detector.settings, detector.camera_id = self.core, self.settings, 'test'
        xy = self.xy.copy(); xy[0,0] = np.nan
        out = detector.process(None, 0, [dict(track_id=1, box=self.box, keypoints=xy, keypoint_conf=self.kc)])
        self.assertEqual(out['tracks'][1]['diagnostics']['status'], 'invalid_skeleton')
        self.assertFalse(self.core.buffers)

if __name__ == '__main__': unittest.main()
