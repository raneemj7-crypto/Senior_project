import unittest
import pandas as pd

from audit import classify, match_frame, GT_COLUMNS, CONSISTENT, CONFLICT, INCONCLUSIVE


def gt(rows):
    return pd.DataFrame(rows, columns=GT_COLUMNS)


def pred(rows):
    return pd.DataFrame(rows, columns=['frame','id','x','y','w','h','confidence'])


class IdentityTests(unittest.TestCase):
    def test_consistent(self):
        g = gt([[1, 7, 0, 0, 10, 10, 1, 1, 1]])
        p = pred([[1, 99, 0, 0, 10, 10, .9]])
        ids = [match_frame(g, p)[99]['gt_id'] for _ in range(30)]
        self.assertEqual(classify(ids), CONSISTENT)
        self.assertEqual(ids, [7]*30)

    def test_observed_switch(self):
        ids = []
        for frame in range(30):
            g = gt([[1, 7 if frame < 15 else 8, 0, 0, 10, 10, 1, 1, 1]])
            ids.append(match_frame(g, pred([[1, 99, 0, 0, 10, 10, .9]]))[99]['gt_id'])
        self.assertEqual(classify(ids), CONFLICT)

    def test_unmatched_is_inconclusive_not_switch(self):
        r = match_frame(gt([[1, 7, 100, 100, 10, 10, 1, 1, 1]]),
                        pred([[1, 99, 0, 0, 10, 10, .9]]))[99]
        self.assertIsNone(r['gt_id'])
        self.assertEqual(classify([7]*29+[r['gt_id']]), INCONCLUSIVE)
        self.assertEqual(classify([None]*30), INCONCLUSIVE)

    def test_conflict_takes_precedence_over_unmatched(self):
        self.assertEqual(classify([7, 8]+[None]*28), CONFLICT)

    def test_competing_gt_boxes(self):
        g = gt([[1, 8, 1, 0, 10, 10, 1, 1, 1], [1, 7, 0, 0, 10, 10, 1, 1, 1]])
        p = pred([[1, 99, 0, 0, 10, 10, .9]])
        r = match_frame(g, p)[99]
        self.assertEqual(r['gt_id'], 7)
        self.assertEqual(r['eligible_candidates'], 2)
        self.assertEqual(r['iou'], 1)

    def test_all_predictions_compete_one_to_one(self):
        g = gt([[1, 7, 0, 0, 10, 10, 1, 1, 1]])
        # Prediction 100 need not appear in any manifest; it still wins the GT.
        p = pred([[1, 99, 1, 0, 10, 10, .9], [1, 100, 0, 0, 10, 10, .9]])
        r = match_frame(g, p)
        self.assertIsNone(r[99]['gt_id'])
        self.assertEqual(r[100]['gt_id'], 7)

    def test_distractor_and_zero_visibility(self):
        g = gt([[1, 7, 0, 0, 10, 10, 1, 7, 1], [1, 8, 30, 0, 10, 10, 1, 1, 0]])
        r = match_frame(g, pred([[1, 99, 0, 0, 10, 10, .9], [1, 100, 30, 0, 10, 10, .9]]))
        self.assertEqual(r[99]['status'], 'distractor_removed')
        self.assertIsNone(r[99]['gt_id'])
        self.assertEqual(r[100]['gt_id'], 8)
        self.assertEqual(r[100]['gt_visibility'], 0)

    def test_threshold_and_unmarked_gt(self):
        g = gt([[1, 7, 0, 0, 10, 10, 1, 1, 1]])
        self.assertEqual(match_frame(g, pred([[1, 99, 0, 0, 5, 10, .9]]))[99]['gt_id'], 7)
        self.assertIsNone(match_frame(g, pred([[1, 99, 0, 0, 4.9, 10, .9]]))[99]['gt_id'])
        g['mark'] = 0
        self.assertIsNone(match_frame(g, pred([[1, 99, 0, 0, 10, 10, .9]]))[99]['gt_id'])

    def test_empty_and_tie_ordering(self):
        self.assertEqual(match_frame(gt([]), pred([])), {})
        self.assertIsNone(match_frame(gt([]), pred([[1, 99, 0, 0, 10, 10, .9]]))[99]['gt_id'])
        g = gt([[1, 8, 0, 0, 10, 10, 1, 1, 1], [1, 7, 0, 0, 10, 10, 1, 1, 1]])
        p = pred([[1, 99, 0, 0, 10, 10, .9]])
        self.assertEqual(match_frame(g, p), match_frame(g.iloc[::-1], p))

    def test_length(self):
        with self.assertRaises(ValueError):
            classify([7]*29)


if __name__ == '__main__':
    unittest.main()
