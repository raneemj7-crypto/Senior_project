import unittest
import pandas as pd
from mot20_protocol import GT_COLUMNS, evaluate, preprocess, validate_rows


def gt(rows): return pd.DataFrame(rows,columns=GT_COLUMNS)
def pred(rows): return pd.DataFrame(rows,columns=['frame','id','x','y','w','h','confidence'])


class ProtocolTests(unittest.TestCase):
    def test_correct(self):
        r=evaluate(gt([[1,1,0,0,10,10,1,1,1]]),pred([[1,9,0,0,10,10,.9]]),1)
        self.assertEqual(r['idf1'],1); self.assertEqual(r['mota'],1)

    def test_miss(self):
        r=evaluate(gt([[1,1,0,0,10,10,1,1,1]]),pred([]),1)
        self.assertEqual(r['num_misses'],1)

    def test_false_positive(self):
        r=evaluate(gt([]),pred([[1,9,0,0,10,10,.9]]),1)
        self.assertEqual(r['num_false_positives'],1)

    def test_distractors(self):
        for cls in (2,6,7,8,12):
            with self.subTest(cls=cls):
                r=evaluate(gt([[1,1,0,0,10,10,1,cls,1]]),pred([[1,9,0,0,10,10,.9]]),1)
                self.assertEqual(r['removed_distractor_predictions'],1)
                self.assertEqual(r['num_false_positives'],0)

    def test_not_all_nonpersons_are_ignore_regions(self):
        for cls in (11,13):
            r=evaluate(gt([[1,1,0,0,10,10,1,cls,1]]),pred([[1,9,0,0,10,10,.9]]),1)
            self.assertEqual(r['num_false_positives'],1)

    def test_zero_marked_pedestrian_not_blanket_suppression(self):
        r=evaluate(gt([[1,1,0,0,10,10,0,1,1]]),pred([[1,9,0,0,10,10,.9]]),1)
        self.assertEqual(r['num_false_positives'],1)

    def test_visibility_no_cutoff(self):
        r=evaluate(gt([[1,1,0,0,10,10,1,1,0]]),pred([]),1)
        self.assertEqual(r['num_misses'],1)

    def test_one_to_one_distractor_removal(self):
        r=evaluate(gt([[1,1,0,0,10,10,1,7,1]]),pred([[1,9,0,0,10,10,.9],[1,10,0,0,10,10,.9]]),1)
        self.assertEqual(r['removed_distractor_predictions'],1)
        self.assertEqual(r['num_false_positives'],1)

    def test_person_competes_with_distractor(self):
        g,p,n=preprocess(gt([[1,1,0,0,10,10,1,1,1],[1,2,2,0,10,10,1,7,1]]),pred([[1,9,0,0,10,10,.9]]))
        self.assertEqual(n,0); self.assertEqual(len(p),1)

    def test_switch(self):
        r=evaluate(gt([[1,1,0,0,10,10,1,1,1],[2,1,0,0,10,10,1,1,1]]),
                   pred([[1,9,0,0,10,10,.9],[2,10,0,0,10,10,.9]]),2)
        self.assertEqual(r['num_switches'],1)

    def test_half_iou_boundary(self):
        r=evaluate(gt([[1,1,0,0,10,10,1,1,1]]),pred([[1,9,0,0,5,10,.9]]),1)
        self.assertEqual(r['num_misses'],0)

    def test_invalid_frames_and_duplicates(self):
        for rows in ([[0,1,0,0,10,10,.9]],[[1,1,0,0,10,10,.9]]*2):
            with self.assertRaises(ValueError): validate_rows(pred(rows),1)


if __name__=='__main__': unittest.main()
