import tempfile
from pathlib import Path
import unittest
import json
from unittest.mock import patch

import compare_buffer60 as comparison
from compare_buffer60 import require_equal, require_prediction, pair_windows, metric_provider


def observations(sid, track=1, offset=0):
    return [dict(sequence_id=sid,sample_index=i,track_id=track,frame=i+1,
                 x1=offset,y1=0,x2=offset+10,y2=10,confidence=.9) for i in range(30)]


class ComparisonTests(unittest.TestCase):
    def test_artifact_verifier_rejects_wrong_coordinate_plan(self):
        # Exercise the actual preflight path before it can reach crop generation.
        frozen = json.loads(comparison.FROZEN.read_text())
        plan = dict(runner_sha256='runner', evaluator_sha256='protocol', prediction_origin='zero-to-one')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'plan.json').write_text(json.dumps(plan))
            hashes = {str(comparison.PROJECT/frozen['detector']['weights']):frozen['detector']['checkpoint_sha256'],
                      str(comparison.PROJECT/'model1_cross_validation/run_sequence_cv.py'):'runner',
                      str(comparison.PROJECT/'model1_cross_validation/evaluation/mot20_protocol.py'):'protocol'}
            with patch.object(comparison,'CV',root), patch.object(comparison,'digest',side_effect=lambda p:hashes[str(p)]):
                with self.assertRaisesRegex(ValueError,'coordinates'):
                    comparison.verify_artifacts()

    def test_artifact_verifier_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'plan.json').write_text('{}')
            with patch.object(comparison,'CV',root), patch.object(comparison,'digest',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'checkpoint'):
                    comparison.verify_artifacts()

    def test_provenance_mismatches(self):
        for actual,expected,label in [(30,60,'tracker buffer'),('zero-to-one','as-stored','coordinates'),
                                       ('bad','good','checkpoint'),(30,25,'frame rate')]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError,'Provenance mismatch'):
                require_equal(actual,expected,label)

    def test_missing_prediction_never_substitutes(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'buffer30.csv').write_text('available but wrong')
            with self.assertRaisesRegex(FileNotFoundError,'no buffer30 substitution'):
                require_prediction(Path(d)/'buffer60.csv')

    def test_metric_hash_mismatch(self):
        lookup = metric_provider({'MOT20-01':dict(prediction='p',protected_sha256={'p':'expected'})})
        with self.assertRaisesRegex(ValueError,'new crop prediction'):
            lookup('MOT20-01','wrong','gt')

    def test_differing_inventory_and_local_track_ids(self):
        a=observations('old',track=1)+observations('old_only',offset=20)
        b=observations('new',track=99)+observations('new_only',offset=30)
        wa=[dict(sequence_id=x,classification='consistent') for x in ['old','old_only']]
        wb=[dict(sequence_id=x,classification='conflict') for x in ['new','new_only']]
        pairs,stats=pair_windows(a,b,wa,wb)
        self.assertEqual(len(pairs),1)
        self.assertEqual((stats['buffer30_unpaired'],stats['buffer60_unpaired']),(1,1))
        self.assertEqual(pairs[0]['buffer30_classification'],'consistent')
        self.assertEqual(pairs[0]['buffer60_classification'],'conflict')

    def test_same_track_id_is_not_pairing_evidence(self):
        pairs,stats=pair_windows(observations('a'),observations('b',offset=1),
            [dict(sequence_id='a',classification='consistent')],[dict(sequence_id='b',classification='consistent')])
        self.assertEqual(pairs,[])
        self.assertEqual(stats['exact_pairs'],0)

    def test_ambiguous_duplicate_geometry_not_paired(self):
        a=observations('a')+observations('b')
        pairs,_=pair_windows(a,observations('c'),[dict(sequence_id=s,classification='x') for s in ['a','b']],
                            [dict(sequence_id='c',classification='x')])
        self.assertEqual(pairs,[])

    def test_missing_sample_rejected(self):
        with self.assertRaisesRegex(ValueError,'inventory'):
            pair_windows(observations('a')[:-1],[],[],[])


if __name__ == '__main__':
    unittest.main()
