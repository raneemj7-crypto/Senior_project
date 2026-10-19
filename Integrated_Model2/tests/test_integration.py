import sys, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated_model2 import IntegratedModel2

class FakeDetector:
    def __init__(self):
        self.core = SimpleNamespace(renamed={})
        self.calls = []
    def process(self, frame, t, tracks):
        self.calls.append(tracks)
        return {'time':t, 'tracks':{int(tr['track_id']):{'fall_probability':0.81,'falling':True, 'keypoints':tr.get('keypoints'), 'keypoint_conf':tr.get('keypoint_conf')} for tr in tracks},
                'events':[{'track_id':int(tracks[0]['track_id']), 'event':'fall_detected'}] if tracks else []}

class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = IntegratedModel2.__new__(IntegratedModel2)
        from collections import defaultdict, deque
        self.model.visibility_history=defaultdict(lambda:deque(maxlen=30))
        self.model.detector=FakeDetector()
    def test_full_body_annotation_preserves_probability(self):
        tracks=[{'track_id':5,'box':[0,0,100,200], 'keypoints':np.ones((17,2)), 'keypoint_conf':np.ones(17)}]
        result=self.model.process(np.zeros((100,100,3),dtype=np.uint8),1.,tracks)
        self.assertEqual(result['tracks'][5]['visibility'],'full_body')
        self.assertEqual(result['tracks'][5]['fall_probability'],0.81)
        self.assertEqual(result['events'][0]['visibility'],'full_body')
    def test_uncertain_annotation(self):
        tracks=[{'track_id':6,'box':[0,0,100,200], 'keypoints':np.zeros((17,2)), 'keypoint_conf':np.zeros(17)}]
        result=self.model.process(None,1.,tracks)
        self.assertEqual(result['tracks'][6]['visibility'],'uncertain')
    def test_model1_input_is_not_modified(self):
        tracks=[{'track_id':1,'box':[1,2,3,4], 'keypoints':np.ones((17,2)), 'keypoint_conf':np.ones(17)}]
        self.model.process(None,1.,tracks)
        self.assertNotIn('visibility',tracks[0])

if __name__=='__main__': unittest.main()
