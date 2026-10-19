import unittest
import numpy as np
from raneem_live.pipeline import RaneemTrackedPipeline

class Tests(unittest.TestCase):
    def test_independent_ids_and_shapes(self):
        m=RaneemTrackedPipeline(mode='parallel')
        img=np.zeros((240,320,3),np.uint8)
        def tr(i): return dict(track_id=i,box=[0,0,60,180],keypoints=np.full((17,2),45,np.float32),keypoint_conf=np.ones(17,np.float32))
        for k in range(30): out=m.process(img,k/15,[tr(1),tr(2)])
        self.assertEqual(len(out['tracks']),2)
        self.assertEqual(out['events'],[])
        self.assertIn('visibility',out['tracks'][1]['predictions'])
        self.assertIn('fullbody',out['tracks'][2]['predictions'])
    def test_missing_pose_does_not_fake_skeleton(self):
        m=RaneemTrackedPipeline(mode='visibility')
        o=m.process(np.zeros((40,40,3),np.uint8),0,[dict(track_id=1,box=[0,0,20,20])])
        self.assertEqual(o['tracks'][1]['status'],'missing_pose')
        self.assertFalse(m.history)
    def test_time_must_increase(self):
        m=RaneemTrackedPipeline(mode='visibility')
        img=np.zeros((40,40,3),np.uint8)
        m.process(img,0,[])
        with self.assertRaises(ValueError):m.process(img,0,[])
if __name__=='__main__':unittest.main()
