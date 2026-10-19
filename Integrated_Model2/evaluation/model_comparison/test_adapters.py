"""Validation fixtures test contracts, not detection accuracy."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
from unittest.mock import Mock
from evaluation.model_comparison.adapters import (
    RaneemAdapter, SouadAdapter, RANEEM, ROOT, definitions, image_normalize,
    stgcn_format, lstm_format, decoded_logits)
from evaluation.model_comparison.run_comparison import aggregate, metrics
from model2_fall_detection.fall_core import load_fall_model


class AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={kind:RaneemAdapter(kind) for kind in ('stgcn_visibility','stgcn_fullbody','lstm')}
        cls.settings,predict=load_fall_model()
        cls.souad_predict=staticmethod(predict)

    def test_shapes_and_coordinate_order(self):
        raw=np.arange(30*17*3,dtype=np.float32).reshape(1,30,17,3)
        raw[...,2]=.9
        normalized=image_normalize(raw,640,480)
        self.assertEqual(stgcn_format(normalized).shape,(1,3,30,17,1))
        self.assertEqual(lstm_format(normalized).shape,(1,30,51))
        np.testing.assert_allclose(normalized[...,0],raw[...,0]/640)
        np.testing.assert_allclose(normalized[...,1],raw[...,1]/480)
        np.testing.assert_array_equal(normalized[...,2],raw[...,2])
        np.testing.assert_array_equal(stgcn_format(normalized)[0,1,:,5,0],normalized[0,:,5,1])
        np.testing.assert_array_equal(lstm_format(normalized)[0,0,:3],normalized[0,0,0])
        with self.assertRaises(ValueError): stgcn_format(np.zeros((1,3,30,17,1)))

    def test_label_mapping_and_tie_rule(self):
        p,labels=decoded_logits(torch.tensor([[2.,-2.],[-2.,2.],[0.,0.]]))
        self.assertEqual(labels.tolist(),[0,1,0])
        self.assertLess(p[0],.5);self.assertGreater(p[1],.5)
        self.assertEqual(self.models['lstm'].metadata['checkpoint_metadata']['classes'],['non_fall','fall'])

    def test_real_checkpoint_loading_and_forward(self):
        clips=np.ones((2,30,17,3),np.float32)*100
        clips[...,2]=.9
        for adapter in self.models.values():
            scores,labels=adapter.predict(clips,640,480)
            self.assertEqual(scores.shape,(2,));self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(np.isin(labels,[0,1]).all())
        clip=clips.copy();clip[...,0]=np.arange(17)*10+100
        adapter=SouadAdapter(self.souad_predict,self.settings,ROOT/'model2_fall_detection/models/stgcnpp_fall_v1_laptop.pt')
        scores,labels=adapter.predict(clip,640,480)
        self.assertTrue(np.isfinite(scores).all());self.assertEqual(labels.shape,(2,))

    def test_bad_checkpoint_and_reversed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.pth'
            saved=torch.load(RANEEM/'results/trained_model/fall_detection_lstm.pth',weights_only=True,map_location='cpu')
            saved['classes']=['fall','non_fall'];torch.save(saved,path)
            with self.assertRaises(ValueError):RaneemAdapter('lstm',path)
            torch.save({'model_state_dict':{}},path)
            with self.assertRaises(RuntimeError):RaneemAdapter('stgcn_visibility',path)
            with self.assertRaises(FileNotFoundError):RaneemAdapter('stgcn_visibility',Path(tmp)/'missing.pth')

    def test_visibility_abstention_not_negative(self):
        for adapter in self.models.values():
            scores,labels=adapter.predict(np.zeros((1,30,17,3),np.float32),640,480)
            self.assertEqual(labels.tolist(),[-1]);self.assertTrue(np.isnan(scores).all())
            row=aggregate(scores,labels,np.array([29]))
            self.assertIsNone(row['prediction']);self.assertEqual(row['status'],'abstained')
        row.update(truth='non_fall',fall_events=None)
        m=metrics([row]);self.assertEqual(m['TN'],0);self.assertEqual(m['abstained'],1)

    def test_saved_training_layout_matches_source_conversion(self):
        x=np.load(RANEEM/'results/training_data/X_train_sample.npy')
        ids=np.load(RANEEM/'results/training_data/event_ids.npy')
        y=np.load(RANEEM/'results/training_data/y_train_sample.npy')
        test=np.isin(ids,['fall-04','adl-19-cam0-rgb'])
        np.testing.assert_array_equal(stgcn_format(x[test]),np.load(RANEEM/'stgcn_project/data/X_test.npy'))
        np.testing.assert_array_equal(y[test],np.load(RANEEM/'stgcn_project/data/y_test.npy'))
        self.assertEqual(set(np.unique(y)),{0,1})
        self.assertTrue(np.all((x[...,:2]>=0)&(x[...,:2]<=1)))

    def test_souad_threshold_unchanged(self):
        predictor=Mock(return_value=np.array([.69,.70]))
        adapter=SouadAdapter(predictor,self.settings,ROOT/'model2_fall_detection/models/stgcnpp_fall_v1_laptop.pt')
        raw=np.ones((2,30,17,3),np.float32)
        raw[...,0]=np.arange(17)*10;raw[...,1]=np.arange(17)*5
        scores,labels=adapter.predict(raw,640,480)
        self.assertEqual(labels.tolist(),[0,1]);self.assertEqual(predictor.call_args[0][0].shape,(2,30,17,3))

if __name__=='__main__':unittest.main()
