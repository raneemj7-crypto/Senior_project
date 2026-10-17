"""Diagnostic motmetrics backend with TrackEval-derived MOT20 preprocessing.

Reference: https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py
get_preprocessed_seq_data, inspected 2026-10-12. No general crowd-region masking.
"""
import numpy as np
import pandas as pd
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

GT_COLUMNS = ['frame','id','x','y','w','h','mark','class_id','visibility']
DISTRACTORS = (2,6,7,8,12)
METRICS = ['idf1','idp','idr','mota','num_switches','num_fragmentations','precision','recall','num_misses','num_false_positives']
STATUS = 'Diagnostic: TrackEval-derived MOT20 preprocessing + motmetrics; not official MOTChallenge scores'


def validate_rows(data, total, gt=False):
    cols = GT_COLUMNS if gt else ['frame','id','x','y','w','h','confidence']
    if not np.isfinite(data[cols].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite numeric values')
    if ((data.frame % 1 != 0)|(data.id % 1 != 0)|(~data.frame.between(1,total))|(data.id<0)).any():
        raise ValueError('Invalid frame or ID (frames are 1..seqLength)')
    if data.duplicated(['frame','id']).any():
        raise ValueError('Duplicate frame/identity')
    if ((data.w<=0)|(data.h<=0)).any():
        raise ValueError('Nonpositive bounding box extent')
    if gt and ((~data.class_id.isin(range(1,14)))|(~data.mark.isin([0,1]))|(~data.visibility.between(0,1))).any():
        raise ValueError('Unexpected GT class/mark/visibility')
    if not gt and (~data.confidence.between(0,1)).any():
        raise ValueError('Confidence outside [0,1]')


def load_predictions(path,total,offset=0):
    data=pd.read_csv(path)
    if 'observation_type' in data and (data.observation_type=='head_only').any():
        raise ValueError('Head-only observations cannot enter body tracking evaluation')
    for field,expected in [('width',data.x2-data.x1),('height',data.y2-data.y1),
                           ('center_x',(data.x1+data.x2)/2),('center_y',(data.y1+data.y2)/2)]:
        if field in data and not np.allclose(data[field],expected,rtol=0,atol=1e-5):
            raise ValueError('Inconsistent '+field)
    result=pd.DataFrame(dict(frame=data.frame,id=data.track_id,x=data.x1+offset,y=data.y1+offset,
                             w=data.x2-data.x1,h=data.y2-data.y1,confidence=data.confidence))
    validate_rows(result,total)
    return result


def ious(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4); b=np.asarray(b,dtype=float).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])-
                            np.maximum(a[:,None,:2],b[None,:,:2])).prod(axis=2)
    union=a[:,2:].prod(axis=1)[:,None]+b[:,2:].prod(axis=1)[None,:]-intersection
    return intersection/np.maximum(union,np.finfo(float).eps)


def preprocess(gt,pred):
    """Match ALL GT first, remove matched distractor predictions, then filter GT."""
    scores=ious(gt[['x','y','w','h']],pred[['x','y','w','h']])
    remove=[]
    if scores.size:
        scores[scores < .5-np.finfo(float).eps]=0
        rows,cols=linear_sum_assignment(-scores)
        remove=[col for row,col in zip(rows,cols) if scores[row,col]>np.finfo(float).eps
                and int(gt.iloc[row].class_id) in DISTRACTORS]
    keep=np.ones(len(pred),dtype=bool); keep[remove]=False
    return gt[(gt.mark!=0)&(gt.class_id==1)],pred.iloc[keep],len(remove)


def evaluate(gt,pred,total,detection_only=False):
    validate_rows(gt,total,gt=True); validate_rows(pred,total)
    gs=dict(tuple(gt.groupby('frame'))); ps=dict(tuple(pred.groupby('frame')))
    acc=mm.MOTAccumulator(auto_id=False)
    removed=tp=fp=fn=0
    for frame in range(1,total+1):
        g,p,n=preprocess(gs.get(frame,gt.iloc[:0]),ps.get(frame,pred.iloc[:0])); removed+=n
        if detection_only:
            scores=ious(g[['x','y','w','h']],p[['x','y','w','h']])
            scores[scores<.5]=0
            matches=0
            if scores.size:
                r,c=linear_sum_assignment(-scores)
                matches=int((scores[r,c]>0).sum())
            tp+=matches; fp+=len(p)-matches; fn+=len(g)-matches
        else:
            distances=mm.distances.iou_matrix(g[['x','y','w','h']].to_numpy(),p[['x','y','w','h']].to_numpy(),max_iou=.5)
            acc.update(g.id.astype(int).tolist(),p.id.astype(int).tolist(),distances,frameid=frame)
    if detection_only:
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        result=dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
                    f1=2*precision*recall/(precision+recall) if precision+recall else 0)
    else:
        result=mm.metrics.create().compute(acc,metrics=METRICS).iloc[0].to_dict()
    return dict(result,removed_distractor_predictions=removed)
