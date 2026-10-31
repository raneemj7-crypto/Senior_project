"""Adapter for Raneem's ORIGINAL YOLOv8n Pose + STGCN++ checkpoint.

Input: complete RGB camera frame, Model 1 ByteTrack outputs.
STGCN input: (N,3,30,17,1), channel order x,y,confidence, x/y
normalized by FULL FRAME dimensions as in prepare_training_data.py.
"""
from collections import defaultdict, deque
from pathlib import Path
import sys
import numpy as np
import torch
from ultralytics import YOLO


def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    area=max(0,x2-x1)*max(0,y2-y1)
    if area<=0: return 0.0
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return float(area/(aa+bb-area+1e-8))

class RaneemModel2:
    def __init__(self, root, *, threshold=.70, visibility_threshold=.35,
                 min_reliable_frames=15, cooldown_frames=90, pose_conf=.25,
                 min_match_iou=.1, device=None):
        root=Path(root).resolve()
        stgcn_dir=root/'stgcn_project'
        if not (stgcn_dir/'stgcn_model.py').is_file():
            raise FileNotFoundError(f'Cannot find Raneem model: {stgcn_dir}')
        sys.path.insert(0,str(stgcn_dir))
        from stgcn_model import STGCNPlusPlus
        if device is None: device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device=device
        self.pose=YOLO(str(root/'yolov8n-pose.pt'))
        self.fall=STGCNPlusPlus().to(device)
        ckpt=torch.load(stgcn_dir/'trained_model'/'best_stgcn_model.pth',map_location=device,weights_only=True)
        self.fall.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        self.fall.eval()
        self.buffers=defaultdict(lambda:deque(maxlen=30))
        self.last_alert={}
        self.frame_index=0
        self.threshold=threshold
        self.visibility_threshold=visibility_threshold
        self.min_reliable_frames=min_reliable_frames
        self.cooldown_frames=cooldown_frames
        self.pose_conf=pose_conf
        self.min_match_iou=min_match_iou

    def process(self, frame, timestamp, tracks):
        self.frame_index+=1
        h,w=frame.shape[:2]
        result=self.pose.predict(frame,conf=self.pose_conf,verbose=False,device=self.device)[0]
        pose_boxes=[]; pose_joints=[]
        if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
            pose_boxes=result.boxes.xyxy.cpu().numpy().tolist()
            pose_joints=result.keypoints.data.cpu().numpy().copy()
        used=set(); output=[]; events=[]
        for track in tracks:
            tid=int(track['track_id'])
            box=track['box']; best_idx=None; best_score=self.min_match_iou
            for j,pbox in enumerate(pose_boxes):
                if j in used: continue
                score=iou(box,pbox)
                if score>best_score: best_idx=j;best_score=score
            pose=np.zeros((17,3),dtype=np.float32)
            pose_pixels=None
            if best_idx is not None:
                used.add(best_idx)
                pose=pose_joints[best_idx].astype(np.float32)
                pose_pixels=pose.copy()  # full-frame pixel coordinates for live skeleton display
                pose[:,0]/=w
                pose[:,1]/=h
            self.buffers[tid].append(pose)
            item={'track_id':tid,'frames_collected':len(self.buffers[tid]),'fall_probability':None,
                  'fall_detected':False,'reliable':False,
                  'keypoints': pose_pixels.tolist() if pose_pixels is not None else None}
            if len(self.buffers[tid])==30:
                arr=np.stack(self.buffers[tid]) # (T,V,C)
                good_frames=((arr[:,:,2]>=self.visibility_threshold).sum(axis=1)>=3).sum()
                item['reliable']=bool(good_frames>=self.min_reliable_frames)
                if item['reliable']:
                    input_arr=np.transpose(arr,(2,0,1))[None,...,None].copy()
                    tensor=torch.from_numpy(input_arr).to(self.device)
                    with torch.no_grad():
                        logits=self.fall(tensor)
                        prob=float(torch.softmax(logits,dim=1)[0,1].item())
                    item['fall_probability']=prob
                    is_fall=prob>=self.threshold
                    item['fall_detected']=is_fall
                    if is_fall and self.frame_index-self.last_alert.get(tid,-999999)>=self.cooldown_frames:
                        self.last_alert[tid]=self.frame_index
                        events.append({'type':'possible_collapse','track_id':tid,
                            'confidence':round(prob,4),'timestamp_seconds':round(float(timestamp),3),
                            'frame':self.frame_index,'source':'raneem_stgcnpp'})
            output.append(item)
        return {'tracks':output,'events':events}
