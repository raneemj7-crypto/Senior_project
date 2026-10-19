"""Experimental per-ID inference for Raneem's two ST-GCN checkpoints.

Model 1 supplies tracked COCO-17 coordinates on each frame. No new pose estimation,
identity interpolation, or tuned fusion is performed. This module is not the final
alert dispatcher: candidate events are for evaluation until externally validated.
"""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from .stgcn_model import STGCNPlusPlus

MODES = ('visibility', 'fullbody', 'parallel')
FILES = {'visibility': 'best_stgcn_model.pth', 'fullbody': 'fullbody_baseline.pth'}

class RaneemTrackedPipeline:
    def __init__(self, checkpoint_dir=None, mode='visibility', fps=15, stride=10,
                 camera_id='cam0', joint_threshold=.35, min_reliable_frames=15,
                 cooldown_s=3.0, device='cpu'):
        if mode not in MODES: raise ValueError(f'mode must be one of {MODES}')
        if fps <= 0 or stride <= 0: raise ValueError('fps/stride must be positive')
        self.mode, self.fps, self.stride = mode, float(fps), int(stride)
        self.camera_id, self.joint_threshold = camera_id, joint_threshold
        self.min_reliable_frames, self.cooldown_s = min_reliable_frames, cooldown_s
        self.device = torch.device(device)
        directory = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).resolve().parent.parent/'checkpoints'
        self.models = {}
        for name in (('visibility','fullbody') if mode == 'parallel' else (mode,)):
            path = directory/FILES[name]
            if not path.is_file(): raise FileNotFoundError(f'Missing checkpoint: {path}')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            state = checkpoint.get('model_state_dict', checkpoint)
            net = STGCNPlusPlus()
            net.load_state_dict(state, strict=True)
            self.models[name] = net.to(self.device).eval()
        self.reset()

    def reset(self):
        self.history = {}
        self.last_emitted = {}
        self.previous_decisions = {}
        self.counts = {}
        self.last_time = None

    def _predict(self, sequence):
        # Sequence: 30 x 17 x (x/img_width, y/img_height, confidence)
        # Raneem input layout: N x 3 x 30 x 17 x 1
        confident = sequence[:, :, 2] >= self.joint_threshold
        reliable = int((confident.sum(axis=1) >= 3).sum())
        if reliable < self.min_reliable_frames:
            return {k: {'decision':'abstain','fall_probability':None,'reliable_frames':reliable}
                    for k in self.models}
        x = torch.from_numpy(np.transpose(sequence[None], (0,3,1,2))[:,:,:,:,None].copy()).float().to(self.device)
        out = {}
        with torch.no_grad():
            for name, network in self.models.items():
                probs = torch.softmax(network(x), dim=1)[0].cpu().numpy()
                label = int(np.argmax(probs)) # original class-argmax; 0 nonfall, 1 fall
                out[name] = {'decision':'fall' if label == 1 else 'non_fall',
                             'fall_probability': float(probs[1]),'reliable_frames':reliable}
        return out

    def process(self, frame, t, tracks):
        """Tracks require {track_id, box, keypoints (17,2), keypoint_conf (17,)}.
        Keypoints must be from THIS image, not normalized. Missing-pose tracks abstain.
        """
        if frame is None or getattr(frame, 'ndim', 0) < 2: raise ValueError('BGR frame required')
        t = float(t)
        if self.last_time is not None and t <= self.last_time: raise ValueError('time must increase')
        self.last_time = t
        height,width = frame.shape[:2]
        result={'time':t,'tracks':{},'events':[]}
        seen=set()
        for tr in tracks:
            tid=int(tr['track_id'])
            if tid in seen: raise ValueError(f'duplicate track_id {tid}')
            seen.add(tid)
            if 'keypoints' not in tr or 'keypoint_conf' not in tr:
                result['tracks'][tid]={'status':'missing_pose','predictions':{}}
                continue
            xy=np.asarray(tr['keypoints'], dtype=np.float32)
            conf=np.asarray(tr['keypoint_conf'], dtype=np.float32)
            if xy.shape!=(17,2) or conf.shape!=(17,) or not np.isfinite(xy).all() or not np.isfinite(conf).all() or np.any((conf<0)|(conf>1)):
                result['tracks'][tid]={'status':'invalid_pose','predictions':{}}
                continue
            sample=np.column_stack((xy[:,0]/width,xy[:,1]/height,conf)).astype('float32')
            history=self.history.setdefault(tid,deque(maxlen=30))
            # A missing temporal frame is explicitly blank; do not invent skeletons.
            if history:
                delta=t-history[-1][0]
                missing=min(30,max(0,round(delta*self.fps)-1))
                for step in range(missing,0,-1):
                    history.append((t-step/self.fps,np.zeros((17,3),dtype=np.float32)))
            history.append((t,sample))
            self.counts[tid]=self.counts.get(tid,0)+1
            if len(history)<30:
                result['tracks'][tid]={'status':'insufficient_history','history_frames':len(history),'predictions':{}}
                continue
            if (self.counts[tid]-30)%self.stride!=0:
                result['tracks'][tid]={'status':'between_windows','predictions':self.previous_decisions.get(tid,{})}
                continue
            seq=np.stack([v for _,v in history])
            preds=self._predict(seq)
            self.previous_decisions[tid]=preds
            result['tracks'][tid]={'status':'ready','predictions':preds}
            # Only standalone visibility mode emits candidate events by default.
            # Parallel mode NEVER silently ORs the two model decisions.
            if self.mode!='parallel' and preds[self.mode]['decision']=='fall':
                key=(tid,self.mode)
                if t-self.last_emitted.get(key,-1e20)>=self.cooldown_s:
                    self.last_emitted[key]=t
                    result['events'].append({'event':'candidate_fall','camera_id':self.camera_id,
                        'track_id':tid,'time':t,'model':f'raneem_{self.mode}',
                        'fall_probability':preds[self.mode]['fall_probability'],
                        'box':list(map(float,tr['box']))})
        # Forget stale identifiers rather than inheriting another person's poses.
        for tid in list(self.history):
            if tid not in seen and self.history[tid] and t-self.history[tid][-1][0]>2:
                self.history.pop(tid,None)
                self.counts.pop(tid,None)
                self.previous_decisions.pop(tid,None)
        return result
