"""Independent inference adapters. Never import training scripts for execution."""
import ast
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model2_fall_detection.fall_core import normalize_clip

ROOT = Path(__file__).resolve().parents[2]
RANEEM = ROOT.parent / 'Raneem_Work'
LABELS = ['non_fall', 'fall']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions(path, names):
    """Compile only explicitly audited class/function definitions, no training I/O.

    This also avoids writing __pycache__ into the read-only teammate repository.
    """
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing definitions in {path}')
    scope = dict(torch=torch, nn=nn, F=F, np=np)
    # This constant is part of the original ST-GCN graph definition.
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COCO_EDGES' for t in n.targets):
            scope['COCO_EDGES'] = ast.literal_eval(n.value)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def validate_clips(clips):
    clips = np.asarray(clips, dtype=np.float32)
    if clips.ndim != 4 or clips.shape[1:] != (30, 17, 3):
        raise ValueError(f'Expected (N,30,17,3), got {clips.shape}')
    if not np.isfinite(clips).all() or (clips[..., 2] < 0).any() or (clips[..., 2] > 1).any():
        raise ValueError('Invalid pose values/confidences')
    return clips


def image_normalize(clips, width, height):
    clips = validate_clips(clips).copy()
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    clips[..., 0] /= width
    clips[..., 1] /= height
    return clips


def stgcn_format(clips):
    return np.transpose(validate_clips(clips), (0, 3, 1, 2))[:, :, :, :, None].copy()


def lstm_format(clips):
    return validate_clips(clips).reshape(-1, 30, 51).copy()


def decoded_logits(logits):
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite (N,2) logits with labels [non_fall, fall]')
    # Preserve torch argmax tie behavior: equal logits predict class zero.
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy(), logits.argmax(1).cpu().numpy()


class RaneemAdapter:
    def __init__(self, kind, checkpoint=None):
        self.kind = kind
        self.model_id = 'raneem_' + kind
        if kind in ('stgcn_visibility', 'stgcn_fullbody'):
            source = RANEEM / 'stgcn_project/stgcn_model.py'
            model_cls = definitions(source, ['build_adjacency', 'STGCNBlock', 'STGCNPlusPlus'])['STGCNPlusPlus']
            name = 'best_stgcn_model.pth' if kind == 'stgcn_visibility' else 'fullbody_baseline.pth'
            path = RANEEM / 'stgcn_project/trained_model' / name
            self.gate = 'At least 15 frames with >=3 joints at confidence >=0.35'
            self.visibility = None
        elif kind == 'lstm':
            source = RANEEM / 'train_fall_model.py'
            model_cls = definitions(source, ['FallDetectionLSTM'])['FallDetectionLSTM']
            path = RANEEM / 'results/trained_model/fall_detection_lstm.pth'
            scope = definitions(RANEEM / 'visibility_detection.py', ['detect_visibility'])
            scope.update(HEAD=[0,1,2,3,4], SHOULDERS=[5,6], ARMS=[7,8,9,10], HIPS=[11,12], LEGS=[13,14,15,16], CONF_THRESHOLD=.35)
            self.visibility = scope['detect_visibility']
            self.gate = 'Abstain if >15 of 30 frames have uncertain visibility (visibility_aware_model.py)'
        else:
            raise ValueError(f'Unknown model kind: {kind}')
        path = Path(checkpoint) if checkpoint else path
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if kind == 'lstm':
            expected = dict(input_size=51, hidden_size=64, num_layers=2, window_size=30, classes=LABELS)
            for key, value in expected.items():
                if saved.get(key) != value:
                    raise ValueError(f'Incompatible LSTM metadata {key}: {saved.get(key)!r}')
        self.net = model_cls().cpu().eval()
        self.net.load_state_dict(saved['model_state_dict'], strict=True)
        self.metadata = dict(available=True, checkpoint=str(path), checkpoint_sha256=digest(path),
            architecture_source=str(source), source_sha256=digest(source),
            checkpoint_metadata={k:v for k,v in saved.items() if k != 'model_state_dict'},
            input_shape='N,30,51' if kind == 'lstm' else 'N,3,30,17,1', labels=LABELS,
            normalization='x/image_width, y/image_height; confidence unchanged',
            decision_rule='argmax two logits; ties => non_fall; no independently documented probability threshold',
            visibility_gate=self.gate, model_error=None)

    def preprocess(self, clips, width, height):
        normalized = image_normalize(clips, width, height)
        if self.kind == 'lstm':
            reliable = np.array([sum(self.visibility(f) == 'uncertain' for f in clip) <= 15 for clip in normalized])
            inputs = lstm_format(normalized)
        else:
            reliable = ((normalized[..., 2] >= .35).sum(2) >= 3).sum(1) >= 15
            inputs = stgcn_format(normalized)
        return inputs, reliable

    @torch.no_grad()
    def predict(self, clips, width, height):
        inputs, reliable = self.preprocess(clips, width, height)
        scores = np.full(len(inputs), np.nan)
        predictions = np.full(len(inputs), -1, dtype=int)
        if reliable.any():
            p, labels = decoded_logits(self.net(torch.from_numpy(inputs[reliable])))
            scores[reliable], predictions[reliable] = p, labels
        return scores, predictions


class SouadAdapter:
    """Original exported classifier and normalization on the shared 15-FPS clips.

    This diagnostic has no tracker/person-confirmation/alarm hold. The existing
    integrated operational baseline is run separately, with its actual events.
    """
    model_id = 'souad_shared_clips'

    def __init__(self, predict, settings, checkpoint):
        self.predict_net, self.settings = predict, settings
        if settings['clip_frames'] != 30 or settings['clip_fps'] != 15 or settings['threshold'] != .7:
            raise ValueError('Unexpected Souad settings; refusing to adjust')
        self.metadata = dict(available=True, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            input_shape='N,30,17,3', labels=LABELS, normalization='Original normalize_clip, joint_ok=0.30',
            decision_rule='exported P(fall) >= 0.70', visibility_gate='max_missing_frac <=0.30; original normalization validity',
            model_error=None)

    def predict(self, clips, width, height):
        clips = validate_clips(clips)
        normalized, indices = [], []
        for i, clip in enumerate(clips):
            valid = np.any(clip[..., 2] > 0, axis=1)
            if (~valid).mean() > self.settings['max_missing_frac']:
                continue
            value = normalize_clip(clip[..., :2], clip[..., 2], valid, self.settings['joint_ok'])
            if value is not None:
                normalized.append(value); indices.append(i)
        scores = np.full(len(clips), np.nan)
        labels = np.full(len(clips), -1, dtype=int)
        if indices:
            p = np.asarray(self.predict_net(np.stack(normalized))).reshape(-1)
            if len(p) != len(indices) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError('Invalid Souad probability output')
            scores[indices] = p
            labels[indices] = (p >= self.settings['threshold']).astype(int)
        return scores, labels
