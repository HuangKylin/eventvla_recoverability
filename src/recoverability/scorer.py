"""Small action-conditioned deletion-cost regressor for the mechanism pilot.

Uses only RGB, public slot/time metadata and proposed motor targets. No scene
truth, action-slot label, seed, recoverability flag, or future query enters it.
This is NOT a trained EventVLA visual feature adapter or a paper-ready method.
"""
import numpy as np
import torch
from torch import nn
from recoverability.fixed_skills import SLOT_ROIS


def evidence_features(image,slot,ordinal):
    a=np.asarray(image,dtype=np.float32)/255.
    h,w=a.shape[:2]
    x0,y0,x1,y1=SLOT_ROIS[slot]
    roi=a[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]
    return np.concatenate([np.eye(4,dtype=np.float32)[slot],
        [ordinal/3.],roi.mean(axis=(0,1)),roi.std(axis=(0,1))]).astype(np.float32)


def public_features(image,slot,ordinal,sampled_drives):
    drives=np.asarray(sampled_drives,dtype=np.float32)
    if drives.ndim!=2 or drives.shape[0]!=16 or not np.isfinite(drives).all():
        raise ValueError('Expected 16 finite samples of the proposed motor sequence')
    return np.concatenate([evidence_features(image,slot,ordinal),drives.ravel()])


class DeletionScorer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x): return self.network(x).squeeze(-1)


def load_scorer(path):
    package=torch.load(path,map_location='cpu',weights_only=True)
    if package['protocol']!='physical_latch_pilot_v1': raise ValueError('Wrong scorer protocol')
    model=DeletionScorer(package['width'])
    model.load_state_dict(package['state_dict'],strict=True);model.eval()
    return model,package


def predict_scores(path,features):
    model,p=load_scorer(path)
    x=(torch.as_tensor(np.asarray(features),dtype=torch.float32)-p['mean'])/p['scale']
    if p['action_ablated']: x[:,11:]=0
    with torch.inference_mode(): return model(x).numpy()
