"""Causal inference for the recommended startup-marginal GMM.

Inputs are raw [upper vibration, lower vibration, current]. Call reset at
each known clip boundary. This is an experimental detector, not a certified
press-stop control. Seed 0 is the example deployable artifact, not a selected
best seed. All reported performance is seed 0/1/2 mean and SD.
"""
from pathlib import Path
import json
import numpy as np
import joblib
from scipy.special import logsumexp

class StartupMarginalGMM:
    def __init__(self, artifact_dir, seed=0):
        artifact_dir=Path(artifact_dir)
        c=json.loads((artifact_dir/f'config_seed{seed}.json').read_text())
        self.model=joblib.load(artifact_dir/c['model_file'])
        self.mean=np.array(c['mean']);self.scale=np.array(c['scale'])
        self.thresholds=np.array(c['threshold_1_2_3_observations'])
        self.reset()

    def reset(self):
        self.buffer=[]

    def update(self, raw_values):
        raw=np.asarray(raw_values,dtype=float)
        if raw.shape!=(3,) or not np.isfinite(raw).all():
            raise ValueError('Expected three finite sensor values in configured order')
        self.buffer.append((raw-self.mean)/self.scale)
        self.buffer=self.buffer[-3:]
        n=len(self.buffer);cols=list(range(9-3*n,9));z=np.array(self.buffer).ravel()
        logs=[]
        for k,w in enumerate(self.model.weights_):
            cov=self.model.covariances_[k][np.ix_(cols,cols)]
            delta=z-self.model.means_[k,cols]
            sign,ld=np.linalg.slogdet(cov)
            if sign<=0:raise ValueError('Non-positive covariance determinant')
            logs.append(np.log(w)-.5*(len(cols)*np.log(2*np.pi)+ld+delta@np.linalg.solve(cov,delta)))
        score=float(-logsumexp(logs));threshold=float(self.thresholds[n-1])
        return {'anomaly':bool(score>threshold),'score':score,'threshold':threshold,'observed_points':n,'dimension':3*n}
