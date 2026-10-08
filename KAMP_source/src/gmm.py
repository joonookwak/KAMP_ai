"""GMM likelihood on the observed coordinates only (no future imputation)."""
import numpy as np
from scipy.special import logsumexp

def marginal_nll(model, values, columns):
    """values: (n_rows, len(columns)); columns index the fitted full-dimensional space."""
    values = np.asarray(values, dtype=float)
    components = []
    for k, weight in enumerate(model.weights_):
        cov = model.covariances_[k][np.ix_(columns, columns)]
        delta = values - model.means_[k, columns]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError('Covariance must be positive definite')
        distance = np.einsum('ij,jk,ik->i', delta, np.linalg.inv(cov), delta)
        components.append(np.log(weight) - .5 * (len(columns)*np.log(2*np.pi) + logdet + distance))
    return -logsumexp(np.stack(components), axis=0)

def observed_windows(frame, sensors, mean, scale, n_points=4):
    """Never join two clips; flatten oldest to newest, sensors in fixed order."""
    arrays = []
    for _, clip in frame.groupby('clip_id', sort=False):
        z = (clip[sensors].to_numpy() - mean) / scale
        arrays.extend(z[i-n_points+1:i+1].ravel() for i in range(n_points-1, len(z)))
    if not arrays:
        raise ValueError('No complete training/calibration windows')
    return np.asarray(arrays)
