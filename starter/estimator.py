"""Existing complete, observational linear-Gaussian regression."""
import numpy as np

def fit_observational(rows, parents):
    complete=[r['values'] for r in rows if not r.get('do') and all(x is not None for x in r['values'])]
    if not complete:raise ValueError('no complete observational rows')
    x=np.asarray(complete,dtype=float);d=x.shape[1];B=np.zeros((d,d));bias=np.zeros(d);variance=np.zeros(d)
    for j,ps in enumerate(parents):
        design=np.column_stack([np.ones(len(x)),x[:,ps]])
        beta=np.linalg.lstsq(design,x[:,j],rcond=None)[0];bias[j]=beta[0];B[j,ps]=beta[1:]
        variance[j]=max(float(np.mean((x[:,j]-design@beta)**2)),1e-6)
    return {'B':B.tolist(),'bias':bias.tolist(),'variance':variance.tolist()}
