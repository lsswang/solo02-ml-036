"""Existing observational conditioning; no causal intervention semantics."""
import numpy as np

def distribution(model):
    B=np.asarray(model['B']);b=np.asarray(model['bias']);v=np.asarray(model['variance'])
    L=np.linalg.solve(np.eye(len(b))-B,np.eye(len(b)))
    return L@b,(L*v)@L.T

def condition(model,evidence):
    mean,cov=distribution(model);ids=np.array(sorted(map(int,evidence)),dtype=int)
    if not len(ids):return mean,cov
    y=np.array([evidence[str(i)] if str(i) in evidence else evidence[i] for i in ids])
    cross=cov[:,ids];gain=np.linalg.solve(cov[np.ix_(ids,ids)],cross.T).T
    return mean+gain@(y-mean[ids]),cov-gain@cross.T
