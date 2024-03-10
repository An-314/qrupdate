import numpy as np
from scipy.linalg import lapack

def dch2up(R, u):
    for i in range(u.shape[0]):
        c, s, r = lapack.dlartg(R[i, i], u[i])
        R[i,:], u = R[i,:] * c + u * s, -R[i,:] * s + u * c
    return R