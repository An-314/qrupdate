import numpy as np
import time
from scipy.linalg import lapack
from scipy.linalg import solve_triangular


def dch1dn(R, u):
    q = solve_triangular(R, u, trans=1)
    v = np.zeros(u.shape[0])
    r = (1 - np.linalg.norm(q) ** 2) ** 0.5
    c, s, r = lapack.dlartg(r, q[u.shape[0] - 1])
    v, R[u.shape[0] - 1] = v * c + R[u.shape[0] - 1] * s, -v * s + R[u.shape[0] - 1] * c
    for i in range(u.shape[0] - 1, 0, -1):
        c, s, r = lapack.dlartg(r, q[i - 1])
        v, R[i - 1] = v * c + R[i - 1] * s, -v * s + R[i - 1] * c
    return R


def dch2dn(R, u):
    column = solve_triangular(R, u, trans=1)
    row = np.zeros(u.shape[0] + 1)
    row[0] = (1 - np.linalg.norm(column) ** 2) ** 0.5
    R = np.hstack((column.reshape(u.shape[0], 1), R))
    R = np.vstack((R, row.reshape(1, u.shape[0] + 1)))
    for i in range(u.shape[0], 0, -1):
        c, s, r = lapack.dlartg(R[i - 1, 0], R[i, 0])
        R[i - 1, :], R[i, :] = (
            R[i - 1, :] * c + R[i, :] * s,
            -R[i - 1, :] * s + R[i, :] * c,
        )
    return R[1:, 1:]


def dch2up(R, u):
    R_plus = np.vstack((u, R))
    for i in range(u.shape[0]):
        c, s, _ = lapack.dlartg(R[i, i], R_plus[0, i])
        G = np.array([[c, s], [-s, c]])
        R_plus[[i + 1, 0], :] = np.dot(G, R_plus[[i + 1, 0], :])
    return R_plus[1:, :]


def dch1up(R, u):
    for i in range(u.shape[0]):
        c, s, _ = lapack.dlartg(R[i, i], u[i])
        R[i, :], u = R[i, :] * c + u * s, -R[i, :] * s + u * c
    return R
