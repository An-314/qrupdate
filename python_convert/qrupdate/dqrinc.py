import numpy as np
from .dqrtv1 import dqrtv1
from .dqrqh import dqrqh
from .dqrot import dqrot
from .dgqvec import dgqvec


def dqrinc( Q, R, j, x):
    """
    Purpose:
        Updates a QR factorization after inserting a new column.
        Given an m-by-k orthogonal matrix Q and an m-by-n upper
        trapezoidal matrix R, this subroutine updates Q -> Q1 and R -> R1.

    Arguments:
    Q (2D array): Orthogonal matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    j (int): The position of the new column in R1.
    x (1D array): The column being inserted.
    """

    '''
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    k (int): Number of columns of Q, and rows of R.
    '''
    m = Q.shape[0]
    n = R.shape[1]
    k = Q.shape[1]

    # Check arguments
    if m < 0 or n < 0 or (k != m and (k != n or n >= m)) or j < 1 or j > n + 1:
        raise ValueError("Invalid arguments in DQRINC")

    full = k == m
    w = np.zeros(k)

    # Insert empty column at j-th position
    R_rows = R.shape[0]
    R = np.c_[R, np.zeros(R_rows)]
    R[:, j : n + 1] = R[:, j - 1 : n]

    # Insert Q'*u into R
    if full:
        k1 = k
        for i in range(k):
            R[i, j - 1] = np.dot((Q[:, i]).T, x)
    else:
        k1 = k + 1
        Q_rows = Q.shape[0]
        Q = np.c_[Q, np.zeros(Q_rows)]
        R_columns = Q.shape[1]
        zero_row = np.zeros((1, R_columns))
        R = np.vstack((R, zero_row))
        for t in range(m):
            Q[t, k] = x[t]
        for i in range(k):
            R[i, j - 1] = np.dot((Q[:, i]).T, Q[:, k])
            Q[:, k] -= R[i, j - 1] * Q[:, i]
        rx = np.linalg.norm(Q[:, k])
        R[k, j - 1] = rx
        if rx == 0:
            Q[:, k] = dgqvec(m, k, Q)
        else:
            Q[:, k] /= rx

    # Eliminate the spike
    if j <= k:
        R[j - 1 :, j - 1], w, v = dqrtv1(k1 + 1 - j, R[j - 1 :, j - 1])
        # Apply rotations to R and Q
        if j <= n:
            R[j - 1 :, j:] = dqrqh(k1 + 1 - j, n - j + 1, R[j - 1 :, j:], w, v)
        Q[:, j - 1 :] = dqrot("B", m, k1 + 1 - j, Q[:, j - 1 :], w, v)

    return Q, R
