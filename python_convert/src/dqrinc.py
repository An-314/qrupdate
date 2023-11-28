import numpy as np
import dqrtv1, dqrqh, dqrot, dgqvec


def dqrinc(m, n, k, Q, R, j, x):
    """
    Purpose:
        Updates a QR factorization after inserting a new column.
        Given an m-by-k orthogonal matrix Q and an m-by-n upper
        trapezoidal matrix R, this subroutine updates Q -> Q1 and R -> R1.

    Arguments:
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    k (int): Number of columns of Q, and rows of R.
    Q (2D array): Orthogonal matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    j (int): The position of the new column in R1.
    x (1D array): The column being inserted.
    """

    # Check arguments
    if m < 0 or n < 0 or (k != m and (k != n or n >= m)) or j < 1 or j > n + 1:
        raise ValueError("Invalid arguments in DQRINC")

    full = k == m
    w = np.zeros(k)

    # Insert empty column at j-th position
    R[:, j : n + 1] = R[:, j - 1 : n]

    # Insert Q'*u into R
    if full:
        for i in range(k):
            R[i, j - 1] = np.dot(Q[:, i], x)
    else:
        Q[:, k] = x.copy()
        for i in range(k):
            R[i, j - 1] = np.dot(Q[:, i], Q[:, k])
            Q[:, k] -= R[i, j - 1] * Q[:, i]
        rx = np.linalg.norm(Q[:, k])
        R[k, j - 1] = rx
        if rx == 0:
            Q[:, k] = dgqvec(m, k, Q)
        else:
            Q[:, k] /= rx

    # Eliminate the spike
    if j <= k:
        w = dqrtv1(k - j + 2, R[j - 1 :, j - 1])
        # Apply rotations to R and Q
        if j <= n:
            dqrqh(k - j + 2, n - j + 1, R[j - 1 :, j:], w, R[j:, j - 1])
        dqrot("B", m, k - j + 2, Q[:, j - 1 :], w, R[j:, j - 1])

    return Q, R
