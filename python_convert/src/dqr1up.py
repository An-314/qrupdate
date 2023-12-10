import numpy as np
from scipy.linalg import blas, lapack
import dch1up, dqrqh, dqhqr, dqrot, dqrtv


def dqr1up(m, n, k, Q, R, u, v):
    """
    Purpose:
        Updates a QR factorization after a rank-1 modification.
        Given an m-by-k orthogonal Q and an m-by-n upper trapezoidal R,
        updates Q -> Q1 and R -> R1 so that Q1*R1 = Q*R + u*v'.

    Arguments:
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    k (int): Number of columns of Q, and rows of R.
    Q (2D array): Orthogonal matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    u (1D array): Left m-vector.
    v (1D array): Right n-vector.
    """

    if k == 0 or n == 0:
        return Q, R

    # Check arguments
    if m < 0 or n < 0 or (k != m and (k != n or n > m)):
        raise ValueError("Invalid arguments in DQR1UP")

    full = k == m
    w = np.zeros(2 * k)
    ru = np.linalg.norm(u) if not full else None

    # Form Q'*u and also u - Q*Q'*u in non-full case
    for i in range(k):
        w[i] = np.dot(Q[:, i], u)
        if not full:
            u -= w[i] * Q[:, i]

    # Generate rotations to eliminate Q'*u
    w, _ = dqrtv(k, w)

    # Apply rotations to R
    dqrqh(k, n, R, w, w[1:k])

    # Apply rotations to Q
    dqrot("B", m, k, Q, w, w[1:k])

    # Update the first row of R
    R[0, :] += w[0] * v

    # Retriangularize R
    dqhqr(k, n, R, w)

    # Apply rotations to Q
    dqrot("F", m, min(k, n) + 1, Q, w, w)

    if full:
        return Q, R

    # Update the orthogonal basis if needed
    ruu = np.linalg.norm(u)
    if ruu > ru * np.finfo(float).eps:
        v *= ruu
        u /= ruu
        dch1up(n, R, v, w)
        for i in range(n):
            drot(m, Q[:, i], u, w[k + i], v[i])

    return Q, R
