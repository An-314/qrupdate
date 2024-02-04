import numpy as np
from .dqrtv1 import dqrtv1
from .dqrqh import dqrqh
from .dqrot import dqrot


def dqrder( Q, R, j):
    """
    Purpose:
        Updates a QR factorization after deleting a row.
        Given an m-by-m orthogonal matrix Q and an m-by-n upper trapezoidal
        matrix R, this subroutine updates Q -> Q1 and R -> R1.

    Arguments:
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    Q (2D array): Orthogonal matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    j (int): The position of the deleted row.
    """

    m = Q.shape[0]
    n = R.shape[1]

    if m == 1:
        return Q, R

    # Check arguments
    if m < 1 or j < 1 or j > m:
        raise ValueError("Invalid arguments in DQRDER")

    u = np.zeros(m)

    # Eliminate Q[j, 2:m]
    for k in range(m):
        u[k] = Q[j - 1, k]
    u, v, w = dqrtv1(m, u)

    # Apply rotations to Q
    Q = dqrot("B", m, m, Q, w, v)
    print(Q)

    # Form Q1
    if j > 1:
        Q[: j - 1, : m - 1] = Q[: j - 1, 1 :]
    if j < m:
        Q[j - 1 : m - 1, : m - 1] = Q[j :, 1 :]

    # Apply rotations to R
    R = dqrqh(m, n, R, w, v)

    # Form R1

    return Q[: m - 1, : m - 1], R[1 :, :]
