import numpy as np


def dlu1up(m, n, L, R, u, v):
    """
    Purpose:
        Updates an LU factorization after a rank-1 modification.
        Given an m-by-k lower-triangular matrix L with unit diagonal
        and a k-by-n upper-trapezoidal matrix R (k = min(m, n)),
        this subroutine updates L -> L1 and R -> R1 so that
        L is again lower unit triangular, R upper trapezoidal,
        and L1*R1 = L*R + u*v'.

    Arguments:
    m (int): Order of the matrix L.
    n (int): Number of columns of the matrix R.
    L (2D array): Unit lower triangular matrix L.
    R (2D array): Upper trapezoidal matrix R.
    u, v (1D arrays): The left m-vector and right n-vector.
    """

    k = min(m, n)
    if k == 0:
        return

    # Check arguments
    if m < 0 or n < 0 or L.shape[0] < m or R.shape[0] < k:
        raise ValueError("Invalid arguments in DLU1UP")

    # The Bennett algorithm, modified for column-major access.
    # The leading part.
    for i in range(k):
        ui = u[i]
        vi = v[i]
        # Delayed R update
        for j in range(i):
            R[j, i] += u[j] * vi
            vi -= v[j] * R[j, i]
        # Diagonal update
        R[i, i] += ui * vi
        vi /= R[i, i]
        # L update
        for j in range(i + 1, m):
            u[j] -= ui * L[j, i]
            L[j, i] += u[j] * vi
        u[i] = ui
        v[i] = vi

    # Finish the trailing part of R if needed.
    for i in range(k, n):
        vi = v[i]
        for j in range(k):
            R[j, i] += u[j] * vi
            vi -= v[j] * R[j, i]
        v[i] = vi
