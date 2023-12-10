import numpy as np


def dqrqh(m, n, R, c, s):
    """
    Purpose:
        Brings an upper trapezoidal matrix R into upper Hessenberg form
        using min(m-1, n) Givens rotations.

    Arguments:
    m (int): Number of rows of the matrix R.
    n (int): Number of columns of the matrix R.
    R (2D array): Upper trapezoidal matrix R.
    c (1D array): Rotation cosines.
    s (1D array): Rotation sines.
    """

    if m == 0 or m == 1 or n == 0:
        return R

    # Check arguments
    if m < 0 or n < 0 or R.shape[0] < m:
        raise ValueError("Invalid arguments in DQRQH")

    for i in range(n):
        k = min(m - 1, i) + 1
        t = R[k, i]
        for j in range(k, 0, -1):
            R[j, i], t = (
                c[j - 1] * t - s[j - 1] * R[j - 1, i],
                c[j - 1] * R[j - 1, i] + s[j - 1] * t,
            )
        R[0, i] = t

    return R
