import cupy as np


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
        ii = min(m - 2, i)
        t = R[ii + 1, i]
        for j in range(ii, -1, -1):
            R[j + 1, i], t = (
                c[j] * t - s[j] * R[j, i],
                c[j] * R[j, i] + s[j] * t,
            )
        R[0, i] = t

    return R
