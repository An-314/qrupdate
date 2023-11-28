import numpy as np
from scipy.linalg import lapack


def dqhqr(m, n, R):
    """
    Purpose:
        Given an m-by-n upper Hessenberg matrix R, this subroutine updates
        R to upper trapezoidal form using min(m-1, n) Givens rotations.

    Arguments:
    m (int): Number of rows of the matrix R.
    n (int): Number of columns of the matrix R.
    R (2D array): Upper Hessenberg matrix R.

    Returns:
    R (2D array): Updated upper trapezoidal matrix.
    c, s (1D arrays): Rotation cosines and sines.
    """

    if m == 0 or m == 1 or n == 0:
        return R, np.array([]), np.array([])

    # Check arguments
    if m < 0 or n < 0 or R.shape[0] < m:
        raise ValueError("Invalid arguments in DQHQR")

    min_mn = min(m - 1, n)
    c = np.zeros(min_mn)
    s = np.zeros(min_mn)

    for i in range(n):
        t = R[0, i]
        ii = min(m, i + 1)
        for j in range(ii - 1):
            R[j, i], t = c[j] * t + s[j] * R[j + 1, i], c[j] * R[j + 1, i] - s[j] * t
        if ii < m:
            # Generate next rotation
            c[i], s[i], R[ii - 1, i] = lapack.dlartg(t, R[ii, i])
            R[ii, i] = 0.0
        else:
            R[ii - 1, i] = t

    return R, c, s


# `scipy.linalg.lapack.dlartg` 用于生成 Givens 旋转的余弦和正弦值。
