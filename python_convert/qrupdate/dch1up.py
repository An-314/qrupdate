import numpy as np
from scipy.linalg import lapack


def dch1up(n, R, u):
    """
    Purpose:
        Given an upper triangular matrix R that is a Cholesky
        factor of a symmetric positive definite matrix A, i.e.,
        A = R'*R, this subroutine updates R -> R1 so that
        R1'*R1 = A + u*u' (real version)

    Arguments:
    n (int): the order of matrix R
    R (2D array): on entry, the upper triangular matrix R
                  on exit, the updated matrix R1
    u (1D array): the vector determining the rank-1 update
                  on exit, u contains the rotation sines
                  used to transform R to R1.
    w (1D array): cosine parts of rotations.
    """

    w = np.zeros(n)
    v = np.zeros(n)

    for i in range(n):
        # Apply stored rotations, column-wise
        ui = u[i]
        for j in range(i):
            t = w[j] * R[j, i] + v[j] * ui
            ui = w[j] * ui - v[j] * R[j, i]
            R[j, i] = t
        if i < n :
            # Generate next rotation
            w[i], v[i], R[i, i] = lapack.dlartg(R[i, i], ui)

    return R
