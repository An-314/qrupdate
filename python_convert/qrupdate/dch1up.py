import numpy as np
from scipy.linalg import lapack

def dch1up(R, u):
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

    n = R.shape[0]
    w = np.zeros(n)
    v = np.zeros(n)

    # for i in range(n):
    #     # Apply stored rotations, column-wise
    #     ui = u[i]
    #     for j in range(i):
    #         t = w[j] * R[j, i] + v[j] * ui
    #         ui = w[j] * ui - v[j] * R[j, i]
    #         R[j, i] = t

    #     # Generate next rotation
    #     w[i], v[i], R[i, i] = lapack.dlartg(R[i, i], ui)

    rot = np.eye(n + 1)
    for i in range(n):
        ui = u[0]
        if i > 0 :
            t = R[:i, i]
            t = t[:, None]
            t = np.r_[(t, np.zeros([n - i, 1]))] 
            t = np.r_[(u[i] * np.ones([1, 1]), t)]
            # Apply stored rotations, column-wise
            single_rot = np.eye((n + 1))
            single_rot[i - 1, i - 1] = v[i - 1]
            single_rot[i, i - 1] = w[i - 1]
            single_rot[i - 1, i] = w[i - 1]
            single_rot[i, i] = -v[i - 1]
            rot = np.dot(single_rot, rot)
            t = np.dot(rot, t)
            ui = t[i, :]
            t = np.r_[t[:i, :], t[i + 1:, :]]
            t = t.ravel()
            R[:i, i] = t[:i]

        # Generate next rotation
        w[i], v[i], R[i, i] = lapack.dlartg(R[i, i], ui)

    return R