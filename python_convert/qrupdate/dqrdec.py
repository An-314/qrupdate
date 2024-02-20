import numpy as np
from .dqhqr import dqhqr
from .dqrot import dqrot


def dqrdec(Q, R, j):
    """
    Purpose:
        Updates a QR factorization after deleting a column.
        Given an m-by-k orthogonal matrix Q and a k-by-n upper
        trapezoidal matrix R, this subroutine updates Q -> Q1 and R -> R1.

    Arguments:
    Q (2D array): Orthogonal matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    j (int): The position of the deleted column in R.
    """
    """
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    k (int): Number of columns of Q, and rows of R.
    """

    m = Q.shape[0]
    n = R.shape[1]
    k = Q.shape[1]

    if m == 0 or n == 0:
        return Q, R

    # Check arguments
    if m < 0 or n < 0 or (k != m and (k != n or n >= m)) or j < 1 or j > n + 1:
        raise ValueError("Invalid arguments in DQRDEC")

    # Delete the j-th column
    R[:, j - 1 : n - 1] = R[:, j:n]

    # Retriangularize
    if k == m:
        if j < k:
            R[j - 1 :, j - 1 :], w, v = dqhqr(k - j + 1, n - j, R[j - 1 :, j - 1 :])
            # Apply rotations to Q
            Q[:, j - 1 :] = dqrot("F", m, min(k, n) - j + 1, Q[:, j - 1 :], w, v)

        return Q, R[:, : n - 1]

    else:
        if j < k:
            R[j - 1 : n, j - 1 : n - 1], w, v = dqhqr(
                n - j + 1, n - j, R[j - 1 : n, j - 1 : n - 1]
            )
            # Apply rotations to Q
            Q[:, j - 1 : n] = dqrot("F", m, n - j + 1, Q[:, j - 1 : n], w, v)

        return Q[:, : n - 1], R[: n - 1, : n - 1]
