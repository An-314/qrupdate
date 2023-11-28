import numpy as np
from scipy.linalg import blas
import dqrtv1, dqrqh, dqhqr


def dchshx(n, R, i, j):
    """
    Purpose:
        Given an upper triangular matrix R, this subroutine swaps
        columns i and j and retriangularizes the matrix.

    Arguments:
    n (int): the order of matrix R.
    R (2D array): on entry, the upper triangular matrix R.
                  on exit, the updated matrix R.
    i, j (int): the positions of columns to be swapped.
    """

    # Check arguments
    if n < 0 or i < 1 or i > n or j < 1 or j > n:
        raise ValueError("Invalid arguments in DCHSHX")

    # Quick return if possible
    if n == 0 or n == 1:
        return

    i -= 1  # Adjust for 0-based indexing
    j -= 1

    w = np.zeros(n)

    if i < j:
        # Shift columns
        w[:] = R[:, i]
        R[:, i:j] = R[:, i + 1 : j + 1]
        R[:, j] = w

        # Retriangularize using dqhqr
        R[i:, i:], _ = dqhqr(n - i, n - i, R[i:, i:], R.shape[1], np.zeros(n - i))

    elif j < i:
        # Shift columns
        w[:] = R[:, i]
        R[:, j + 1 : i + 1] = R[:, j:i]
        R[:, j] = w

        # Eliminate the introduced spike and apply rotations to R
        R[j:, j:], _ = dqrtv1(n - j, R[j:, j:], np.zeros(n - j))
        R[j:, j + 1 :], _ = dqrqh(
            n - j, n - j - 1, R[j:, j + 1 :], R.shape[1], np.zeros(n - j)
        )

        # Zero spike
        R[j + 1 :, j] = 0

    return
