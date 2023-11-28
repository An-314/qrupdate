import numpy as np
from scipy.linalg import blas, lapack
import dqhqr


def dchdex(n, R, j):
    """
    Purpose:
        Given an upper triangular matrix R that is a Cholesky
        factor of a symmetric positive definite matrix A, i.e.,
        A = R'*R, this subroutine updates R -> R1 so that
        R1'*R1 = A(jj,jj), where jj = [1:j-1, j+1:n+1] (real version)

    Arguments:
    n (int): the order of matrix R.
    R (2D array): on entry, the original upper trapezoidal matrix R.
                  on exit, the updated matrix R1.
    j (int): the position of the deleted row/column.
    """

    # Check arguments
    if n < 0 or j < 1 or j > n:
        raise ValueError(f"Invalid argument in DCHDEX")

    # Quick return if possible
    if n == 1:
        return R

    j -= 1  # Adjusting for 0-based indexing in Python

    # Delete the j-th column
    R[:, j : n - 1] = R[:, j + 1 : n]

    # Retriangularize
    if j < n - 1:
        # Assuming dqhqr is a function from your library
        R, w = dqhqr(n - j, n - j - 1, R[j:, j:], R.shape[1], R[:, n - 1])

    return R
