import numpy as np
from scipy.linalg import blas, lapack
import dqrtv1, dqrqh


def dchinx(n, R, j, u):
    """
    Purpose:
        Given an upper triangular matrix R that is a Cholesky
        factor of a symmetric positive definite matrix A, i.e.,
        A = R'*R, this subroutine updates R -> R1 so that
        R1'*R1 = A1, A1(jj,jj) = A, A(j,:) = u', A(:,j) = u,
        jj = [1:j-1, j+1:n+1] (real version)

    Arguments:
    n (int): the order of matrix R.
    R (2D array): on entry, the original upper trapezoidal matrix R.
                  on exit, the updated matrix R1.
    j (int): the position of the inserted row/column.
    u (1D array): on entry, the inserted row/column.
                  on exit, u is destroyed.
    """

    # Check arguments
    if n < 0 or j < 1 or j > n + 1:
        raise ValueError(f"Invalid argument in DCHINX")

    j -= 1  # Adjusting for 0-based indexing in Python

    # Shift vector
    t = u[j]
    u[j:n] = u[j + 1 : n + 1]

    # Check for singularity of R
    if np.any(np.diag(R) == 0):
        return 2

    # Form R' \ u using BLAS dtrsv
    u = blas.dtrsv(R, u, trans=2, lower=0, overwrite_b=0)

    # Check positive definiteness
    rho = np.linalg.norm(u)
    rho = t - rho**2
    if rho <= 0:
        return 1

    # Shift columns
    R[:, j + 1 : n + 1] = R[:, j:n]
    R[j : n + 1, j : n + 1] = 0  # Zero out the new column

    # Copy u to R and update R
    R[:n, j] = u
    R[n, j] = np.sqrt(rho)

    # Retriangularize
    if j < n:
        # Eliminate the introduced spike and apply rotations to R
        R[j:, j:], w = dqrtv1(n - j + 1, R[j:, j:], np.zeros(n - j + 1))
        R[j:, j + 1 :], _ = dqrqh(n - j + 1, n - j, R[j:, j + 1 :], R.shape[1], w)

    return 0
