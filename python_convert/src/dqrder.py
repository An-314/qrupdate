import numpy as np
import dqrtv1, dqrot, dqrqh


def dqrder(m, n, Q, R, j):
    """
    Purpose:
        Updates a QR factorization after deleting a row.
        Given an m-by-m orthogonal matrix Q and an m-by-n upper trapezoidal
        matrix R, this subroutine updates Q -> Q1 and R -> R1.

    Arguments:
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    Q (2D array): Orthogonal matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    j (int): The position of the deleted row.
    """

    if m == 1:
        return Q, R

    # Check arguments
    if m < 1 or j < 1 or j > m:
        raise ValueError("Invalid arguments in DQRDER")

    w = np.zeros(2 * m)

    # Eliminate Q[j, 2:m]
    w[:m] = Q[j - 1, :]
    w[m:] = dqrtv1(m, w[:m])

    # Apply rotations to Q
    Q = dqrot("B", m, m, Q, w[m:], w[1:])

    # Form Q1
    for k in range(m - 1):
        if j > 1:
            Q[: j - 1, k] = Q[: j - 1, k + 1]
        if j < m:
            Q[j - 1 :, k] = Q[j:, k + 1]

    # Apply rotations to R
    dqrqh(m, n, R, w[m:], w[1:])

    # Form R1
    for k in range(n):
        R[: m - 1, k] = R[1:m, k]

    return Q, R
