import numpy as np


def dqrshc(m, n, k, Q, R, i, j):
    """
    Purpose:
        Updates a QR factorization after circular shift of columns.
        Given an m-by-k orthogonal matrix Q and a k-by-n upper
        trapezoidal matrix R, this subroutine updates the matrix Q -> Q1
        and R -> R1 so that Q1 is again orthogonal, R1 upper trapezoidal.

    Arguments:
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    k (int): Number of columns of Q1 and rows of R1.
    Q (2D array): Unitary m-by-k matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    i, j (int): Indices determining the range of column shift.
    """

    if m == 0 or n == 1:
        return Q, R

    # Check arguments
    if (
        m < 0
        or n < 0
        or (k != m and (k != n or n > m))
        or i < 1
        or i > n
        or j < 1
        or j > n
    ):
        raise ValueError("Invalid arguments in DQRSHC")

    w = np.zeros(2 * k)
    i -= 1  # Adjust for 0-based indexing
    j -= 1

    if i < j:
        # Shift columns
        w[:k] = R[:, i].copy()
        R[:, i:j] = R[:, i + 1 : j + 1]
        R[:, j] = w[:k]

        # Retriangularize (requires dqhqr implementation)
        if i < k:
            # Call dqhqr and dqrot (or their Python equivalents)
            pass  # Implement dqhqr and dqrot logic here

    elif j < i:
        # Shift columns
        w[:k] = R[:, i].copy()
        R[:, j + 1 : i + 1] = R[:, j:i]
        R[:, j] = w[:k]

        # Retriangularize (requires dqrtv1, dqrqh, dqrot implementation)
        if j < k:
            # Call dqrtv1, dqrqh, dqrot (or their Python equivalents)
            pass  # Implement dqrtv1, dqrqh, dqrot logic here

    return Q, R
