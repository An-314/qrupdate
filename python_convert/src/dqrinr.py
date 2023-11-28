import numpy as np
import dqhqr, dqrot


def dqrinr(m, n, Q, R, j, x):
    """
    Purpose:
        Updates a QR factorization after inserting a new row.
        Given an m-by-m unitary matrix Q and an m-by-n upper trapezoidal
        matrix R, this subroutine updates Q -> Q1 and R -> R1 so that
        Q1 is again unitary, R1 upper trapezoidal.

    Arguments:
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    Q (2D array): Unitary matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    j (int): The position of the new row in R1.
    x (1D array): The row being added.
    """

    # Check arguments
    if n < 0 or j < 1 or j > m + 1:
        raise ValueError("Invalid arguments in DQRINR")

    # Permute the columns of Q1 and rows of R1
    Q[:, j : m + 1] = Q[:, j - 1 : m]
    Q[j - 1, :] = 0
    Q[j - 1, 0] = 1

    # Set up the new matrix R1
    R[j : m + 1, :] = R[j - 1 : m, :]
    R[j - 1, :] = x

    # Retriangularize R
    w, x = dqhqr(m + 1, n, R)

    # Apply rotations to Q
    Q = dqrot("F", m + 1, min(m, n) + 1, Q, w, x)

    return Q, R
