import numpy as np
from .dqhqr import dqhqr
from .dqrot import dqrot

def dqrinr( Q, R, j, x):
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

    m = Q.shape[0]
    n = R.shape[1]

    # Check arguments
    if n < 0 or j < 1 or j > m + 1:
        raise ValueError("Invalid arguments in DQRINR")
    
    # Only full QR decomposition

    # Permute the columns of Q1 and rows of R1
    Q_columns = Q.shape[1]
    Q = np.r_[Q, np.zeros([1,Q_columns])]
    Q = np.c_[Q, np.zeros(Q_columns + 1)]
    R_columns = R.shape[1]
    R = np.r_[R, np.zeros([1,R_columns])]
    if j > 1:
        Q[: j - 1, 1 : m + 1] = Q[: j - 1, 0 : m]
    if j <= m:
        Q[j : , 1 : m + 1] = Q[j - 1 : m, 0 : m]
    Q[j - 1, :] = 0
    Q[: , 0] = 0
    Q[j - 1, 0] = 1

    # Set up the new matrix R1
    R[1 : n + 1, :] = R[: n, :]
    R[0] = x

    # Retriangularize R
    R, w, v = dqhqr(m + 1, n, R)

    # Apply rotations to Q
    Q = dqrot("F", m + 1, min(m, n) + 1, Q, w, v)

    return Q, R
