import numpy as np


def dqrot(dir, m, n, Q, c, s):
    """
    Purpose:
        Apply a sequence of inverse rotations to the matrix Q.

    Arguments:
    dir (char): 'B'/'b' for backward, 'F'/'f' for forward rotations.
    m (int): Number of rows of matrix Q.
    n (int): Number of columns of the matrix Q.
    Q (2D array): The matrix Q.
    c (1D array): n-1 rotation cosines.
    s (1D array): n-1 rotation sines.
    """

    if m == 0 or n == 0 or n == 1:
        return Q

    # Check arguments
    if m < 0 or n < 0 or Q.shape[0] < m:
        raise ValueError("Invalid arguments in DQROT")

    # Apply rotations
    forward = dir in ["F", "f"]
    if forward:
        for i in range(n - 1):
            Q[:, i], Q[:, i + 1] = (
                c[i] * Q[:, i] + Q[:, i + 1] * s[i],
                c[i] * Q[:, i + 1] - Q[:, i] * s[i],
            )
    else:
        for i in range(n - 2, -1, -1):
            Q[:, i], Q[:, i + 1] = (
                c[i] * Q[:, i] + s[i] * Q[:, i + 1],
                c[i] * Q[:, i + 1] - s[i] * Q[:, i],
            )

    return Q
