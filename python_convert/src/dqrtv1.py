import numpy as np
from scipy.linalg import lapack


def dqrtv1(n, u):
    """
    Purpose:
        Generates a sequence of n-1 Givens rotations that eliminate all
        but the first element of a vector u.
    Arguments:
    n (int): The length of the vector u.
    u (1D array): On entry, the vector u. On exit, u[1:n] contains the
                  rotation sines, u[0] contains the remaining element.
    Returns:
    u (1D array): Updated vector u.
    w (1D array): Rotation cosines.
    """

    if n <= 0:
        return u, np.array([])

    w = np.zeros(n-1)
    v = np.zeros(n-1)
    rr = u[n - 1]

    for i in range(n - 2, -1, -1):
        c, s, rr = lapack.dlartg(u[i], rr)
        w[i] = c
        v[i] = s
        u[i + 1] = 0

    u[0] = rr

    return u, v, w
'''
u = np.array([4,3,0])
print(dqrtv1(3,u))
'''