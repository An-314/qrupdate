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
    
    输入参数 向量u,向量长度n(可优化掉)
    输出参数 新向量u(只有第一个分量非零,且保模长),
             givens参数向量w(cos),v(sin)(长度n,实际只有前n-1个分量有效)
    """
    n = len(u)
    if n <= 1:
        return u, np.array([])

    w = np.zeros(n)
    v = np.zeros(n)
    rr = u[n - 1]

    for i in range(n - 2, -1, -1):
        c, s, rr = lapack.dlartg(u[i], rr)
        w[i] = c
        v[i] = s
        u[i + 1] = 0

    u[0] = rr

    return u, w , v
