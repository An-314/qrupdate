import numpy as np
import sys

sys.path.append(
    "/home/anzrew/Documents/qrupdate/src/build/lib.linux-x86_64-cpython-311"
)
import qrupdate


# import numpy as np
# import sys

# sys.path.append("lib")
# import qrupdate


def appending_column(Q, R, j, x):
    """
    对矩阵A的完全QR分解，传入QR与加入的列x，将x加入A的第j列得到A1，返回A1的QR分解

    Parameters
    ----------
    Q : (m, m)
        正交矩阵
    R : (m, n)
        上三角矩阵
    j : int
        加入的列的位置
    x : (m,)
        加入的列

    Returns
    -------
    Q : (m, m)
        正交矩阵
    R : (m, n+1)
        上三角矩阵
    """
    m, n = R.shape
    k = n
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")
    x = x.astype(np.float64, order="F")
    R = np.append(R, np.zeros((m, 1)), axis=1)

    qrupdate.dqrinc(k, Q, R, j, x)

    Q = Q.copy(order="C")
    R = R.copy(order="C")
    return Q, R


def deleting_column(Q, R, j):
    """
    对矩阵A的完全QR分解，传入QR，将A的第j列删除得到A1，返回A1的QR分解

    Parameters
    ----------
    Q : (m, m)
        正交矩阵
    R : (m, n)
        上三角矩阵
    j : int
        删除的列的位置

    Returns
    -------
    Q : (m, m)
        正交矩阵
    R : (m, n-1)
        上三角矩阵
    """
    m, n = R.shape
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")

    qrupdate.dqrdec(Q, R, j)

    Q = Q.copy(order="C")
    R = R.copy(order="C")
    R = R[:, :-1]
    return Q, R


def appending_row(Q, R, j, x):
    """
    对矩阵A的完整QR分解，传入QR与加入的行x，将x加入A的第j行得到A1，返回A1的QR分解

    Parameters
    ----------
    Q : (m, m)
        正交矩阵
    R : (m, n)
        上三角矩阵
    j : int
        加入的行的位置
    x : (n,)
        加入的行

    Returns
    -------
    Q : (m+1, m+1)
        正交矩阵
    R : (m+1, n)
        上三角矩阵
    """
    m, n = R.shape
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")
    x = x.astype(np.float64, order="F")

    Q = np.append(Q, np.zeros((1, m)), axis=0)
    Q = np.append(Q, np.zeros((m + 1, 1)), axis=1)
    R = np.append(R, np.zeros((1, n)), axis=0)

    w = np.zeros(2 * n).astype(np.float64, order="F")
    qrupdate.dqrinr(m, Q, R, j, x, w)

    Q = Q.copy(order="C")
    R = R.copy(order="C")
    return Q, R


def deleting_row(Q, R, j):
    """
    对矩阵A的完整QR分解，传入QR，将A的第j行删除得到A1，返回A1的QR分解

    Parameters
    ----------
    Q : (m, m)
        正交矩阵
    R : (m, n)
        上三角矩阵
    j : int
        删除的行的位置

    Returns
    -------
    Q : (m-1, m-1)
        正交矩阵
    R : (m-1, n)
        上三角矩阵
    """
    m, n = R.shape
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")

    w = np.zeros(2 * m).astype(np.float64, order="F")
    qrupdate.dqrder(Q, R, j, w)

    Q = Q.copy(order="C")
    R = R.copy(order="C")
    Q = Q[:-1, :-1]
    R = R[:-1, :]
    return Q, R


def cholesky_update(R, j, x):
    """
    对A的Cholesky分解进行更新，A1=A+x*x.T，返回A1的Cholesky分解
    """
    R = R.astype(np.float64, order="F")
    x = x.astype(np.float64, order="F")
    w = np.zeros(j).astype(np.float64, order="F")
    qrupdate.dch1up(R, x, w)
    R = R.copy(order="C")
    return R


def cholesky_downdate(R, j, x):
    """
    对A的Cholesky分解进行降级，A1=A-x*x.T，返回A1的Cholesky分解
    """
    R = R.astype(np.float64, order="F")
    x = x.astype(np.float64, order="F")
    w = np.zeros(j).astype(np.float64, order="F")
    qrupdate.dch1dn(R, x, w, 0)
    R = R.copy(order="C")
    return R
