import numpy as np
import sys
sys.path.append('/home/anzrew/Documents/qrupdate/src/build/lib.linux-x86_64-cpython-311')
import qrupdate

def appending_row(Q,R,j,x):
    """
    对矩阵A的完整QR分解，传入QR与加入的行x，将x加入A的第j行得到A1，返回A1的QR分解
    """
    m, n = R.shape
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")

    Q = np.append(Q, np.zeros((1, m)), axis=0)
    Q = np.append(Q, np.zeros((m + 1, 1)), axis=1)
    R = np.append(R, np.zeros((1, n)), axis=0)

    w = np.zeros(2*n).astype(np.float64, order="F")
    qrupdate.dqrinr(m,Q,R,j,x,w)

    Q = Q.copy(order='C')
    R = R.copy(order='C')
    return Q, R

def deleting_row(Q,R,j):
    """
    对矩阵A的完整QR分解，传入QR，将A的第j行删除得到A1，返回A1的QR分解
    """
    m, n = R.shape
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")

    w = np.zeros(2*m).astype(np.float64, order="F")
    qrupdate.dqrder(Q,R,j,w)

    Q = Q[:-1,:-1]
    R = R[:-1, :]

    Q = Q.copy(order='C')
    R = R.copy(order='C')
    return Q, R

def cholesky_update(R, j, x):
    """
    对A的Cholesky分解进行更新，A1=A+x*x.T，返回A1的Cholesky分解
    """
    R = R.astype(np.float64, order="F")
    w = np.zeros(j).astype(np.float64, order="F")
    qrupdate.dch1up(R, x, w)
    R = R.copy(order='C')
    return R

def cholesky_downdate(R, j, x):
    """
    对A的Cholesky分解进行降级，A1=A-x*x.T，返回A1的Cholesky分解
    """
    R = R.astype(np.float64, order="F")
    w = np.zeros(j).astype(np.float64, order="F")
    qrupdate.dch1dw(R, x, w)
    R = R.copy(order='C')
    return R