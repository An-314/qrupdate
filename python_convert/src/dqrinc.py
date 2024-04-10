import numpy as np
import dqrtv1, dqrqh, dqrot, dgqvec


def dqrinc(m, n, Q, R, j, x):
    """
    Purpose:
        Updates a QR factorization after inserting a new column.
        Given an m-by-k orthogonal matrix Q and an m-by-n upper
        trapezoidal matrix R, this subroutine updates Q -> Q1 and R -> R1.

    Arguments:
    m (int): Number of rows of the matrix Q.
    n (int): Number of columns of the matrix R.
    k (int): Number of columns of Q, and rows of R.
    Q (2D array): Orthogonal matrix Q.
    R (2D array): Upper trapezoidal matrix R.
    j (int): The position of the new column in R1.
    x (1D array): The column being inserted.
    """

    k = Q.shape[1]

    # Check arguments
    if m < 0 or n < 0 or (k != m and (k != n or n >= m)) or j < 1 or j > n + 1:
        raise ValueError("Invalid arguments in DQRINC")

    full = k == m
    w = np.zeros(k)

    # Insert empty column at j-th position
    R_rows = R.shape[0]
    R = np.c_[R, np.zeros(R_rows)]
    #print(R)
    for i in range(n, j-1 , -1):
        R[:,i] = R[:,i-1]       

    # Insert Q'*u into R
    if full:
        k1 = k
        for i in range(k):
            R[i, j - 1] = np.dot((Q[:, i]).T, x)
    else:
        k1 = k + 1
        Q_rows = Q.shape[0]
        Q = np.c_[Q, np.zeros(Q_rows)]
        
        R_columns = Q.shape[1]
        zero_row = np.zeros((1, R_columns))
        R = np.vstack((R, zero_row))
        
        Q[:, k] = x.copy()
        for i in range(k):
            R[i, j-1 ] = np.dot(Q[:, i], Q[:, k])
            Q[:, k] -= R[i, j-1 ] * Q[:, i]
        rx = np.linalg.norm(Q[:, k])
        R[k, j-1 ] = rx
        if rx == 0:
            Q[:, k] = dgqvec.dgqvec(m, k, Q)
        else:
            Q[:, k] /= rx
    #print(R)
    #print(Q)
    # Eliminate the spike
    if j >= k:
        pass
    # Apply rotations to R and Q
    else:
        R[j - 1 :, j - 1], u, w = dqrtv1.dqrtv1(k - j + 2, R[j - 1 :, j - 1])
        dqrqh.dqrqh(k - j + 2, n - j + 1, R[j - 1 :, j:], w, u)
        
    dqrot.dqrot("B", m, k - j + 2, Q[:, j - 1 :], w, u)

    return Q, R

'''
m = 3
n = 2
k = 2
A = np.array([3,0,0,0,0,-1,2,-2,0,2,2,1,0,-2,1,2]).reshape(8,2)
Q, R = np.linalg.qr(A)
x = np.ones(8)
B = np.insert(A, 1, x, axis=1)
Q1, R1 = np.linalg.qr(B)

Q2 , R2 = dqrinc(m, n, k, Q, R, 1, x)

print(Q2)
print(R2)
print(np.dot(Q2.T, Q2))

print(np.dot(Q2, R2))
print(np.c_[x, A])
'''