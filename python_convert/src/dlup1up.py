import numpy as np


def dlup1up(m, n, L, R, p, u, v):
    """
    Purpose:
        Updates a row-pivoted LU factorization after a rank-1 modification.
        Given an m-by-k lower-triangular matrix L with unit diagonal,
        a k-by-n upper-trapezoidal matrix R, and a permutation vector p,
        this subroutine updates L -> L1, R -> R1 and p -> P1 so that
        L is again lower unit triangular, R upper trapezoidal,
        and P1'*L1*R1 = P'*L*R + u*v'.

    Arguments:
    m (int): Order of the matrix L.
    n (int): Number of columns of the matrix R.
    L (2D array): Unit lower triangular matrix L.
    R (2D array): Upper trapezoidal matrix R.
    p (1D array): Permutation vector representing P.
    u, v (1D arrays): The left m-vector and right n-vector.
    """

    k = min(m, n)
    if k == 0:
        return

    # Check arguments
    if m < 0 or n < 0 or L.shape[0] < m or R.shape[0] < k:
        raise ValueError("Invalid arguments in DLUP1UP")

    tau = 1e-1  # Pivot threshold
    w = np.zeros(m)

    # Form L \ P*u
    w[:] = u[p]
    w[:k] = np.linalg.solve(L[:k, :k], w[:k])

    # Subtract the trailing part if m > k
    if m > k:
        w[k:] -= L[k:, :k] @ w[:k]

    # Backward substitution and pivoting
    for j in range(k - 1, 0, -1):
        # Pivoting condition
        if abs(w[j]) < tau * abs(L[j + 1, j] * w[j] + w[j + 1]):
            # Swap j and j+1 in p, L, R, and w
            p[j], p[j + 1] = p[j + 1], p[j]
            L[[j, j + 1]], R[[j, j + 1]] = L[[j + 1, j]], R[[j + 1, j]]
            L[:, [j, j + 1]], R[:, [j, j + 1]] = L[:, [j + 1, j]], R[:, [j + 1, j]]
            w[j], w[j + 1] = w[j + 1], w[j]

            # Make L lower triangular again and update R
            tmp = -L[j, j + 1]
            L[j:, j + 1] += tmp * L[j:, j]
            R[j:, j] += tmp * R[j:, j]

        # Eliminate w[j+1]
        tmp = w[j + 1] / w[j]
        w[j + 1] = 0
        R[j + 1, j:] -= tmp * R[j, j:]
        L[j + 1, j:] += tmp * L[j, j:]

    # Add a multiple of v to R
    R[0, :] += w[0] * v

    # Forward sweep for pivoting
    for j in range(k - 1):
        # Pivoting condition
        if abs(R[j, j]) < tau * abs(L[j + 1, j] * R[j, j] + R[j + 1, j]):
            # Swap j and j+1 in p, L, R
            p[j], p[j + 1] = p[j + 1], p[j]
            L[[j, j + 1]], R[[j, j + 1]] = L[[j + 1, j]], R[[j + 1, j]]
            L[:, [j, j + 1]], R[:, [j, j + 1]] = L[:, [j + 1, j]], R[:, [j + 1, j]]

            # Make L lower triangular again and update R
            tmp = -L[j, j + 1]
            L[j:, j + 1] += tmp * L[j:, j]
            R[j:, j] += tmp * R[j:, j]

        # Eliminate R[j+1, j]
        tmp = R[j + 1, j] / R[j, j]
        R[j + 1, j] = 0
        R[j + 1, j + 1 :] -= tmp * R[j, j + 1 :]
        L[j + 1, j:] += tmp * L[j, j:]

    # Complete the update by updating the lower part of L
    if m > k:
        w[:k] = v
        w[:k] = np.linalg.solve(R[:k, :k].T, w[:k])
        L[k:, :k] += np.outer(w[k:], w[:k])

    return L, R, p
