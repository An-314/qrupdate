import cupy as np


def dgqvec(m, n, Q):
    """
    Purpose:
        Given an orthogonal m-by-n matrix Q, n < m, generates
        a vector u such that Q'*u = 0 and norm(u) = 1.

    Arguments:
    m (int): number of rows of matrix Q.
    n (int): number of columns of matrix Q.
    Q (2D array): the orthogonal matrix Q.

    Returns:
    u (1D array): the generated vector.
    """

    # Quick return if possible
    if m == 0:
        return np.array([])
    if n == 0:
        u = np.zeros(m)
        u[0] = 1
        return u

    # Check arguments
    if m < 0 or n < 0 or Q.shape[0] < m or Q.shape[1] < n:
        raise ValueError("Invalid arguments in DGQVEC")

    for j in range(n):
        # Probe j-th canonical unit vector
        u = np.zeros(m)
        u[j] = 1

        # Form u - Q*Q'*u
        for i in range(n):
            r = np.dot(Q[:, i], u)
            u -= r * Q[:, i]

        r = np.linalg.norm(u)
        if r != 0:
            u /= r
            return u

    raise RuntimeError("Fatal: impossible condition in DGQVEC")
