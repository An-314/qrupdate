import numpy as np
from scipy.linalg import blas, lapack

def dch1dn(R, u):
    """
    purpose:      given an upper triangular matrix R that is a Cholesky
                factor of a hermitian positive definite matrix A, i.e.
                A = R'*R, this subroutine downdates R -> R1 so that
                R1'*R1 = A - u*u'
                (real version)
    arguments:
    c n (in)        the order of matrix R
    R (io)        on entry, the upper triangular matrix R
                on exit, the updated matrix R1
    ldr (in)      leading dimension of R. ldr >= n.
    u (io)        the vector determining the rank-1 updateon exit, u contains the reflector sinesused to transform R to R1.
    w (out)       cosine parts of reflectors.
    importlibnfo (out)    on exit, error code:
                info = 0: success.
                info = 1: update violates positive-definiteness.
                info = 2: R is singular.
    """

    n = R.shape[0]

    # Form R' \ u using BLAS dtrsv
    u = blas.dtrsv(R, u, trans=1, lower=0)

    # Check positive definiteness
    rho = np.linalg.norm(u)
    rho = 1 - rho**2
    rho = np.sqrt(rho)

    # Eliminate R' \ u and apply rotations
    w = np.zeros(n)
    v = np.zeros(n)
    rr = np.zeros(n + 1)
    rr[n] = rho

    for i in reversed(range(n)):
        w[i], v[i], rr[i] = lapack.dlartg(rr[i + 1], u[i])

    # for i in reversed(range(n)):
    #     ui = 0
    #     for j in reversed(range(i + 1)):
    #         t = w[j] * ui + v[j] * R[j, i]
    #         R[j, i] = w[j] * R[j, i] - v[j] * ui
    #         ui = t

    # return R
    
    rot = np.eye(n + 1)
    
    for i in range(n):
        single_rot = np.eye(n + 1)
        single_rot[i, i] = v[i]
        single_rot[i + 1, i] = w[i]
        single_rot[i, i + 1] = w[i]
        single_rot[i + 1, i + 1] = -v[i]
        rot = np.dot(rot, single_rot)
        
    R = np.r_[(R, np.zeros([1, n]))]
    R = np.dot(rot, R)

    return R[1:, :]