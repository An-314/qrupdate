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
    # Check for quick return
    if n == 0:
        return R

    # Check arguments
    info = 0
    if n < 0:
        info = -1
    elif R.shape[0] < n or R.shape[1] < n:
        info = -3
    if info != 0:
        raise ValueError(f"Invalid argument in DCH1DN: {info}")

    # Check for singularity of R
    for i in range(n):
        if R[i, i] == 0:
            return R

    # Form R' \ u using BLAS dtrsv
    u = blas.dtrsv(R, u, trans=1, lower=0)

    # Check positive definiteness
    rho = np.linalg.norm(u)
    rho = 1 - rho**2
    if rho <= 0:
        return R
    rho = np.sqrt(rho)

    # Eliminate R' \ u and apply rotations
    w = np.zeros(n)
    v = np.zeros(n)
    for i in reversed(range(n)):
        ui = u[i]
        w[i], v[i], rr = lapack.dlartg(rho, ui)
        rho = rr
    for i in reversed(range(n)):
        ui = 0
        for j in reversed(range(i + 1)):
            t = w[j] * ui + v[j] * R[j, i]
            R[j, i] = w[j] * R[j, i] - v[j] * ui
            ui = t

    return R
