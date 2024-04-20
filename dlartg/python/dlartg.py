import numpy as np
import scipy.linalg.lapack as lapack


def dlartg(f, g):
    """Computes the modified Givens transformation matrix H which zeros the
    second component of the 2-vector transpose([F, G]).
    Given two real scalars f and g, this routine computes the Givens
    rotation matrix H = |  c s |
                        | -s c |
    which zeros the second component of the 2-vector transpose([f, g]).
    The quantity c**2 + s**2 is returned in the variable 'sn'.
    The value of 'sn' is used if f and g are to be rotated further by
    another Givens rotation.
    """
    if g == 0:
        cs = 1
        sn = 0
    elif f == 0:
        cs = 0
        sn = 1
    else:
        if abs(g) > abs(f):
            tau = -f / g
            sn = 1 / np.sqrt(1 + tau**2)
            cs = sn * tau
        else:
            tau = -g / f
            cs = 1 / np.sqrt(1 + tau**2)
            sn = cs * tau
    return cs, sn


def test():
    # 生成float64的随机数
    np.random.seed(0)
    f = np.random.randn()
    g = np.random.randn()
    print(f, g)

    cs1, sn1, _ = lapack.dlartg(f, g)
    cs2, sn2 = dlartg(f, g)
    print(cs1, sn1)
    print(cs2, sn2)


if __name__ == "__main__":
    test()
