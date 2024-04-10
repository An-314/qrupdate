import numpy as np


def dlartg(f, g, cs, sn):
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
        cs[0] = 1
        sn[0] = 0
        return
    if abs(g) > abs(f):
        tau = -f / g
        sn[0] = 1 / np.sqrt(1 + tau**2)
        cs[0] = sn[0] * tau
    else:
        tau = -g / f
        cs[0] = 1 / np.sqrt(1 + tau**2)
        sn[0] = cs[0] * tau
    return
