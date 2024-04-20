import numpy as np


def dlartg(f, g):
    # 变量初始化
    first = True
    safmn2 = safmx2 = None

    if first:
        first = False
        safmin = np.finfo(float).tiny  # DLAMCH('S')
        eps = np.finfo(float).eps  # DLAMCH('E')
        safmn2 = safmin ** (
            int(np.log(safmin / eps) / np.log(np.finfo(float).tiny) / 2)
        )
        safmx2 = 1 / safmn2

    if g == 0:
        cs = 1
        sn = 0
        r = f
    elif f == 0:
        cs = 0
        sn = 1
        r = g
    else:
        f1 = f
        g1 = g
        scale = max(abs(f1), abs(g1))
        count = 0
        if scale >= safmx2:
            while scale >= safmx2:
                count += 1
                f1 *= safmn2
                g1 *= safmn2
                scale = max(abs(f1), abs(g1))
            r = np.sqrt(f1**2 + g1**2)
            cs = f1 / r
            sn = g1 / r
            r *= safmx2**count
        elif scale <= safmn2:
            while scale <= safmn2:
                count += 1
                f1 *= safmx2
                g1 *= safmx2
                scale = max(abs(f1), abs(g1))
            r = np.sqrt(f1**2 + g1**2)
            cs = f1 / r
            sn = g1 / r
            r *= safmn2**count
        else:
            r = np.sqrt(f1**2 + g1**2)
            cs = f1 / r
            sn = g1 / r

        if abs(f) > abs(g) and cs < 0:
            cs = -cs
            sn = -sn
            r = -r

    return cs, sn, r
