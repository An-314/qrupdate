import numpy as np
import scipy.linalg.lapack as lapack
import matplotlib

matplotlib.use("Agg")  # 设置非交互式后端
import matplotlib.pyplot as plt


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
            tau = f / g
            sn = 1 / np.sqrt(1 + tau**2)
            cs = sn * tau
        else:
            tau = g / f
            cs = 1 / np.sqrt(1 + tau**2)
            sn = cs * tau
    return cs, sn


def test():
    # 随机生成一个1000 - 10000之间的float64
    f = np.random.uniform(100, 1000)
    # 随机生成一个0 - 100之间的float64
    g = np.random.uniform(0, 1)
    print(f, g)
    cs1, sn1, _ = lapack.dlartg(f, g)
    cs2, sn2 = dlartg(f, g)
    print(cs1 - cs2, sn1 - sn2)
    return f, g, cs1 - cs2, sn1 - sn2


if __name__ == "__main__":
    # 生成四个长为k的list
    k = 100
    f, g, cs, sn = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
    for i in range(k):
        f[i], g[i], cs[i], sn[i] = test()
    # 绘制散点图：横坐标为0-k，纵坐标为f，g
    plt.plot(f)
    plt.plot(g)
    plt.xlabel("k")
    plt.ylabel("f, g")
    plt.legend(["f", "g"])
    plt.title("given f and g")
    plt.savefig("f_g.png")
    # 绘制误差图
    plt.cla()
    plt.plot(cs)
    plt.plot(sn)
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend(["cs", "sn"])
    plt.title("error of cs and sn")
    plt.savefig("cs_sn.png")
