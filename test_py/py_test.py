# cd到test-py目录下执行

import cupy as cp
import numpy as np
import sys

sys.path.append("../python_convert/")

import matplotlib.pyplot as plt

import time

import qrupdate


def test_dqrinc():
    # 设置测试参数
    m, n = 6000, 4000
    k = n
    ldq, ldr = m, n + 1
    j = 2  # 插入新列的位置

    # 生成随机矩阵 A 和列向量 x
    A = cp.random.rand(m, n).astype(cp.float64)
    x = cp.random.rand(m).astype(cp.float64)

    ##### 简略QR分解测试 #####
    """
    简要QR分解的Q和R矩阵输入时，python版本不需要预留R和Q的空行和空列
    """
    Q, R = cp.linalg.qr(A)
    Q = Q.astype(cp.float64)
    R = R.astype(cp.float64)
    # R = cp.append(R, cp.zeros((n, 1)), axis=1)
    # R = cp.append(R, cp.zeros((1, n + 1)), axis=0)
    # Q = cp.append(Q, cp.zeros((m, 1)), axis=1)

    print(Q.shape)
    print(R.shape)
    # print("修改之前的Q和R")
    # print(Q)
    # print(R)

    time_start = time.time()

    # 调用 dqrinc 更新 QR 分解
    w = cp.zeros(k).astype(cp.float64)
    Q1, R1 = qrupdate.dqrinc(Q, R, j, x)

    time_end = time.time()

    print("dqrinc time cost", time_end - time_start, "s")

    # 验证 QR 分解的正确性
    A_updated = cp.hstack([A[:, : j - 1], x.reshape(-1, 1), A[:, j - 1 :]])
    # Q1, R1 = Q, R  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    # print("修改之后的Q和R")
    # print(Q1)
    # print(R1)

    cp.savetxt("dqrinc.A_updated.txt", A_updated)
    cp.savetxt("dqrinc.A_reconstructed.txt", A_reconstructed)

    if not cp.allclose(A_updated, A_reconstructed):
        print("dqrinc_simp:QR update failed")

    ##### 完整QR分解测试 #####
    """
    完整QR分解的Q和R矩阵输入时，不需要预留R的空列
    """
    Qp, Rp = cp.linalg.qr(A, mode="complete")
    Qp = Qp.astype(cp.float64)
    Rp = Rp.astype(cp.float64)
    # Rp = cp.append(Rp, cp.zeros((m211, 1)), axis=1)

    print(Qp.shape)
    print(Rp.shape)

    Q1, R1 = qrupdate.dqrinc(Qp, Rp, j, x)
    A_updated = cp.hstack([A[:, : j - 1], x.reshape(-1, 1), A[:, j - 1 :]])
    # Q1, R1 = Qp, Rp  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    cp.savetxt("dqrinc.Af_updated.txt", A_updated)
    cp.savetxt("dqrinc.Af_reconstructed.txt", A_reconstructed)

    if not cp.allclose(A_updated, A_reconstructed):
        print("dqrinc_full:QR update failed")


def test_dqrdec():
    # 设置测试参数
    m, n = 10, 5
    k = n
    ldq, ldr = m, n + 1
    j = 2  # 删除列的位置

    # 生成随机矩阵 A 和列向量 x
    A = cp.random.rand(m, n).astype(cp.float64)

    ##### 简略QR分解测试 #####
    """
    简要QR分解的Q和R矩阵输入时，不需要预留R和Q的空行和空列
    Q需要输入m*n的矩阵，R需要输入n*n的矩阵

    这时候出来的Q和R的矩阵形状和原来一致，但最后一列不是空的（计算时候用到），需要手动去掉
    """
    Qd, Rd = cp.linalg.qr(A)
    Qd = Qd.astype(cp.float64)
    Rd = Rd.astype(cp.float64)

    print(Qd.shape)
    print(Rd.shape)

    # 调用 dqrdc 删除 QR 分解的列
    w = cp.zeros(k).astype(cp.float64)
    Qd, Rd = qrupdate.dqrdec(Qd, Rd, j)

    print(Qd.shape)
    print(Rd.shape)

    # 去掉Q和R的最后一列
    Qd = Qd[:, :-1]
    Rd = Rd[:-1, :]

    print(Qd.shape)
    print(Rd.shape)

    # 验证 QR 分解的正确性
    A_updated = cp.hstack([A[:, :j], A[:, j + 1 :]])
    Q1, R1 = Qd, Rd  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    # 将结果输出到文件中
    cp.savetxt("dqrdec.A_updated.txt", A_updated)
    cp.savetxt("dqrdec.A_reconstructed.txt", A_reconstructed)

    if not cp.allclose(A_updated, A_reconstructed):
        print("dqrdec_simp:QR update failed")

    ##### 完整QR分解测试 #####
    """
    完整QR分解的Q和R矩阵输入时，不需要预留R的空列
    Q需要输入m*m的矩阵，R需要输入m*n的矩阵

    这时候出来的R的矩阵形状和原来一致，但最后一列不是空的（计算时候用到），需要手动去掉
    """
    # Qdp, Rdp = cp.linalg.qr(A, mode="complete")
    # Qdp = Qdp.astype(cp.float64)
    # Rdp = Rdp.astype(cp.float64)

    # print(Qdp.shape)
    # print(Rdp.shape)

    # qrupdate.dqrdec(Qdp, Rdp, j)

    # # 去掉R的最后一列
    # Rdp = Rdp[:, :-1]

    # A_updated = cp.hstack([A[:, :j], A[:, j + 1 :]])
    # Q1, R1 = Qdp, Rdp  # 更新后的 Q 和 R
    # A_reconstructed = Q1 @ R1

    # # 将结果输出到文件中
    # cp.savetxt("dqrdec.Af_updated.txt", A_updated)
    # cp.savetxt("dqrdec.Af_reconstructed.txt", A_reconstructed)

    # if not cp.allclose(A_updated, A_reconstructed):
    #     print("dqrdec_full:QR update failed")


def updating_test_dqrinc():
    m, n, begin = 6000, 5, 5
    # 生成随机矩阵 A
    A_updated = cp.random.rand(m, n).astype(cp.float64)
    # 建一个表格储存误差
    errors = cp.zeros((m - begin, 1))
    ##### 简略QR分解测试 #####
    # 计算QR分解
    Q, R = cp.linalg.qr(A_updated)
    Q = Q.astype(cp.float64)
    R = R.astype(cp.float64)
    for n in range(begin, m):
        k = n
        j = n + 1  # 插入新列的位置

        # 生成随机列向量 x
        x = cp.random.rand(m).astype(cp.float64)

        # print(f"Q.shape{Q.shape}")
        # print(f"R.shape{R.shape}")

        # 调用 dqrinc 更新 QR 分解
        Q1, R1 = qrupdate.dqrinc(Q, R, j, x)
        Q = Q1
        R = R1

        # 验证 QR 分解的正确性
        A_updated = cp.hstack([A_updated[:, :], x.reshape(-1, 1)])
        A_reconstructed = Q1 @ R1
        # print("finish1")

        # 计算误差
        error = cp.linalg.norm(A_updated - A_reconstructed)
        errors[n - 5] = error
        print(f"finish:{n}")

    errors = cp.asnumpy(errors)
    print(errors)
    plt.plot(errors)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.savefig("error_plot.png")


def updating_test_dqrincAndDqrdec():
    m, n, begin = 60, 5, 5
    # 生成随机矩阵 A
    A_updated = cp.random.rand(m, n).astype(cp.float64)
    # 建一个表格储存误差
    count = 0
    length = 2 * (m - begin) + 1
    errors = cp.zeros((length, 1))
    ##### 简略QR分解测试 #####
    # 计算QR分解
    Q, R = cp.linalg.qr(A_updated)
    Q = Q.astype(cp.float64)
    R = R.astype(cp.float64)
    for n in range(begin, m):
        j = n + 1  # 插入新列的位置
        count += 1

        # 生成随机列向量 x
        x = cp.random.rand(m).astype(cp.float64)

        # print(f"Q.shape{Q.shape}")
        # print(f"R.shape{R.shape}")

        # 调用 dqrinc 更新 QR 分解
        Q, R = qrupdate.dqrinc(Q, R, j, x)

        # 验证 QR 分解的正确性
        A_updated = cp.hstack([A_updated[:, :], x.reshape(-1, 1)])
        A_reconstructed = Q @ R
        # print("finish1")

        # 计算误差
        error = cp.linalg.norm(A_updated - A_reconstructed)
        errors[count] = error
        print(f"finish:{count}")
    for n in range(m, begin, -1):
        j = n - 1  # 删除列的位置
        count += 1

        # print(f"Q.shape{Q.shape}")
        # print(f"R.shape{R.shape}")

        # 调用 dqrinc 更新 QR 分解
        Q, R = qrupdate.dqrdec(Q, R, j)

        # 验证 QR 分解的正确性

        A_updated = A_updated[:, :-1]
        A_reconstructed = Q @ R
        # print("finish1")

        # 计算误差
        error = cp.linalg.norm(A_updated - A_reconstructed)
        errors[count] = error
        print(f"finish:{count}")

    errors = cp.asnumpy(errors)
    print(errors)
    plt.plot(errors)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.savefig("error_plot.png")


# 执行测试
# test_dqrinc()
test_dqrdec()
# updating_test_dqrinc()
# updating_test_dqrincAndDqrdec()
