# cd到test-py目录下执行

import numpy as np
import sys

sys.path.append("../python_convert/")

import qrupdate


def test_dqrinc():
    # 设置测试参数
    m, n = 6, 4
    k = n
    ldq, ldr = m, n + 1
    j = 2  # 插入新列的位置

    # 生成随机矩阵 A 和列向量 x
    A = np.random.rand(m, n).astype(np.float64)
    x = np.random.rand(m).astype(np.float64)

    ##### 简略QR分解测试 #####
    """
    简要QR分解的Q和R矩阵输入时，python版本不需要预留R和Q的空行和空列
    """
    Q, R = np.linalg.qr(A)
    Q = Q.astype(np.float64)
    R = R.astype(np.float64)
    # R = np.append(R, np.zeros((n, 1)), axis=1)
    # R = np.append(R, np.zeros((1, n + 1)), axis=0)
    # Q = np.append(Q, np.zeros((m, 1)), axis=1)

    print(Q.shape)
    print(R.shape)
    # print("修改之前的Q和R")
    # print(Q)
    # print(R)

    # 调用 dqrinc 更新 QR 分解
    w = np.zeros(k).astype(np.float64)
    Q1, R1 = qrupdate.dqrinc(m, n, Q, R, j, x)

    # 验证 QR 分解的正确性
    A_updated = np.hstack([A[:, : j - 1], x.reshape(-1, 1), A[:, j - 1 :]])
    # Q1, R1 = Q, R  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    # print("修改之后的Q和R")
    # print(Q1)
    # print(R1)

    np.savetxt("dqrinc.A_updated.txt", A_updated)
    np.savetxt("dqrinc.A_reconstructed.txt", A_reconstructed)

    if not np.allclose(A_updated, A_reconstructed):
        print("dqrinc_simp:QR update failed")

    ##### 完整QR分解测试 #####
    """
    完整QR分解的Q和R矩阵输入时，不需要预留R的空列
    """
    Qp, Rp = np.linalg.qr(A, mode="complete")
    Qp = Qp.astype(np.float64)
    Rp = Rp.astype(np.float64)
    # Rp = np.append(Rp, np.zeros((m211, 1)), axis=1)

    print(Qp.shape)
    print(Rp.shape)

    Q1, R1 = qrupdate.dqrinc(m, n, Qp, Rp, j, x)
    A_updated = np.hstack([A[:, : j - 1], x.reshape(-1, 1), A[:, j - 1 :]])
    # Q1, R1 = Qp, Rp  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    np.savetxt("dqrinc.Af_updated.txt", A_updated)
    np.savetxt("dqrinc.Af_reconstructed.txt", A_reconstructed)

    if not np.allclose(A_updated, A_reconstructed):
        print("dqrinc_full:QR update failed")


def test_dqrdec():
    # 设置测试参数
    m, n = 60, 40
    k = n
    ldq, ldr = m, n + 1
    j = 28  # 删除列的位置

    # 生成随机矩阵 A 和列向量 x
    A = np.random.rand(m, n).astype(np.float64)

    ##### 简略QR分解测试 #####
    """
    简要QR分解的Q和R矩阵输入时，不需要预留R和Q的空行和空列
    Q需要输入m*n的矩阵，R需要输入n*n的矩阵

    这时候出来的Q和R的矩阵形状和原来一致，但最后一列不是空的（计算时候用到），需要手动去掉
    """
    Qd, Rd = np.linalg.qr(A)
    Qd = Qd.astype(np.float64)
    Rd = Rd.astype(np.float64)

    print(Qd.shape)
    print(Rd.shape)

    # 调用 dqrdc 删除 QR 分解的列
    w = np.zeros(k).astype(np.float64)
    qrupdate.dqrdec(m, n, Qd, Rd, j)

    # 去掉Q和R的最后一列
    Qd = Qd[:, :-1]
    Rd = Rd[:-1, :-1]

    # 验证 QR 分解的正确性
    A_updated = np.hstack([A[:, :j], A[:, j + 1 :]])
    Q1, R1 = Qd, Rd  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    # 将结果输出到文件中
    np.savetxt("dqrdec.A_updated.txt", A_updated)
    np.savetxt("dqrdec.A_reconstructed.txt", A_reconstructed)

    if not np.allclose(A_updated, A_reconstructed):
        print("dqrdec_simp:QR update failed")

    ##### 完整QR分解测试 #####
    """
    完整QR分解的Q和R矩阵输入时，不需要预留R的空列
    Q需要输入m*m的矩阵，R需要输入m*n的矩阵

    这时候出来的R的矩阵形状和原来一致，但最后一列不是空的（计算时候用到），需要手动去掉
    """
    Qdp, Rdp = np.linalg.qr(A, mode="complete")
    Qdp = Qdp.astype(np.float64)
    Rdp = Rdp.astype(np.float64)

    print(Qdp.shape)
    print(Rdp.shape)

    qrupdate.dqrdec(m, n, Qdp, Rdp, j)

    # 去掉R的最后一列
    Rdp = Rdp[:, :-1]

    A_updated = np.hstack([A[:, :j], A[:, j + 1 :]])
    Q1, R1 = Qdp, Rdp  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    # 将结果输出到文件中
    np.savetxt("dqrdec.Af_updated.txt", A_updated)
    np.savetxt("dqrdec.Af_reconstructed.txt", A_reconstructed)

    if not np.allclose(A_updated, A_reconstructed):
        print("dqrdec_full:QR update failed")


# 执行测试
test_dqrinc()
test_dqrdec()
