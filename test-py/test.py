import numpy as np
import sys

sys.path.append(
    "/home/anzrew/Documents/qrupdate/src/build/lib.linux-x86_64-cpython-311"
)
import matplotlib

matplotlib.use("Agg")  # 设置非交互式后端
import matplotlib.pyplot as plt
import qrupdate


def print_para():
    print(qrupdate.__doc__)
    print(qrupdate.dch1up.__doc__)
    print(qrupdate.dqrder.__doc__)


def test_dqrinc():
    # 设置测试参数
    m, n = 6, 4
    k = n
    ldq, ldr = m, n + 1
    j = 2  # 插入新列的位置

    # 生成随机矩阵 A 和列向量 x
    A = np.random.rand(m, n).astype(np.float64, order="F")
    x = np.random.rand(m).astype(np.float64, order="F")

    ##### 简略QR分解测试 #####
    """
    简要QR分解的Q和R矩阵输入时，需要预留R和Q的空行和空列
    Q需要输入m*(n+1)的矩阵，R需要输入(n+1)*(n+1)的矩阵
    """
    Q, R = np.linalg.qr(A)
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")
    R = np.append(R, np.zeros((n, 1)), axis=1)
    R = np.append(R, np.zeros((1, n + 1)), axis=0)
    Q = np.append(Q, np.zeros((m, 1)), axis=1)

    print(Q.shape)
    print(R.shape)

    # 调用 dqrinc 更新 QR 分解
    w = np.zeros(k).astype(np.float64, order="F")
    qrupdate.dqrinc(k, Q, R, j, x)

    # 验证 QR 分解的正确性
    A_updated = np.hstack([A[:, : j - 1], x.reshape(-1, 1), A[:, j - 1 :]])
    Q1, R1 = Q, R  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    assert np.allclose(A_updated, A_reconstructed), "dqrinc:QR update failed"

    ##### 完整QR分解测试 #####
    """
    完整QR分解的Q和R矩阵输入时，需要预留R的空列
    Q需要输入m*m的矩阵，R需要输入m*(n+1)的矩阵
    """
    Qp, Rp = np.linalg.qr(A, mode="complete")
    Qp = Qp.astype(np.float64, order="F")
    Rp = Rp.astype(np.float64, order="F")
    Rp = np.append(Rp, np.zeros((m, 1)), axis=1)

    print(Qp.shape)
    print(Rp.shape)

    qrupdate.dqrinc(k, Qp, Rp, j, x)

    A_updated = np.hstack([A[:, : j - 1], x.reshape(-1, 1), A[:, j - 1 :]])
    Q1, R1 = Qp, Rp  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    assert np.allclose(A_updated, A_reconstructed), "QR update failed"


def test_dqrdec():
    # 设置测试参数
    m, n = 8, 6
    k = n
    ldq, ldr = m, n + 1
    j = 2  # 删除列的位置

    # 生成随机矩阵 A 和列向量 x
    A = np.random.rand(m, n).astype(np.float64, order="F")

    ##### 简略QR分解测试 #####
    """
    简要QR分解的Q和R矩阵输入时，不需要预留R和Q的空行和空列
    Q需要输入m*n的矩阵，R需要输入n*n的矩阵

    这时候出来的Q和R的矩阵形状和原来一致，但最后一列不是空的（计算时候用到），需要手动去掉
    """
    Qd, Rd = np.linalg.qr(A)
    Qd = Qd.astype(np.float64, order="F")
    Rd = Rd.astype(np.float64, order="F")

    print(Qd.shape)
    print(Rd.shape)

    # 调用 dqrdc 删除 QR 分解的列
    w = np.zeros(k).astype(np.float64, order="F")
    qrupdate.dqrdec(Qd, Rd, j)

    # 去掉Q和R的最后一列
    Qd = Qd[:, :-1]
    Rd = Rd[:-1, :-1]

    # 验证 QR 分解的正确性
    A_updated = np.hstack([A[:, :j], A[:, j + 1 :]])
    Q1, R1 = Qd, Rd  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    # 将结果输出到文件中
    np.savetxt("A_updated.txt", A_updated)
    np.savetxt("A_reconstructed.txt", A_reconstructed)

    # assert np.allclose(A_updated, A_reconstructed), "dqrdec:QR update failed"

    ##### 完整QR分解测试 #####
    """
    完整QR分解的Q和R矩阵输入时，不需要预留R的空列
    Q需要输入m*m的矩阵，R需要输入m*n的矩阵

    这时候出来的R的矩阵形状和原来一致，但最后一列不是空的（计算时候用到），需要手动去掉
    """
    Qdp, Rdp = np.linalg.qr(A, mode="complete")
    Qdp = Qdp.astype(np.float64, order="F")
    Rdp = Rdp.astype(np.float64, order="F")

    print(Qdp.shape)
    print(Rdp.shape)

    qrupdate.dqrdec(Qdp, Rdp, j)

    # 去掉R的最后一列
    Rdp = Rdp[:, :-1]

    A_updated = np.hstack([A[:, :j], A[:, j + 1 :]])
    Q1, R1 = Qdp, Rdp  # 更新后的 Q 和 R
    A_reconstructed = Q1 @ R1

    # 将结果输出到文件中
    np.savetxt("Af_updated.txt", A_updated)
    np.savetxt("Af_reconstructed.txt", A_reconstructed)

    # assert np.allclose(A_updated, A_reconstructed), "QR update failed"


def updating_test_dqrinc():
    m, n, begin = 6, 5, 5
    # 生成随机矩阵 A
    A_updated = np.random.rand(m, n).astype(np.float64, order="F")
    # 建一个表格储存误差
    errors = np.zeros((m - begin, 1))
    ##### 简略QR分解测试 #####
    # 计算QR分解
    Q, R = np.linalg.qr(A_updated)
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")
    for n in range(begin, m):
        k = n
        j = n + 1  # 插入新列的位置

        # 生成随机列向量 x
        x = np.random.rand(m).astype(np.float64, order="F")

        R = np.append(R, np.zeros((n, 1)), axis=1)
        R = np.append(R, np.zeros((1, n + 1)), axis=0)
        Q = np.append(Q, np.zeros((m, 1)), axis=1)

        # print(f"Q.shape{Q.shape}")
        # print(f"R.shape{R.shape}")

        # 调用 dqrinc 更新 QR 分解
        qrupdate.dqrinc(k, Q, R, j, x)

        # 验证 QR 分解的正确性
        A_updated = np.hstack([A_updated[:, :], x.reshape(-1, 1)])
        Q1, R1 = Q, R  # 更新后的 Q 和 R
        A_reconstructed = Q1 @ R1
        # print("finish1")

        # 验证Q的正交性
        Q1TQ1 = Q1.T @ Q1

        # 计算误差
        # error = np.linalg.norm(A_updated - A_reconstructed)
        error = np.linalg.norm(Q1TQ1 - np.eye(n + 1))
        errors[n - 5] = error
        # print("finish2")

    # print(errors)
    print(errors[-1])
    plt.plot(errors)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.savefig("error_plot_Q.png")


def test_dqrinr():

    print(qrupdate.dqrinr.__doc__)

    print("=" * 20)
    print("简化QR分解的dqrinr测试")

    print("-" * 20)
    print("参数设置")
    # 设置测试参数
    m, n = 10, 10
    k = n
    ldq, ldr = m + 1, m
    j = 11  # 插入新列的位置

    # 生成随机矩阵 A 和列向量 x
    A = np.random.rand(m, n).astype(np.float64, order="F")
    x = np.random.rand(n).astype(np.float64, order="F")

    print(f"m={m}，n={n}")

    print("-" * 20)
    print("测试规模")

    Q, R = np.linalg.qr(A, mode="complete")
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")

    print(f"A:{A.shape}")
    print(f"对A进行QR分解，Q:{Q.shape}，R:{R.shape}")

    print("-" * 20)
    print("预处理")

    Q = np.append(Q, np.zeros((1, m)), axis=0)
    Q = np.append(Q, np.zeros((m + 1, 1)), axis=1)
    R = np.append(R, np.zeros((1, n)), axis=0)

    print(f"Q:{Q.shape}，R:{R.shape}")

    print("-" * 20)
    print("调用dqrinr函数")

    w = np.zeros(2 * n).astype(np.float64, order="F")
    x_history = np.copy(x)
    print(f"插入{x}")
    qrupdate.dqrinr(m, Q, R, j, x, w)

    print("调用成功")

    print("-" * 20)
    print("更新后规模为")

    print(f"Q:{Q.shape}，R:{R.shape}")

    print("-" * 20)
    print("检测结果")
    A_updated = np.vstack([A[: j - 1, :], x_history.reshape(1, -1), A[j - 1 :, :]])
    A_reconstructed = Q @ R
    print("更新后的QR与A增添一行对比")
    # print("A_updated:")
    # print(A_updated)
    # print("A_reconstructed:")
    # print(A_reconstructed)
    error = np.allclose(A_updated, A_reconstructed)
    print(f"结果是否在误差范围内：{error}")


def test_dqrder():

    print(qrupdate.dqrder.__doc__)

    print("=" * 20)
    print("简化QR分解的dqrder测试")

    print("-" * 20)
    print("参数设置")
    # 设置测试参数
    m, n = 10, 4
    k = n
    ldq, ldr = m + 1, m
    j = 2  # 插入新列的位置

    # 生成随机矩阵 A 和列向量 x
    A = np.random.rand(m, n).astype(np.float64, order="F")
    x = np.random.rand(n).astype(np.float64, order="F")

    print(f"m={m}，n={n}")

    print("-" * 20)
    print("测试规模")

    Q, R = np.linalg.qr(A, mode="complete")
    Q = Q.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")

    print(f"A:{A.shape}")
    print(f"对A进行QR分解，Q:{Q.shape}，R:{R.shape}")

    print("-" * 20)
    print("调用dqrder函数")

    w = np.zeros(2 * m).astype(np.float64, order="F")
    print(f"删除第{j}行")
    qrupdate.dqrder(Q, R, j, w)

    print("调用成功")

    print("-" * 20)
    print("更新后规模为")

    print(f"Q:{Q.shape}，R:{R.shape}")

    print("-" * 20)
    print("后处理")

    Q = Q[:-1, :-1]
    R = R[:-1, :]

    print(f"Q:{Q.shape}，R:{R.shape}")

    print("-" * 20)
    print("检测结果")
    A_updated = np.vstack([A[: j - 1, :], A[j:, :]])
    A_reconstructed = Q @ R
    print("更新后的QR与A减少一行对比")
    # print("A_updated:")
    # print(A_updated)
    # print("A_reconstructed:")
    # print(A_reconstructed)
    error = np.allclose(A_updated, A_reconstructed)
    print(f"结果是否在误差范围内：{error}")


def test_dch1up():

    print(qrupdate.dch1up.__doc__)

    print("=" * 20)
    print("dch1up测试")

    print("-" * 20)
    print("参数设置")
    # 设置测试参数
    m = 5

    # 生成一个上三角矩阵 R
    R = np.random.rand(m, m).astype(np.float64, order="F")
    R = np.triu(R)

    print(f"m={m}")
    print(f"R:{R}")
    print(f"A=R^T@R:{R.T@R}")

    print("-" * 20)
    print("测试规模")

    # A=R^T@R
    A = R.T @ R

    print(f"A:{A.shape}")
    print(f"A的Cholesky分解：R:{R.shape}")

    print("-" * 20)
    print("调用dch1up函数")

    R = R.astype(np.float64, order="F")
    x = np.random.rand(m).astype(np.float64, order="F")
    x_history = np.copy(x)
    print(f"插入{x}")
    qrupdate.dch1up(R, x, np.zeros(m).astype(np.float64, order="F"))

    print("调用成功")

    print("-" * 20)
    print("更新后规模为")

    print(f"R:{R.shape}")

    print("-" * 20)
    print("检测结果")
    A_updated = A + x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
    A_reconstructed = R.T @ R
    print("更新后的QR与A增添一行对比")
    # print("A_updated:")
    print(A_updated)
    # print("A_reconstructed:")
    print(A_reconstructed)
    error = np.allclose(A_updated, A_reconstructed)
    print(f"结果是否在误差范围内：{error}")

def test_dch1dn():

    print(qrupdate.dch1dn.__doc__)

    print("="*20)
    print("dch1dn测试")

    print("-"*20)
    print("参数设置")
    # 设置测试参数
    m = 5

    # 生成一个上三角矩阵 R
    R = np.random.rand(m, m).astype(np.float64, order="F")
    R = np.triu(R)

    print(f"m={m}")
    print(f"R:{R}")
    print(f"A=R^T@R:{R.T@R}")
    print(f"R:{R}")
    print(f"A=R^T@R:{R.T@R}")

    print("-"*20)
    print("测试规模")

    # A=R^T@R
    A = R.T @ R

    print(f"A:{A.shape}")
    print(f"A的Cholesky分解：R:{R.shape}")

    print("-"*20)
    print("调用dch1up函数")

    R = R.astype(np.float64, order="F")
    R = R.astype(np.float64, order="F")
    x = np.random.rand(m).astype(np.float64, order="F")
    x_history = np.copy(x)
    print(f"插入{x}")

    print("x@x^T:")
    print(x_history.reshape(-1, 1) @ x_history.reshape(1, -1))

    w = np.zeros(m).astype(np.float64, order="F")
    qrupdate.dch1up(R, x, w)

    print(f"R:{R}")


    print("调用成功")

    print("-"*20)
    print("更新后规模为")

    print(f"R:{R.shape}")

    print("-"*20)
    print("检测结果")
    A_updated = A - x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
    A_reconstructed = R.T @ R
    print("更新后")
    # print("A_updated:")
    print(A_updated)
    print(A_updated)
    # print("A_reconstructed:")
    print(A_reconstructed)
    print(A_reconstructed)
    error = np.allclose(A_updated, A_reconstructed)
    print(f"结果是否在误差范围内：{error}")

def test_dch1dn():

    print(qrupdate.dch1dn.__doc__)

    print("="*20)
    print("dch1dn测试")

    print("-"*20)
    print("参数设置")
    # 设置测试参数
    m = 5

    # 生成一个上三角矩阵 R
    R = np.random.rand(m, m).astype(np.float64, order="F")
    R = np.triu(R)

    print(f"m={m}")
    print(f"R:{R}")
    print(f"A=R^T@R:{R.T@R}")

    print("-"*20)
    print("测试规模")

    # A=R^T@R
    A = R.T @ R

    print(f"A:{A.shape}")
    print(f"A的Cholesky分解：R:{R.shape}")

    print("-"*20)
    print("调用dch1up函数")

    R = R.astype(np.float64, order="F")
    x = np.random.rand(m).astype(np.float64, order="F")
    x_history = np.copy(x)
    print(f"插入{x}")

    print("x@x^T:")
    print(x_history.reshape(-1, 1) @ x_history.reshape(1, -1))

    w = np.zeros(m).astype(np.float64, order="F")
    qrupdate.dch1dn(R, x, w, 0)

    print(f"R:{R}")


    print("调用成功")

    print("-"*20)
    print("更新后规模为")

    print(f"R:{R.shape}")

    print("-"*20)
    print("检测结果")
    A_updated = A - x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
    A_reconstructed = R.T @ R
    print("更新后")
    # print("A_updated:")
    print(A_updated)
    # print("A_reconstructed:")
    print(A_reconstructed)
    error = np.allclose(A_updated, A_reconstructed)
    print(f"结果是否在误差范围内：{error}")

if __name__ == "__main__":
    # 执行测试
    # test_dqrinc()
    # test_dqrdec()
    # updating_test_dqrinc()
    # print_para()
    # test_dqrinr()
    # test_dqrder()
    test_dch1dn()
    test_dch1dn()
