import numpy as np
import qrupdate

def test_dqr1up():
    # 设置测试参数
    m = 5    # Q的行数
    n = 4    # R的列数
    k = min(m, n)  # Q的列数和R的行数
    ldq = m  # Q的主维度
    ldr = k  # R的主维度

    # 创建随机输入矩阵和向量
    Q = np.random.rand(m, k)
    R = np.random.rand(k, n)
    u = np.random.rand(m)
    v = np.random.rand(n)
    w = np.zeros(2 * k)  # 工作空间向量

    # 确保Q是正交的
    Q, _ = np.linalg.qr(Q)

    # 确保维度正确
    assert Q.shape == (m, k), "Q的维度不正确"
    assert R.shape == (k, n), "R的维度不正确"

    # 调用子程序
    qrupdate.dqr1up(m, n, k, Q, ldq, R, ldr, u, v, w)

    # 检查结果（这里需要根据实际情况添加检查条件）
    # 例如，检查R是否为上梯形矩阵，Q是否仍然正交等

    # 打印结果，或进行断言测试
    print("Updated Q:", Q)
    print("Updated R:", R)

# 运行测试
if __name__ == "__main__":
    test_dqr1up()
