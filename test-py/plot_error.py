from pyexpat.model import XML_CTYPE_CHOICE
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 设置非交互式后端
import matplotlib.pyplot as plt
from dch1up import dch1up
from dch1dn import dch1dn
from dch2up import dch2up
from dch2dn import dch2dn

def polt_error_ch(n, times):
    """
    对一个矩阵不断插入最后一行，然后删除最后一行，画出QR分解的更新误差，绘制：
    1. A的1-范数误差
    3. R的1-范数误差

    Parameters
    ----------
    m : int
        矩阵A的行数
    n : int
        矩阵A的列数
    times : int
        操作次数
    """
    # 生成随机矩阵 R
    R_updated = np.random.rand(n, n).astype(np.float64)
    R_updated = np.triu(R_updated)
    # 建一个表格储存误差
    count = 0
    length = 2 * times + 1
    errors1 = np.zeros((length, 1))
    errors2 = np.zeros((length, 1))
    ##### 简略QR分解测试 #####
    # 计算 A = R.T @ R
    A_updated = np.dot(R_updated.T, R_updated)
    count = 0
    x_mat = np.zeros((times, n))
    # 插入新列的位置
    for i in range(times):
        count += 1
        # 生成随机行向量 x
        x = np.random.rand(n).astype(np.float64)
        x_history = np.copy(x)
        x_mat[count - 1, :] = x_history
        # 调用 dqrinc
        R_updated = dch2up(R_updated, x)
        # 验证更新后的 R
        A_updated = A_updated + x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
        A_reconstructed = R_updated.T @ R_updated
        L_reconstructed = np.linalg.cholesky(A_updated)
        R_reconstructed = L_reconstructed.T
        # print(R_reconstructed)
        # print(R_updated)
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error2 = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated, 1)
        errors1[count] = error1
        errors2[count] = error2
        print(f"finish:{count}")
    for i in range(times - 10):
        count += 1
        # 生成随机行向量 x
        x = x_mat[times- (count - times), :]
        x_history = np.copy(x)
        # 调用 dqrinc
        R_updated = dch2dn(R_updated, x)
        # 验证更新后的 R
        A_updated = A_updated - x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
        A_reconstructed = R_updated.T @ R_updated
        L_reconstructed = np.linalg.cholesky(A_updated)
        R_reconstructed = L_reconstructed.T
        # print(R_reconstructed)
        # print(R_updated)
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error2 = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated, 1)
        errors1[count] = error1
        errors2[count] = error2
        print(f"finish:{count}")
    # print(errors)
    # 截取后200个数据
    # errors1 = errors1[50:351]
    # errors2 = errors2[50:351]
    plt.plot(errors1)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of A_updated and A_reconstructed")
    plt.savefig(f"1norm_error_plot_A_ch.png")
    # 清空图像
    plt.cla()
    plt.plot(errors2)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of R and R_updated")
    plt.savefig(f"1norm_error_plot_R_updated_ch.png")


if __name__ == "__main__":
    # polt_error(100, 100, 100)
    polt_error_ch(100, 200)
