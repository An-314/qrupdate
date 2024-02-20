from pyexpat.model import XML_CTYPE_CHOICE
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import updating

def polt_error(m, n, times):
    """
    对一个矩阵不断插入最后一行，然后删除最后一行，画出QR分解的更新误差，绘制：
    1. A的1-范数误差
    2. Q.T@Q与1的1-范数误差
    3. A.T@A与R.T@R的1-范数误差
    4. R的1-范数误差
    5. R.T@R与R_updated.T@R_updated的1-范数误差

    Parameters
    ----------
    m : int
        矩阵A的行数
    n : int
        矩阵A的列数
    times : int
        操作次数
    """
    # 生成随机矩阵 A
    A_updated = np.random.rand(m, n).astype(np.float64)
    # 建一个表格储存误差
    count = 0
    length = 2 * times + 1
    errors1 = np.zeros((length, 1))
    errors2 = np.zeros((length, 1))
    errors3 = np.zeros((length, 1))
    errors4 = np.zeros((length, 1))
    errors5 = np.zeros((length, 1))
    ##### 简略QR分解测试 #####
    # 计算QR分解
    Q, R = np.linalg.qr(A_updated, mode="complete")
    Q = Q.astype(np.float64)
    R = R.astype(np.float64)
    count = 0
    j = n # 插入新列的位置 
    for i in range(times):
        j += 1
        count += 1
        # 生成随机行向量 x
        x = np.random.rand(n).astype(np.float64)
        x_history = np.copy(x)
        # 调用 dqrinc 更新 QR 分解
        Q, R = updating.appending_row(Q, R, j, x)
        # 验证 QR 分解的正确性
        A_updated = np.vstack([A_updated[:j-1,:], x_history.reshape(1, -1), A_updated[j-1:,:]])
        A_reconstructed = Q @ R
        ATA = np.dot(A_updated.T, A_updated)
        RTR = np.dot(R.T, R)
        _,R_updated = np.linalg.qr(A_updated, mode="complete") 
        RuTRu = np.dot(R_updated.T, R_updated)
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error2 = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 1)
        error3 = np.linalg.norm(ATA - RTR, 1)
        error4 = np.linalg.norm(R - R_updated, 1)
        error5 = np.linalg.norm(RTR - RuTRu, 1)
        errors1[count] = error1
        errors2[count] = error2
        errors3[count] = error3
        errors4[count] = error4
        errors5[count] = error5
        print(f"finish:{count}")
    for i in range(times):
        j -= 1
        count += 1
        # 调用 dqrinc 更新 QR 分解
        Q, R = updating.deleting_row(Q, R, j)
        # 验证 QR 分解的正确性
        A_updated = np.vstack([A_updated[:j-1,:], A_updated[j:,:]])
        A_reconstructed = Q @ R
        ATA = np.dot(A_updated.T, A_updated)
        RTR = np.dot(R.T, R)
        _,R_updated = np.linalg.qr(A_updated, mode="complete") 
        RuTRu = np.dot(R_updated.T, R_updated)
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error2 = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 1)
        error3 = np.linalg.norm(ATA - RTR, 1)
        error4 = np.linalg.norm(R - R_updated, 1)
        error5 = np.linalg.norm(RTR - RuTRu, 1)
        errors1[count] = error1
        errors2[count] = error2
        errors3[count] = error3
        errors4[count] = error4
        errors5[count] = error5
        print(f"finish:{count}")
    # print(errors)
    plt.plot(errors1)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of A_updated and A_reconstructed")
    plt.savefig(f"1norm_error_plot_A_.png")
    # 清空图像
    plt.cla()
    plt.plot(errors2)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of Q.T@Q and I")
    plt.savefig(f"1norm_error_plot_Q_.png")
    # 清空图像
    plt.cla()
    plt.plot(errors3)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of A.T@A and R.T@R")
    plt.savefig(f"1norm_error_plot_R_.png")
    # 清空图像
    plt.cla()
    plt.plot(errors4)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of R and R_updated")
    plt.savefig(f"1norm_error_plot_R_updated.png")
    # 清空图像
    plt.cla()
    plt.plot(errors5)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of RTR and R_updated.T@R_updated")
    plt.savefig(f"1norm_error_plot_RTR_.png")

if __name__ == "__main__":
    polt_error(100, 60, 100)