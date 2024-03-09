from pyexpat.model import XML_CTYPE_CHOICE
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 设置非交互式后端
import matplotlib.pyplot as plt
import updating

def polt_error_r(m, n, times):
    """
    对一个矩阵不断插入最后一列，然后删除最后一列，画出QR分解的更新误差，绘制：
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
        x = np.random.rand(m).astype(np.float64)
        # 调用 dqrinc 更新 QR 分解
        Q, R = updating.appending_column(Q, R, j, x)
        # 验证 QR 分解的正确性
        A_updated = np.hstack([A_updated[:, :j-1], x.reshape(-1, 1), A_updated[:, j-1:]])
        A_reconstructed = Q @ R
        ATA = np.dot(A_updated.T, A_updated)
        RTR = np.dot(R.T, R)
        _,R_updated = np.linalg.qr(A_updated, mode="complete") 
        RuTRu = np.dot(R_updated.T, R_updated)
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error2 = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 1)
        error3 = np.linalg.norm(ATA - RTR, 1)
        error4 = np.linalg.norm(R, 1) - np.linalg.norm(R_updated, 1)
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
        Q, R = updating.deleting_column(Q, R, j)
        # 验证 QR 分解的正确性
        A_updated = np.hstack([A_updated[:, :j-1], A_updated[:, j:]])
        A_reconstructed = Q @ R
        ATA = np.dot(A_updated.T, A_updated)
        RTR = np.dot(R.T, R)
        _,R_updated = np.linalg.qr(A_updated, mode="complete") 
        RuTRu = np.dot(R_updated.T, R_updated)
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error2 = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 1)
        error3 = np.linalg.norm(ATA - RTR, 1)
        error4 = np.linalg.norm(R, 1) - np.linalg.norm(R_updated, 1)
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
    plt.title("1-norm error of A_updated and A_reconstructed:column")
    plt.savefig(f"1norm_error_plot_A_column.png")
    # 清空图像
    plt.cla()
    plt.plot(errors2)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of Q.T@Q and I:column")
    plt.savefig(f"1norm_error_plot_Q_column.png")
    # 清空图像
    plt.cla()
    plt.plot(errors3)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of A.T@A and R.T@R:column")
    plt.savefig(f"1norm_error_plot_R_column.png")
    # 清空图像
    plt.cla()
    plt.plot(errors4)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of R and R_updated:column")
    plt.savefig(f"1norm_error_plot_R_updated_column.png")
    # 清空图像
    plt.cla()
    plt.plot(errors5)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("1-norm error of RTR and R_updated.T@R_updated:column")
    plt.savefig(f"1norm_error_plot_RTR_column.png")

def polt_error_d(m, n, times):
    """
    对一个矩阵不断插入最后一列，然后删除最后一列，画出QR分解的更新误差，绘制：
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
        x = np.random.rand(m).astype(np.float64)
        x_history = np.copy(x)
        # 调用 dqrinc 更新 QR 分解
        Q, R = updating.appending_column(Q, R, j, x)
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
        error4 = np.linalg.norm(R, 1) - np.linalg.norm(R_updated, 1)
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
        Q, R = updating.deleting_column(Q, R, j)
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
        error4 = np.linalg.norm(R, 1) - np.linalg.norm(R_updated, 1)
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
    # 插入新列的位置 
    for i in range(2 * times):
        count += 1
        # 生成随机行向量 x
        x = np.random.rand(n).astype(np.float64)
        x_history = np.copy(x)
        # 调用 dqrinc 
        R_updated = updating.cholesky_update(R_updated, n, x)
        print(R_updated.shape)
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
    # for i in range(times):
    #     count += 1
    #     # 调用 dqrinc 更新 QR 分解
    #     R_updated = updating.cholesky_downdate(R_updated, n, x)
    #     # 验证更新后的 R
    #     A_updated = A_updated - x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
    #     A_reconstructed = R_updated.T @ R_updated
    #     # 计算误差
    #     error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
    #     # error2 = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated, 1)
    #     errors1[count] = error1
    #     print(f"finish:{count}")
    # print(errors)
    # 截取后200个数据
    errors1 = errors1[100:301]
    errors2 = errors2[100:301]
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
    polt_error_r(100, 60, 20)
    polt_error_d(100, 100, 100)
    polt_error_ch(100, 200)