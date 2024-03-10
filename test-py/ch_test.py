from os import error
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 设置非交互式后端
import matplotlib.pyplot as plt
import updating
import qrupdating
import qrupdating_fix

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
    errors1_qr = np.zeros((length, 1))
    errors2_qr = np.zeros((length, 1))
    errors1_qr_fix = np.zeros((length, 1))
    errors2_qr_fix = np.zeros((length, 1))
    # 计算 A = R.T @ R
    A_updated = np.dot(R_updated.T, R_updated)
    count = 0
    # 用一个矩阵存x
    x_mat = np.zeros((times, n))
    # 插入新列的位置 
    for i in range(times):
        count += 1
        # 生成随机行向量 x
        x = np.random.rand(n).astype(np.float64)
        x_history = np.copy(x)
        x_mat[count - 1, :] = x_history
        # 调用 dch1up
        R_update_before = np.copy(R_updated)
        R_updated = updating.cholesky_update(R_update_before, n, x)
        R_updated_qr = qrupdating.dch1up(R_update_before, x)
        R_updated_qr_fix = qrupdating_fix.dch1up(R_update_before, x)
        # 验证更新后的 R
        A_updated = A_updated + x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
        A_reconstructed = R_updated.T @ R_updated
        A_reconstructed_qr = R_updated_qr.T @ R_updated_qr
        A_reconstructed_qr_fix = R_updated_qr_fix.T @ R_updated_qr_fix
        L_reconstructed = np.linalg.cholesky(A_updated)
        R_reconstructed = L_reconstructed.T
        # print(R_reconstructed)
        # print(R_updated)
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error1_qr = np.linalg.norm(A_updated - A_reconstructed_qr, 1)
        error1_qr_fix = np.linalg.norm(A_updated - A_reconstructed_qr_fix, 1)
        error2 = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated, 1)
        error2_qr = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated_qr, 1)
        error2_qr_fix = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated_qr_fix, 1)
        errors1[count] = error1
        errors1_qr[count] = error1_qr
        errors1_qr_fix[count] = error1_qr_fix
        errors2[count] = error2
        errors2_qr[count] = error2_qr
        errors2_qr_fix[count] = error2_qr_fix
        print(f"finish:{count}")
    for i in range(times - 10):
        count += 1
        # 验证更新后的 R
        x = x_mat[times- (count - times), :]
        x_history = np.copy(x)
        # 调用 dch1dn
        R_update_before = np.copy(R_updated)
        R_updated = updating.cholesky_downdate(R_update_before, n, x)
        R_updated_qr = qrupdating.dch1dn(R_update_before, x)
        R_updated_qr_fix = qrupdating_fix.dch1dn(R_update_before, x)
        # 验证更新后的 R
        A_updated = A_updated - x_history.reshape(-1, 1) @ x_history.reshape(1, -1)
        A_reconstructed = R_updated.T @ R_updated
        A_reconstructed_qr = R_updated_qr.T @ R_updated_qr
        A_reconstructed_qr_fix = R_updated_qr_fix.T @ R_updated_qr_fix
        L_reconstructed = np.linalg.cholesky(A_updated)
        R_reconstructed = L_reconstructed.T
        # 计算误差
        error1 = np.linalg.norm(A_updated - A_reconstructed, 1)
        error2 = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated, 1)
        error1_qr = np.linalg.norm(A_updated - A_reconstructed_qr, 1)
        error2_qr = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated_qr, 1)
        error1_qr_fix = np.linalg.norm(A_updated - A_reconstructed_qr_fix, 1)
        error2_qr_fix = np.linalg.norm(R_reconstructed, 1) - np.linalg.norm(R_updated_qr_fix, 1)
        errors1[count] = error1
        errors2[count] = error2
        errors1_qr[count] = error1_qr
        errors2_qr[count] = error2_qr
        errors1_qr_fix[count] = error1_qr_fix
        errors2_qr_fix[count] = error2_qr_fix
        print(f"finish:{count}")
    # print(errors)
    # 截取后200个数据
    errors1 = errors1[10:2 * times - 10]
    errors2 = errors2[10:2 * times - 10]
    errors1_qr = errors1_qr[10:2 * times - 10]
    errors2_qr = errors2_qr[10:2 * times - 10]
    errors1_qr_fix = errors1_qr_fix[10:2 * times - 10]
    errors2_qr_fix = errors2_qr_fix[10:2 * times - 10]
    plt.plot(errors1)
    plt.plot(errors1_qr)
    plt.plot(errors1_qr_fix)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["f2py","python", "python_fix"])
    plt.title("1-norm error of A_updated and A_reconstructed")
    plt.savefig(f"1norm_error_plot_A_ch.png")
    # 清空图像
    plt.cla()
    plt.plot(errors2)
    plt.plot(errors2_qr)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["f2py","python", "python_fix"])
    plt.title("1-norm error of R and R_updated")
    plt.savefig(f"1norm_error_plot_R_updated_ch.png")
    # 清空图像
    plt.cla()
    plt.plot(errors1)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["f2py"])
    plt.title("1-norm error of A_updated and A_reconstructed")
    plt.savefig(f"1norm_error_plot_A_ch_f2py.png")
    # 清空图像
    plt.cla()
    plt.plot(errors1_qr)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["python"])
    plt.title("1-norm error of A_updated and A_reconstructed")
    plt.savefig(f"1norm_error_plot_A_ch_python.png")
    # 清空图像
    plt.cla()
    plt.plot(errors1_qr_fix)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["python_fix"])
    plt.title("1-norm error of A_updated and A_reconstructed")
    plt.savefig(f"1norm_error_plot_A_ch_python_fix.png")
    # 清空图像
    plt.cla()
    plt.plot(errors2)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["f2py"])
    plt.title("1-norm error of R and R_updated")
    plt.savefig(f"1norm_error_plot_R_updated_ch_f2py.png")
    # 清空图像
    plt.cla()
    plt.plot(errors2_qr)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["python"])
    plt.title("1-norm error of R and R_updated")
    plt.savefig(f"1norm_error_plot_R_updated_ch_python.png")
    # 清空图像
    plt.cla()
    plt.plot(errors2_qr_fix)
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend(["python_fix"])
    plt.title("1-norm error of R and R_updated")
    plt.savefig(f"1norm_error_plot_R_updated_ch_python_fix.png")


polt_error_ch(100, 200)