from numpy.distutils.core import setup, Extension

# import os

# blas_src_dir = "../OpenBLAS-0.3.24/OpenBLAS-0.3.24/lapack-netlib/BLAS/SRC"
# blas_files = [
#     os.path.join(blas_src_dir, f) for f in os.listdir(blas_src_dir) if f.endswith(".f")
# ]


# 列出所有的 Fortran 源文件
src_files = [
    "caxcpy.f",
    "cch1dn.f",
    "cch1up.f",
    "cchdex.f",
    "cchinx.f",
    "cchshx.f",
    "cgqvec.f",
    "clu1up.f",
    "clup1up.f",
    "cqhqr.f",
    "cqr1up.f",
    "cqrdec.f",
    "cqrder.f",
    "cqrinc.f",
    "cqrinr.f",
    "cqrot.f",
    "cqrqh.f",
    "cqrshc.f",
    "cqrtv1.f",
    "dch1dn.f",
    "dch1up.f",
    "dchdex.f",
    "dchinx.f",
    "dchshx.f",
    "dgqvec.f",
    "dlu1up.f",
    "dlup1up.f",
    "dqhqr.f",
    "dqr1up.f",
    "dqrdec.f",
    "dqrder.f",
    "dqrinc.f",
    "dqrinr.f",
    "dqrot.f",
    "dqrqh.f",
    "dqrshc.f",
    "dqrtv1.f",
    "sch1dn.f",
    "sch1up.f",
    "schdex.f",
    "schinx.f",
    "schshx.f",
    "sgqvec.f",
    "slu1up.f",
    "slup1up.f",
    "sqhqr.f",
    "sqr1up.f",
    "sqrdec.f",
    "sqrder.f",
    "sqrinc.f",
    "sqrinr.f",
    "sqrot.f",
    "sqrqh.f",
    "sqrshc.f",
    "sqrtv1.f",
    "zaxcpy.f",
    "zch1dn.f",
    "zch1up.f",
    "zchdex.f",
    "zchinx.f",
    "zchshx.f",
    "zgqvec.f",
    "zlu1up.f",
    "zlup1up.f",
    "zqhqr.f",
    "zqr1up.f",
    "zqrdec.f",
    "zqrder.f",
    "zqrinc.f",
    "zqrinr.f",
    "zqrot.f",
    "zqrqh.f",
    "zqrshc.f",
    "zqrtv1.f",
]

# src_files = src_files + blas_files

# OpenBLAS 库的路径
openblas_lib_dir = "C:/Program Files/OpenBLAS-0.3.25-x64-64/lib"

# 创建扩展模块，包括链接到 BLAS 和 LAPACK
qrupdate_extension = Extension(
    name="qrupdate",
    sources=src_files,
    library_dirs=[openblas_lib_dir],  # 添加库路径
    libraries=["openblas"],  # 假设库文件名为 'libopenblas.a' 或 'libopenblas.dll.a'
    extra_f77_compile_args=[
        "-lopenblas",
        '-L"C:/Program Files/OpenBLAS-0.3.25-x64-64/lib"',
    ],
    extra_f90_compile_args=[
        "-lopenblas",
        '-L"C:/Program Files/OpenBLAS-0.3.25-x64-64/lib"',
    ],
    # extra_f77_compile_args=[
    #     '/LIBPATH:"C:\Program Files\OpenBLAS-0.3.25-x64-64\lib"',
    #     "openblas.lib",
    # ],
)


# 设置包的配置
setup(
    name="qrupdate",
    version="1.0",
    author="Anzreww",
    description="Python 接口的 qrupdate Fortran 库",
    ext_modules=[qrupdate_extension],
)
