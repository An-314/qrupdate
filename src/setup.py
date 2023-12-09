from numpy.distutils.core import setup, Extension

# 列出所有的 Fortran 源文件
src_files = [
    "dgqvec.f",
    "dqhqr.f",
    "dqrdec.f",
    "dqrinc.f",
    "dqrot.f",
    "dqrqh.f",
    "dqrtv1.f",
]

# 创建扩展模块，包括链接到 BLAS 和 LAPACK
qrupdate_extension = Extension(
    name="qrupdate",
    sources=src_files,
    libraries=['blas','lapack'],
)


# 设置包的配置
setup(
    name="qrupdate",
    version="1.0",
    author="Anzreww",
    description="Python 接口的 qrupdate Fortran 库",
    ext_modules=[qrupdate_extension],
)

# python3 setup.py build --cpu-dispatch="max -avx512_spr" bdist > build_log.log 2>&1