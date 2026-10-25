from glob import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 只编译 C++ 文件，不编译 CUDA 文件
_ext_sources = glob("*.cpp")  # 去掉 *.cu

requirements = ["torch>=1.4"]

setup(
    name="balanced_bn",
    version="0.0.1",
    author="Yongyi Su",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CppExtension(
            name="balanced_bn",
            sources=_ext_sources,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)