from glob import glob
from setuptools import setup

from pybind11.setup_helpers import build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Specify the path to the Eigen headers
EIGEN_INCLUDE_PATH = '/opt/homebrew/include/eigen3/'

# Use Pybind11Extension instead of intree_extensions for better control
ext_modules = [
    Pybind11Extension(
        name="py_eigen",
        sources=glob('py_eigen.cpp'),
        include_dirs=[EIGEN_INCLUDE_PATH]  # Add include directories here
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)

