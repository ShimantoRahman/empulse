from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "_empulse",
        ["cpp/_utils.cpp"],
    ),
]

if __name__ == "__main__":
    setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False)
