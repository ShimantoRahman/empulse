import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

if __name__ == '__main__':
    extensions = [
        Extension('empulse.metrics._loss.loss', ['empulse/metrics/_loss/*.pyx'], include_dirs=[np.get_include()]),
    ]
    setup(ext_modules=cythonize(extensions))
