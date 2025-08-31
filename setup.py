import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

if __name__ == '__main__':
    extensions = [
        Extension('empulse.metrics._loss.loss', ['empulse/metrics/_loss/*.pyx'], include_dirs=[np.get_include()]),
        Extension(
            'empulse.models.cost_sensitive._impurity.cost_impurity',
            ['empulse/models/cost_sensitive/_impurity/*.pyx'],
            include_dirs=[np.get_include(), 'sklearn.utils._typedefs', 'sklearn.tree._criterion'],
        ),
    ]
    setup(ext_modules=cythonize(extensions))
