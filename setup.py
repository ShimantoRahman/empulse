import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

DEBUG = False

if __name__ == '__main__':
    extensions = [
        Extension('empulse.metrics._loss.loss', ['empulse/metrics/_loss/*.pyx'], include_dirs=[np.get_include()]),
        Extension(
            'empulse.metrics._cy_convex_hull.convex_hull',
            ['empulse/metrics/_cy_convex_hull/*.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cost_sensitive._impurity.cost_impurity',
            ['empulse/models/cost_sensitive/_impurity/*.pyx'],
            include_dirs=[np.get_include(), 'sklearn.utils._typedefs', 'sklearn.tree._criterion'],
        ),
        # ProfTree modules
        Extension(
            'empulse.models.cy_proftree.random',
            ['empulse/models/cy_proftree/random.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cy_proftree.node',
            ['empulse/models/cy_proftree/node.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cy_proftree.tree',
            ['empulse/models/cy_proftree/tree.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cy_proftree.forest',
            ['empulse/models/cy_proftree/forest.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cy_proftree.operators',
            ['empulse/models/cy_proftree/operators.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cy_proftree.evolution',
            ['empulse/models/cy_proftree/evolution.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cy_proftree.max_profit',
            ['empulse/models/cy_proftree/max_profit.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.cy_proftree.evolutionary_tree',
            ['empulse/models/cy_proftree/evolutionary_tree.pyx'],
            include_dirs=[np.get_include()],
        ),
    ]
    if DEBUG:
        compiler_directives = {
            'language_level': 3,
            'boundscheck': True,
            'wraparound': False,
            'initializedcheck': True,
            'nonecheck': True,
            'cdivision': True,
        }
    else:
        compiler_directives = {
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
        }
    setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives))
