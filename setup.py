import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

DEBUG = False

if __name__ == '__main__':
    extensions = [
        Extension(
            'empulse.metrics._loss.loss',
            ['empulse/metrics/_loss/*.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.metrics._cy_convex_hull.convex_hull',
            ['empulse/metrics/_cy_convex_hull/*.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._impurity.cost_impurity',
            ['empulse/models/tree/_impurity/*.pyx'],
            include_dirs=[np.get_include(), 'sklearn.utils._typedefs', 'sklearn.tree._criterion'],
        ),
        # ProfTree modules
        Extension(
            'empulse.models.tree._cy_proftree.random',
            ['empulse/models/tree/_cy_proftree/random.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._cy_proftree.node',
            ['empulse/models/tree/_cy_proftree/node.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._cy_proftree.tree',
            ['empulse/models/tree/_cy_proftree/tree.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._cy_proftree.forest',
            ['empulse/models/tree/_cy_proftree/forest.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._cy_proftree.operators',
            ['empulse/models/tree/_cy_proftree/operators.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._cy_proftree.evolution',
            ['empulse/models/tree/_cy_proftree/evolution.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._cy_proftree.max_profit',
            ['empulse/models/tree/_cy_proftree/max_profit.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.tree._cy_proftree.evolutionary_tree',
            ['empulse/models/tree/_cy_proftree/evolutionary_tree.pyx'],
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
            'freethreading_compatible': True,
        }
    else:
        compiler_directives = {
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
            'freethreading_compatible': True,
        }
    setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives))
