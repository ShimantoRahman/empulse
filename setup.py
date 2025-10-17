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
        # ProfTree modules
        Extension(
            'empulse.models.proftree.random',
            ['empulse/models/proftree/random.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.proftree.node',
            ['empulse/models/proftree/node.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.proftree.tree',
            ['empulse/models/proftree/tree.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.proftree.forest',
            ['empulse/models/proftree/forest.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.proftree.operators',
            ['empulse/models/proftree/operators.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.proftree.evolution',
            ['empulse/models/proftree/evolution.pyx'],
            include_dirs=[np.get_include()],
        ),
        Extension(
            'empulse.models.proftree.evolutionary_tree',
            ['empulse/models/proftree/evolutionary_tree.pyx'],
            include_dirs=[np.get_include()],
        ),
    ]
    setup(
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                'language_level': 3,
                'boundscheck': False,
                'wraparound': False,
                'initializedcheck': False,
                'nonecheck': False,
                'cdivision': True,
            },
        )
    )
