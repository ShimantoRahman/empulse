import os
import sys
import tempfile
import textwrap
import warnings

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.errors import CompileError, LinkError
from setuptools._distutils.sysconfig import customize_compiler

DEBUG = False


def openmp_flags() -> tuple[list[str], list[str]]:
    """
    Return the compile and link flags that enable OpenMP, or empty lists if the compiler lacks it.

    OpenMP only parallelizes ProfTree's population loop (``prange``), which Cython compiles to an
    ordinary serial loop without it, so a compiler without OpenMP support (e.g. Apple clang without
    libomp) still builds a working, single-threaded package. Set ``EMPULSE_DISABLE_OPENMP=1`` to
    build without it regardless.
    """
    if os.environ.get('EMPULSE_DISABLE_OPENMP', '0') not in {'', '0'}:
        return [], []
    compiler = new_compiler()
    customize_compiler(compiler)
    if compiler.compiler_type == 'msvc':
        candidates = [(['/openmp'], [])]
    elif sys.platform == 'darwin':
        # Apple clang needs the preprocessor flag and an explicit libomp; GCC takes -fopenmp.
        candidates = [(['-Xpreprocessor', '-fopenmp'], ['-lomp']), (['-fopenmp'], ['-fopenmp'])]
    else:
        candidates = [(['-fopenmp'], ['-fopenmp'])]

    probe = textwrap.dedent("""
        #include <omp.h>
        int main(void) {
            int n = 0;
            #pragma omp parallel reduction(+:n)
            n += 1;
            return n > 0 ? 0 : 1;
        }
    """)
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, 'openmp_probe.c')
        with open(source, 'w', encoding='utf-8') as f:
            f.write(probe)
        for compile_flags, link_flags in candidates:
            try:
                objects = compiler.compile([source], output_dir=tmp, extra_postargs=compile_flags)
                compiler.link_executable(objects, 'openmp_probe', output_dir=tmp, extra_postargs=link_flags)
            except (CompileError, LinkError):
                continue
            return compile_flags, link_flags
    warnings.warn('OpenMP is unavailable: ProfTreeClassifier will ignore n_jobs and run single-threaded.', stacklevel=1)
    return [], []


if __name__ == '__main__':
    openmp_compile_flags, openmp_link_flags = openmp_flags()
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
            'empulse.metrics._cy_max_profit.piecewise',
            ['empulse/metrics/_cy_max_profit/*.pyx'],
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
            extra_compile_args=openmp_compile_flags,
            extra_link_args=openmp_link_flags,
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
