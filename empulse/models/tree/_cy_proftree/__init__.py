"""
Compiled Cython extensions for :class:`~empulse.models.ProfTreeClassifier`.

This package holds only the ``.pyx``/``.pxd`` sources and the extension modules they compile to;
the plain-Python estimator lives one level up, at ``empulse/models/tree/proftree.py``, and imports
``EvolutionaryTree`` from here directly (``from ._cy_proftree.evolutionary_tree import
EvolutionaryTree``).
"""
