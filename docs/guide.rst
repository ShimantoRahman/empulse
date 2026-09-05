==========
User Guide
==========

These pages cover each part of Empulse in depth: what every component does, the parameters that
matter, and how it fits into a scikit-learn workflow.

They are written as reference material to dip into once you know roughly what you need. If you are
new to the package, read the :doc:`getting_started/quickstart` first, then the :doc:`tutorial`,
which works through a complete problem end to end. :doc:`getting_started/overview` maps common
problems to the component that solves them.

The sections build on each other in the order below. **Metrics** comes first because everything
else consumes a cost matrix or a metric: the models train on them, the samplers resample by them,
and the datasets ship them.

.. toctree::
    :maxdepth: 2
    :numbered:

    guide/metrics_guide.rst
    guide/models_guide.rst
    guide/preprocessing_guide.rst
    guide/instance_based_cv.rst
    guide/datasets_guide.rst
