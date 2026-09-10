.. _deciding:

======================
Deciding who to act on
======================

A trained model gives you a score per instance. Turning that into an action — contact this
customer, reject this application — needs one more decision: where to draw the line.

:doc:`deciding/calibration` covers what a score has to mean before a cost number computed from it
is trustworthy. :doc:`deciding/thresholds` covers picking the cut-off itself, either as a
probability threshold or as a fraction of the population to target.

.. toctree::
    :maxdepth: 2

    deciding/calibration.rst
    deciding/thresholds.rst
