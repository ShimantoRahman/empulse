.. _deciding:

======================
Deciding who to act on
======================

A trained model gives you a score per instance. Turning that into an action — contact this
customer, reject this application — needs one more decision: where to draw the line.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Scores, probabilities and calibration
        :link: calibration
        :link-type: ref

        What a score has to mean before a cost computed from it is trustworthy, and how to get
        there.

    .. grid-item-card:: Threshold Tuning
        :link: csthreshold
        :link-type: ref

        Picking the cut-off itself, either as a probability threshold or as a fraction of the
        population to target.

.. toctree::
    :maxdepth: 2
    :hidden:

    deciding/calibration.rst
    deciding/thresholds.rst
