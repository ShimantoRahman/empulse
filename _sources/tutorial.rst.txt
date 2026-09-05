========
Tutorial
========

This tutorial works through one problem from beginning to end: a telecom company wants to run a
retention campaign, and needs to decide which customers to contact.

We use the :ref:`Iranian churn dataset <iranian_churn>`, which is well suited to value-driven
modelling because every customer carries their own lifetime value. That makes the interesting
question not "who is most likely to leave?" but "who is most profitable to keep?" — and those two
questions have different answers.

Along the way you will see why a model with a good ROC AUC can still lose money, how to write a
cost matrix that captures the campaign's economics, how to train a model on it, and how to choose
the threshold and validate the whole thing.

.. note::
    Needs ``pip install empulse[optional] pandas``. The dataset downloads once and is cached under
    ``~/empulse_data``.

    Each page is self-contained: the first code block re-loads the data, so you can jump in
    anywhere.

If you only have five minutes, read the :doc:`getting_started/quickstart` instead.

.. toctree::
    :maxdepth: 2
    :numbered:

    tutorial/01_problem.rst
    tutorial/02_cost_matrix.rst
    tutorial/03_evaluate.rst
    tutorial/04_train.rst
    tutorial/05_threshold.rst
    tutorial/06_pipeline.rst
