========
Tutorial
========

This tutorial works through one problem from beginning to end: a telecom company wants to run a
retention campaign, and needs to decide which customers to contact.

We use the :ref:`Iranian churn dataset <iranian_churn>`, which is well suited to value-driven
modelling because every customer carries their own lifetime value. That makes the interesting
question not "who is most likely to leave?" but "who is most profitable to keep?" — and those two
questions have different answers.

Along the way you will see why a model with a good ROC AUC can still leave most of the available
value uncaptured, how to write a cost matrix that captures the campaign's economics, how to train a
model on it, and how to choose the threshold and validate the whole thing.

.. note::
    Needs ``pip install empulse[boosting] pandas``. The dataset downloads once and is cached under
    ``~/empulse_data``.

    Each page is self-contained: the first code block re-loads the data, so you can jump in
    anywhere.

If you only have five minutes, read the :doc:`getting_started/quickstart` instead.

.. grid:: 1
    :gutter: 2
    :class-container: step-cards

    .. grid-item-card:: The Problem with AUC
        :link: tutorial_problem
        :link-type: ref

        Why a classifier that looks good by every conventional measure still leaves most of the
        campaign's money on the table.

    .. grid-item-card:: Writing the Cost Matrix
        :link: tutorial_cost_matrix
        :link-type: ref

        What each of the four outcomes is worth, written in terms of lifetime value, contact cost,
        incentive and accept rate.

    .. grid-item-card:: Measuring in Money
        :link: tutorial_evaluate
        :link-type: ref

        Turn that matrix into a number, and see what the strategies say about the same set of
        predictions.

    .. grid-item-card:: Training on the Cost Matrix
        :link: tutorial_train
        :link-type: ref

        Pass the matrix into training, so the model chases business value instead of accuracy.

    .. grid-item-card:: Deciding Who to Actually Contact
        :link: tutorial_threshold
        :link-type: ref

        Where to draw the line. The default 0.5 is a convention, and with asymmetric costs it is
        almost always the wrong one.

    .. grid-item-card:: Validating the Whole Thing
        :link: tutorial_pipeline
        :link-type: ref

        Cross-validate and tune the pipeline, with each customer's costs following them into the
        right fold.

.. toctree::
    :maxdepth: 2
    :numbered:
    :hidden:

    tutorial/01_problem.rst
    tutorial/02_cost_matrix.rst
    tutorial/03_evaluate.rst
    tutorial/04_train.rst
    tutorial/05_threshold.rst
    tutorial/06_pipeline.rst
