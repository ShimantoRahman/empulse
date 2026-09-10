.. _imbalance:

====================================
Cost sensitivity and class imbalance
====================================

Cost-sensitive learning and imbalanced classification are neighbours, and the two problems are
routinely confused. They are not the same problem, and the standard imbalance remedy —
``class_weight='balanced'`` — is a *guess* at a cost matrix. Sometimes a good one. Sometimes it
makes things worse.

This page places the imbalance toolkit next to the three things Empulse can do instead, so you can
tell which problem you actually have.

Rarity is not cost
==================

Imbalance is a fact about the **data**: one class occurs far less often than the other. Cost
asymmetry is a fact about the **business**: one kind of mistake hurts more than the other.

They often coincide — fraud, churn and default are all rare and all expensive to miss — and that
coincidence is why weighting by frequency often helps. But the two ratios are independent, and
``class_weight='balanced'`` sets the weight ratio to the *imbalance* ratio, which is a claim about
cost that nobody checked.

Watch what happens when the two ratios disagree. Here the positive class is about 17 times rarer
than the negative one, and we score the same three models under two different cost matrices:

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from empulse.metrics import Cost, CostMatrix, Metric
    from empulse.models import CSLogitClassifier

    X, y = make_classification(n_samples=3000, weights=[0.95], n_informative=6, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    print(f'imbalance ratio: {(1 - y_train.mean()) / y_train.mean():.1f} to 1')

    for fn_cost in (3.0, 20.0):
        matrix = (
            CostMatrix()
            .add_fp_cost('c_fp')
            .add_fn_cost('c_fn')
            .set_default(c_fp=1.0, c_fn=fn_cost)
        )
        expected_cost = Metric(matrix, Cost())

        models = {
            'plain': LogisticRegression(max_iter=500),
            'balanced': LogisticRegression(max_iter=500, class_weight='balanced'),
            'cost-sensitive': CSLogitClassifier(fp_cost=1, fn_cost=fn_cost),
        }
        print(f'--- a false negative costs {fn_cost:g}x a false positive')
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.predict_proba(X_test)[:, 1]
            print(f'  {name:15s} cost={expected_cost(y_test, score):.3f}')

At a 20:1 cost ratio — close to the 17:1 imbalance — reweighting helps: 0.859 down to 0.660. The
cost-sensitive model still does better, at 0.537, because it is given the real ratio rather than a
proxy for it.

At a 3:1 cost ratio the picture inverts. Reweighting makes the model **worse** than doing nothing,
0.338 against 0.162, because it corrects for a 17:1 asymmetry that the business does not have. The
cost-sensitive model, at 0.151, beats both.

The lesson is not that ``class_weight`` is bad. It is that ``'balanced'`` encodes a specific cost
assumption — that the cost ratio equals the inverse class ratio — and if you know the real costs
there is no reason to assume anything.

.. note::
    If you know your costs but want to keep an ordinary estimator, you can pass them as a
    ``class_weight`` dict: ``class_weight={0: fp_cost, 1: fn_cost}``. That is a genuine
    cost-sensitive approach and often a reasonable first step. Its limits are that it only handles
    class-dependent costs — no per-row values — and that it cannot express benefits on the
    diagonal.

Three places to intervene
=========================

Once you have a cost matrix, there are three points in the pipeline where it can act. They are not
alternatives so much as different constraints on what you are allowed to change.

.. list-table::
    :widths: 22 40 38
    :header-rows: 1

    * - Change
      - How
      - Use when
    * - The **objective**
      - :doc:`../training` — a model that optimises the cost matrix during fitting
      - You are free to choose the model.
    * - The **threshold**
      - :doc:`../deciding` — keep the model, move the cut-off
      - The model is already trained, or owned by someone else, and you only control the decision
        rule.
    * - The **data**
      - :ref:`cost_sampling` — resample so costly instances appear more often
      - You must use a specific estimator that has no cost-sensitive variant and no
        ``sample_weight``.

Changing the threshold deserves particular emphasis, because it is the cheapest of the three and is
frequently sufficient. A well-ranked model with a badly chosen cut-off is a solved problem; you do
not need to retrain anything.

Cost-proportionate sampling
===========================

:ref:`cost_sampling` covers the sampler in detail. The one-line version is that
:class:`~empulse.samplers.CostSensitiveSampler` draws instances with probability proportional to
their misclassification cost, so an ordinary estimator fitted on the resampled data behaves
approximately as if it had been trained on the cost matrix.

Unlike ``class_weight``, this handles **instance-dependent** costs: two customers of the same class
can be sampled at different rates because they are worth different amounts.

Reading the results
===================

One consequence trips up almost everyone arriving from imbalanced classification: **a
cost-sensitive model usually scores worse on accuracy, and that is correct.**

It is deliberately trading many cheap false positives for a few expensive false negatives. Accuracy
counts both equally, so it sees the trade as a loss. So does F1, and so does anything else built
from counts rather than money.

Judge these models with :doc:`../measuring` — a cost, a savings ratio or a profit — and treat a
drop in accuracy as evidence the objective changed, not that the model got worse. Precision and
recall remain useful for *understanding* what the model is doing; they are just not the thing being
optimised.

Where next
==========

- :ref:`cost_sampling` — the sampler in detail.
- :ref:`bias_mitigation` — a different way of reshaping the training data, aimed at a subgroup
  rather than a class.
- :doc:`../training` — changing the objective instead.
- :ref:`threshold_tuning` — changing only the cut-off.
