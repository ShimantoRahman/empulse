.. _bias_mitigation:

===============
Bias Mitigation
===============

In many business problems,
there is often an inverse relationship between the likelihood of an event and the value of a subgroup.
For example, in customer churn scenarios,
high-value customers might be less likely to churn compared to low-value customers.
As a result, models designed to predict this event may focus more on the majority subgroup,
which is often less valuable from a business perspective.
This leads to suboptimal targeting, as the models neglect the minority subgroup that holds higher value.

We can say that the models are biased towards the majority subgroup.
To address this issue, we can use bias mitigation techniques from fairness literature.
The goal is to improve the representation of the high-value minority subgroup within the targeted group
by treating this subgroup as the ‘discriminated’ sensitive group.

Three techniques are available for bias mitigation:

1. `Relabeling`_: This technique relabels the training data to remove the bias.
2. `Resampling`_: This technique resamples the training data to remove the bias.
3. `Reweighing`_: This technique assigns weights to the training data to remove the bias.

Relabeling
==========

To reduce bias,
a subset of training labels from the high-value subgroup—called promotion candidates—
is switched from negative (non-event) to positive (event).
At the same time, an equal number of training labels from the low-value subgroup—called demotion candidates
—are switched from positive (event) to negative (non-event).
This will make the prediction model predict more high-value instances as events,
while reducing the number of low-value instances classified as events.

The key is selecting which subset to flip the labels to minimize the impact on predictive accuracy.
For this purpose, a first iteration of the prediction model is used to rank all high-value non-events
from the highest likelihood to the lowest.

The top :math:`N` non-events are then selected for promotion.
Conversely, low-value events are ranked from the lowest to the highest likelihood,
and the bottom :math:`N` events are chosen as demotion candidates.
The parameter :math:`N` is carefully chosen to create a discrimination-free classifier.

A second discrimination-free prediction model is then trained on the massaged dataset.
Note that only the training data is altered to retain objective model evaluation on holdout data.

To use the relabeling technique, you can use the :class:`~empulse.samplers.BiasRelabler` sampler.
You should pass the model which is used to rank the high-value non-events and low-value events.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.samplers import BiasRelabler

    X, y = make_classification(random_state=42)
    high_clv = np.random.randint(0, 2, X.shape[0])

    relabler = BiasRelabler(estimator=LogisticRegression())
    X_relabeled, y_relabeled = relabler.fit_resample(X, y, sensitive_feature=high_clv)

This can easily used inside a imbalanced-learn :class:`imblearn:imblearn.pipeline.Pipeline`
(note that the scikit-learn :class:`sklearn:sklearn.pipeline.Pipeline` does not support samplers):

.. code-block:: python

    from imblearn.pipeline import Pipeline
    from sklearn import config_context

    with config_context(enable_metadata_routing=True):
        pipeline = Pipeline([
            ('sampler', BiasRelabler(
                LogisticRegression()
            ).set_fit_resample_request(sensitive_feature=True)),
            ('model', LogisticRegression())
        ])

        pipeline.fit(X, y, sensitive_feature=high_clv)

Alternatively, the :class:`~empulse.model.BiasRelabelingClassifier` does this in one step and
can be used with scikit learn pipelines.
It will use the same model to rank the high-value non-events and low-value events as to train the final model.

.. code-block:: python

    from empulse.models import BiasRelabelingClassifier

    model = BiasRelabelingClassifier(estimator=LogisticRegression())
    model.fit(X, y, sensitive_feature=high_clv)

If you have a continuous feature which you want to dynamically convert to a binary sensitive feature,
you can pass a function to the ``transform_feature`` parameter.
This works for both the :class:`~empulse.samplers.BiasRelabler` and :class:`~empulse.model.BiasRelabelingClassifier`.

For example, here we convert the clv feature to a high clv indicator
if the clv is in the top 20% of all clv values in the training data.
This can be useful to avoid accidental data leakage.

.. code-block:: python

    clv = np.random.rand(X.shape[0]) * 100

    model = BiasRelabelingClassifier(
        estimator=LogisticRegression(),
        transform_feature=lambda clv: (clv > np.quantile(clv, 0.8)).astype(int)
    )
    model.fit(X, y, sensitive_feature=clv)

By default the relabeling technique will try to achieve statistical parity.
However, if you wish to use a different strategy, you can pass a function to the ``strategy`` parameter.
This function takes the true labels and the sensitive feature as input and should return how many labels to flip.

For example, here we flip the labels of the top 10% of high clv customers and the bottom 10% of low clv customers:

.. code-block:: python

    model = BiasRelabelingClassifier(
        estimator=LogisticRegression(),
        strategy=lambda y, sensitive_feature: (sensitive_feature == 1).sum() // 10
    )
    model.fit(X, y, sensitive_feature=high_clv)

Resampling
==========

The resampling method computes the weights for each group in the training data.
The calculation of these weights involves comparing class labels and membership
in both high-value and low-value segments.
Each weight represents the ratio between the expected probability of a particular class label being identified and
being a member of the high-value segment, and the observed probability of the same.
The expected probability assumes no discrimination is present, or in other words,
the probability when events are randomly distributed over the high- and low-value segments.

These weights are then used to systematically under- or oversample each group in proportion to their weight.
In this process, overrepresented groups are undersampled, while underrepresented groups are oversampled.
This approach is particularly useful for algorithms where you cannot pass sample weights during training.

To use the relabeling technique, you can use the :class:`~empulse.samplers.BiasResampler` sampler.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.samplers import BiasResampler

    X, y = make_classification(random_state=42)
    high_clv = np.random.randint(0, 2, X.shape[0])

    resampler = BiasResampler()
    X_resampled, y_resampled = resampler.fit_resample(X, y, sensitive_feature=high_clv)

This can easily used inside a imbalanced-learn :class:`imblearn:imblearn.pipeline.Pipeline`
(note that the scikit-learn :class:`sklearn:sklearn.pipeline.Pipeline` does not support samplers):

.. code-block:: python

    from imblearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    with config_context(enable_metadata_routing=True):
        pipeline = Pipeline([
            ('sampler', BiasResampler().set_fit_resample_request(sensitive_feature=True)),
            ('model', LogisticRegression())
        ])

        pipeline.fit(X, y, sensitive_feature=high_clv)

Alternatively, the :class:`~empulse.model.BiasResamplingClassifier` does this in one step and
can be used with scikit learn pipelines.
You should pass the model which is fitted with the resampled data.

.. code-block:: python

    from empulse.models import BiasResamplingClassifier

    model = BiasResamplingClassifier(LogisticRegression())
    model.fit(X, y, sensitive_feature=high_clv)

If you have a continuous feature which you want to dynamically convert to a binary sensitive feature,
you can pass a function to the ``transform_feature`` parameter.
This works for both the :class:`~empulse.samplers.BiasResampler` and :class:`~empulse.model.BiasResamplingClassifier`.

For example, here we convert the clv feature to a high clv indicator
if the clv is in the top 20% of all clv values in the training data.
This can be useful to avoid accidental data leakage.

.. code-block:: python

    clv = np.random.rand(X.shape[0]) * 100

    model = BiasResamplingClassifier(
        estimator=LogisticRegression(),
        transform_feature=lambda clv: (clv > np.quantile(clv, 0.8)).astype(int)
    )
    model.fit(X, y, sensitive_feature=clv)

By default the resampling technique will try to achieve statistical parity.
However, if you wish to use a different strategy, you can pass a function to the ``strategy`` parameter.
This function takes the true labels and the sensitive feature as input and
should return a 2X2 numpy array with the weights for each group,
where the rows represent the true labels and the columns represent the sensitive feature.

For example, here we assign a weight of 2 to high clv customers who are events and
a weight of 0.5 to low clv customers who are non-events:

.. code-block:: python

    model = BiasResamplingClassifier(
        estimator=LogisticRegression(),
        strategy=lambda y, sensitive_feature: np.array([[0.5, 1], [1, 2]])
    )
    model.fit(X, y, sensitive_feature=high_clv)

Reweighing
==========

In the reweighing approach the same weights as the resampling method are used.
However, instead of resampling the data, the weights are used to influence the training process.
The weights are passed to the training algorithm to adjust the loss function.
This way, the algorithm gives more weight to underrepresented groups and less weight to overrepresented groups.

To use the relabeling technique, you can use the :class:`~empulse.model.BiasReweighingClassifier`.
You should pass the model which is fitted with the computed sample weights.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.models import BiasReweighingClassifier

    X, y = make_classification(random_state=42)
    high_clv = np.random.randint(0, 2, X.shape[0])

    model = BiasReweighingClassifier(estimator=LogisticRegression())
    model.fit(X, y, sensitive_feature=high_clv)

If you have a continuous feature which you want to dynamically convert to a binary sensitive feature,
you can pass a function to the ``transform_feature`` parameter.

For example, here we convert the clv feature to a high clv indicator
if the clv is in the top 20% of all clv values in the training data.
This can be useful to avoid accidental data leakage.

.. code-block:: python

    clv = np.random.rand(X.shape[0]) * 100

    model = BiasReweighingClassifier(
        estimator=LogisticRegression(),
        transform_feature=lambda clv: (clv > np.quantile(clv, 0.8)).astype(int)
    )
    model.fit(X, y, sensitive_feature=clv)

By default the reweighing technique will try to achieve statistical parity.
However, if you wish to use a different strategy, you can pass a function to the ``strategy`` parameter.
This function takes the true labels and the sensitive feature as input and
should return a 1D numpy array containing the sample weights

For example, here we assign a weight of 0.5 to low clv customers and a weight of 1 to high clv customers:

.. code-block:: python

    def strategy(y_true, sensitive_feature):
        sample_weights = np.ones(len(sensitive_feature))
        sample_weights[np.where(sensitive_feature == 0)] = 0.5
        return sample_weights

    model = BiasReweighingClassifier(
        estimator=LogisticRegression(),
        strategy=strategy
    )
    model.fit(X, y, sensitive_feature=high_clv)