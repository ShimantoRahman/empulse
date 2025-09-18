[![PyPI Downloads](https://static.pepy.tech/badge/empulse)](https://pepy.tech/projects/empulse)
[![Python Version](https://img.shields.io/pypi/v/empulse)](https://pypi.org/project/empulse/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ShimantoRahman/empulse)
![](https://img.shields.io/pypi/pyversions/empulse)
![Tests](https://github.com/ShimantoRahman/empulse/actions/workflows/tests.yml/badge.svg)
[![Docs](https://img.shields.io/readthedocs/empulse)](https://empulse.readthedocs.io/en/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/654945788.svg)](https://zenodo.org/doi/10.5281/zenodo.11185663)

# Empulse

<a href="https://empulse.readthedocs.io/en/latest/"><img src="https://empulse.readthedocs.io/en/latest/_static/assets/empulse_logo_light.png" alt="Empulse Logo" width="25%" height="25%" align="right" /></a>

<!-- start-of-readme-intro -->

Empulse is a package aimed to enable value-driven and cost-sensitive analysis in Python.
The package implements popular value-driven and cost-sensitive metrics and algorithms 
in accordance to sci-kit learn conventions.
This allows the measures to seamlessly integrate into existing ML workflows.

## Installation

Empulse requires python 3.10 or higher.

Install `empulse` via pip with

```bash
pip install empulse
```

<!-- end-of-readme-install -->

## Documentation
You can find the documentation [here](https://empulse.readthedocs.io/en/stable/).

<!-- end-of-readme-intro -->

## Features

- [Ready to use out of the box with scikit-learn](#ready-to-use-out-of-the-box-with-scikit-learn)
- [Use case specific profit and cost metrics](#use-case-specific-profit-and-cost-metrics)
- [Build your own profit and cost metrics](#build-your-own-profit-and-cost-metrics)
- [Various profit-driven and cost-sensitive models](#various-profit-driven-and-cost-sensitive-models)
- [Easy passing of instance-dependent costs](#easy-passing-of-instance-dependent-costs)
- [Cost-aware resampling and relabeling](#cost-aware-resampling-and-relabeling)
- [Find the optimal decision threshold](#find-the-optimal-decision-threshold)
- [Easy access to real-world datasets for benchmarking](#easy-access-to-real-world-datasets-for-benchmarking)

## Take the tour

### Ready to use out of the box with scikit-learn

All components of the package are designed to work seamlessly with scikit-learn.

Models are implemented as scikit-learn estimators and can be used anywhere a scikit-learn estimator can be used.

#### Pipelines
```python
from empulse.models import CSLogitClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification()
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", CSLogitClassifier())
])
pipeline.fit(X, y, model__fp_cost=10, model__fn_cost=1)
```

#### Cross-validation
```python
from sklearn.model_selection import cross_val_score

cross_val_score(
    pipeline, 
    X, 
    y, 
    scoring="roc_auc", 
    params={"model__fp_cost": 10, "model__fn_cost": 1}
)
```

#### Grid search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {"model__C": [0.1, 1, 10]}
grid_search = GridSearchCV(pipeline, param_grid, scoring="roc_auc")
grid_search.fit(X, y, model__fp_cost=10, model__fn_cost=1)
```

All metrics can easily be converted as scikit-learn scorers 
and can be used in the same way as any other scikit-learn scorer.

```python
from empulse.metrics import expected_cost_loss
from sklearn.metrics import make_scorer

scorer = make_scorer(
    expected_cost_loss, 
    response_method="predict_proba", 
    greater_is_better=False,
    fp_cost=10,
    fn_cost=1
)

cross_val_score(
    pipeline, 
    X, 
    y, 
    scoring=scorer, 
    params={"model__fp_cost": 10, "model__fn_cost": 1}
)
```

### Use-case specific profit and cost metrics

Empulse offers a wide range of profit and cost metrics that are tailored to specific use cases such as:
- [customer churn](https://empulse.readthedocs.io/en/stable/reference/metrics.html#customer-churn-metrics), 
- [customer acquisition](https://empulse.readthedocs.io/en/stable/reference/metrics.html#customer-acquisition-metrics),
- [credit scoring](https://empulse.readthedocs.io/en/stable/reference/metrics.html#credit-scoring-metrics),
- and fraud detection (coming soon).

For other use cases, the package provides generic implementations for:
- the [cost loss](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.cost_loss.html),
- the [expected cost loss](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.expected_cost_loss.html),
- the [expected log cost loss](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.expected_log_cost_loss.html),
- the [savings score](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.savings_score.html),
- the [expected savings score](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.expected_savings_score.html),
- and the [maximum profit score](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.max_profit_score.html).


### Build your own profit and cost metrics

The [Metric](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.Metric.html) class allows
you to easily build your own profit and cost metrics.

First, you start to define the cost matrix 
through the [CostMatrix](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.CostMatrix.html)
class.

A tiny example: suppose you’re building a spam filter.
- False positive (legit email marked as spam) costs 5, because you might miss something important
- False negative (spam slips through) costs 1, because it wastes your time

```python
from empulse.metrics import CostMatrix

cost_matrix = CostMatrix().add_fp_cost('fp').add_fn_cost('fn')
cost_matrix.alias({'opportunity_cost': 'fp', 'time_wasted_cost': 'fn'})
```

Now, you can define your own profit and cost metrics using different strategies.

- [Cost](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.Cost.html): lower is better.
- [Savings](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.Savings.html): higher is better.
- [MaxProfit](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.MaxProfit.html):
  higher is better.

```python
from empulse.metrics import Metric, Cost, Savings, MaxProfit

expected_cost_loss = Metric(cost_matrix=cost_matrix, strategy=Cost()) 
expected_savings_score = Metric(cost_matrix=cost_matrix, strategy=Savings())
expected_max_profit_score = Metric(cost_matrix=cost_matrix, strategy=MaxProfit())

expected_cost_loss(
    y, pipeline.predict_proba(X)[:, 1], opportunity_cost=5, time_wasted_cost=1
)
```

Your custom metric can also be optimized by the models:

```python
from empulse.models import CSBoostClassifier

csboost = CSBoostClassifier(loss=expected_cost_loss)
csboost.fit(X, y, opportunity_cost=5, time_wasted_cost=1)
```

Read more in the [User Guide](https://empulse.readthedocs.io/en/stable/guide/metrics/user_defined_value_metric.html).

### Various profit-driven and cost-sensitive models

Empulse provides a range of profit-driven and cost-sensitive models such as:
- [CSLogitClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.CSLogitClassifier.html),
- [CSBoostClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.CSBoostClassifier.html),
- [CSTreeClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.CSTreeClassifier.html),
- [CSForestClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.CSForestClassifier.html),
- [CSBaggingClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.CSBaggingClassifier.html),
- [B2BoostClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.B2BoostClassifier.html),
- [RobustCSClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.RobustCSClassifier.html),
- [ProfLogitClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.ProfLogitClassifier.html),
- [BiasRelabelingClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.BiasRelabelingClassifier.html),
- [BiasResamplingClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.BiasResamplingClassifier.html),
- and [BiasReweighingClassifier](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.BiasReweighingClassifier.html).

Each classifier tries to mimic the behaviour of sklearn's classifiers with a cost-sensitive twist.

```python
from empulse.models import CSTreeClassifier
from sklearn.tree import DecisionTreeClassifier

cstree = CSTreeClassifier(max_depth=2, min_samples_leaf=1, random_state=42)
dtree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, random_state=42)

cstree.fit(X, y, fp_cost=10, fn_cost=1)
dtree.fit(X, y)
```

### Easy passing of instance-dependent costs

Instance-dependent costs can easily be passed to the models through [metadata routing](https://scikit-learn.org/stable/metadata_routing.html).

For instance, the instance-dependent costs are passed dynamically to each fold of the cross-validation
through requesting the costs in the `set_fit_request` method of the model 
and the `set_score_request` method of the scorer.
    
```python
import numpy as np
from empulse.models import CSTreeClassifier
from empulse.metrics import expected_cost_loss
from sklearn import set_config
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

set_config(enable_metadata_routing=True)

X, y = make_classification()
fp_cost = np.random.rand(y.size)
fn_cost = np.random.rand(y.size)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", CSTreeClassifier().set_fit_request(fp_cost=True, fn_cost=True))
])

scorer = make_scorer(
    expected_cost_loss,
    response_method="predict_proba",
    greater_is_better=False,
).set_score_request(fp_cost=True, fn_cost=True)

cross_val_score(pipeline, X, y, scoring=scorer, params={"fp_cost": fp_cost, "fn_cost": fn_cost})
```

### Cost-aware resampling and relabeling

Empulse uses the [imbalanced-learn](https://imbalanced-learn.org/) 
package to provide cost-aware resampling and relabeling techniques:
- [CostSensitiveSampler](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.samplers.CostSensitiveSampler.html)
- [BiasResampler](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.samplers.BiasResampler.html)
- [BiasRelabler](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.samplers.BiasRelabler.html)

```python
from empulse.samplers import CostSensitiveSampler
from sklearn.datasets import make_classification

X, y = make_classification()
sampler = CostSensitiveSampler()
X_resampled, y_resampled = sampler.fit_resample(X, y, fp_cost=2, fn_cost=1)
```

They can be used in an imbalanced-learn pipeline:

```python
import numpy as np
from empulse.samplers import CostSensitiveSampler
from imblearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

set_config(enable_metadata_routing=True)

X, y = make_classification()
fp_cost = np.random.rand(y.size)
fn_cost = np.random.rand(y.size)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("sampler", CostSensitiveSampler().set_fit_resample_request(fp_cost=True, fn_cost=True)),
    ("model", LogisticRegression())
])

pipeline.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)
```

### Find the optimal decision threshold

Empulse provides the 
[`CSThresholdClassifier`](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.models.CSThresholdClassifier.html)
which allows you to find the optimal decision threshold for a given cost matrix to minimize the expected cost loss.

The meta-estimator changes the `predict` method of the base estimator to predict the class with the lowest expected cost.

```python
from empulse.models import CSThresholdClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification()
model = CSThresholdClassifier(estimator=LogisticRegression())
model.fit(X, y)
model.predict(X, fp_cost=2, fn_cost=1)
```

Metrics like the maximum profit score conveniently return the optimal target threshold.
For example, the Expected Maximum Profit measure for customer churn (EMPC) 
tells you what fraction of the customer base should be targeted to maximize profit.

```python
from empulse.metrics import empc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification()
model = LogisticRegression()
predictions = model.fit(X, y).predict_proba(X)[:, 1]

score, threshold = empc(y, predictions, clv=50)
```

This score can then be converted to a decision threshold by using the 
[`classification_threshold`](https://empulse.readthedocs.io/en/stable/reference/generated/empulse.metrics.classification_threshold.html) 
function.

```python
from empulse.metrics import classification_threshold

decision_threshold = classification_threshold(y, predictions, customer_threshold=threshold)
```

This can then be combined with sci-kit learn's 
[`FixedThresholdClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.FixedThresholdClassifier.html)
to create a model that predicts the class with the highest expected profit.

```python
from sklearn.model_selection import FixedThresholdClassifier

model = FixedThresholdClassifier(estimator=model, threshold=decision_threshold)
model.predict(X)
```

### Easy access to real-world datasets for benchmarking

Empulse provides easy access to real-world datasets for benchmarking cost-sensitive models.

Each dataset returns the features, the target, and the instance-dependent costs, ready to use in a cost-sensitive model.

```python
from empulse.datasets import load_give_me_some_credit
from empulse.models import CSLogitClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y, tp_cost, fp_cost, tn_cost, fn_cost = load_give_me_some_credit(return_X_y_costs=True)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', CSLogitClassifier())
])
pipeline.fit(
    X, 
    y, 
    model__tp_cost=tp_cost, 
    model__fp_cost=fp_cost, 
    model__tn_cost=tn_cost, 
    model__fn_cost=fn_cost
)
```

<!-- end-of-readme-usage -->
