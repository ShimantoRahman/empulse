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

**Not every mistake costs the same.**

Accuracy, F1 and AUC treat a false positive and a false negative as equally bad. Your business does
not. Flagging a loyal customer as a churner wastes a discount; missing a real churner loses their
entire lifetime value. A model tuned for accuracy quietly optimises the wrong thing.

Empulse lets you write down what each outcome is actually worth, then use that same definition to
**evaluate** models, **train** them, and **set their decision threshold** — all as ordinary
scikit-learn estimators.

## Installation

Empulse requires Python 3.11 or higher.

```bash
pip install empulse
```

XGBoost, LightGBM and CatBoost are optional extras needed for the boosting models:

```bash
pip install empulse[optional]
```

## A model that makes money

On a real telecom churn dataset, an ordinary logistic regression reaches **0.93 ROC AUC** and
**89% accuracy** — and *loses 7.20 per customer* once the retention campaign is priced in. Training
the same data against the cost matrix turns that into a **profit of 2.78 per customer**.

```python
import pandas as pd
from empulse.datasets import fetch_iranian_churn
from empulse.metrics import Cost, Metric
from empulse.models import CSBoostClassifier

dataset = fetch_iranian_churn(backend=pd)
X, y = dataset.data, dataset.target
clv = dataset.instance_costs['clv']  # each customer's lifetime value

# The dataset ships a cost matrix; turn it into a metric
expected_cost = Metric(dataset.cost_matrix, Cost())

# Train a model that optimises it directly
model = CSBoostClassifier(loss=expected_cost)
model.fit(X, y, clv=clv)

print(expected_cost(y, model.predict_proba(X)[:, 1], clv=clv))
```

Define your own cost matrix just as easily:

```python
from empulse.metrics import Cost, CostMatrix, Metric

# A spam filter: a legitimate email marked as spam is 5x worse than spam getting through
cost_matrix = (
    CostMatrix()
    .add_fp_cost('missed_email')
    .add_fn_cost('wasted_time')
    .set_default(missed_email=5, wasted_time=1)
)
expected_cost = Metric(cost_matrix, Cost())
```

Read the [5-minute quickstart](https://empulse.readthedocs.io/en/stable/getting_started/quickstart.html)
or the [full tutorial](https://empulse.readthedocs.io/en/stable/tutorial.html).

## Features

| | |
|---|---|
| **[Profit-driven metrics](https://empulse.readthedocs.io/en/stable/reference/metrics.html)** | Ready-made metrics for [churn](https://empulse.readthedocs.io/en/stable/guide/metrics/prebuilt_churn_metrics.html), [acquisition](https://empulse.readthedocs.io/en/stable/guide/metrics/prebuilt_acquisition_metrics.html) and [credit scoring](https://empulse.readthedocs.io/en/stable/guide/metrics/prebuilt_credit_scoring_metrics.html), or [build your own](https://empulse.readthedocs.io/en/stable/guide/metrics/user_defined_value_metric.html) from a symbolic cost matrix. |
| **[Cost-sensitive models](https://empulse.readthedocs.io/en/stable/reference/models.html)** | Logistic regression, gradient boosting, trees, forests, bagging, minimax probability machines and evolutionary trees — all trained on your cost matrix. |
| **[Threshold tuning](https://empulse.readthedocs.io/en/stable/guide/models/threshold_tuning.html)** | Pick the cut-off that maximises profit instead of defaulting to 0.5, or classify the top-k% by score. |
| **[Instance-dependent costs](https://empulse.readthedocs.io/en/stable/guide/instance_based_cv.html)** | Costs that differ per row are routed through pipelines and cross-validation automatically via scikit-learn metadata routing. |
| **[Robustness](https://empulse.readthedocs.io/en/stable/guide/models/robustcs.html)** | `RobustCSClassifier` detects and imputes outliers in noisy instance-dependent costs. |
| **[Samplers](https://empulse.readthedocs.io/en/stable/reference/samplers.html)** | Cost-proportionate resampling and bias mitigation to make any estimator cost-sensitive. |
| **[Optimizers](https://empulse.readthedocs.io/en/stable/reference/optimizers.html)** | L-BFGS-B, SGD/Adam/RMSProp with learning-rate schedules, and genetic/memetic algorithms for non-smooth objectives. |
| **[Datasets](https://empulse.readthedocs.io/en/stable/guide/datasets_guide.html)** | Five real-world cost-sensitive datasets, each shipping its own cost matrix. |

## Works with scikit-learn

Every estimator follows scikit-learn conventions, so it drops into `Pipeline`, `GridSearchCV` and
`cross_val_score` unchanged, and every metric can be wrapped with `make_scorer`.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from empulse.models import CSLogitClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', CSLogitClassifier()),
])
search = GridSearchCV(pipeline, {'model__C': [0.1, 1, 10]})
search.fit(X, y, model__fp_cost=10, model__fn_cost=1)
```

## Documentation

Full documentation at [empulse.readthedocs.io](https://empulse.readthedocs.io/en/stable/):

- [Getting Started](https://empulse.readthedocs.io/en/stable/getting_started.html) — install, quickstart, and which tool you need
- [Tutorial](https://empulse.readthedocs.io/en/stable/tutorial.html) — a complete worked churn example
- [User Guide](https://empulse.readthedocs.io/en/stable/guide.html) — in-depth guides per component
- [API Reference](https://empulse.readthedocs.io/en/stable/api.html) — every class and function

## Citing

If you use Empulse in your research, please cite it via its
[Zenodo DOI](https://zenodo.org/doi/10.5281/zenodo.11185663).

## License

MIT — see [LICENSE](LICENSE).
