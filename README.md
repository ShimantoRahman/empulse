[![PyPI Downloads](https://static.pepy.tech/badge/empulse)](https://pepy.tech/projects/empulse)
[![Python Version](https://img.shields.io/pypi/v/empulse)](https://pypi.org/project/empulse/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ShimantoRahman/empulse)
![](https://img.shields.io/pypi/pyversions/empulse)
![Tests](https://github.com/ShimantoRahman/empulse/actions/workflows/tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/654945788.svg)](https://zenodo.org/doi/10.5281/zenodo.11185663)

# Empulse

Empulse is a package aimed to enable value-driven and cost-sensitive analysis in Python.
The package implements popular value-driven and cost-sensitive metrics and algorithms 
in accordance to sci-kit learn conventions.
This allows the measures to seamlessly integrate into existing ML workflows.

## Installation

Install `empulse` via pip with

```bash
pip install empulse
```

## Documentation
You can find the documentation [here](https://shimantorahman.github.io/empulse/).

## Usage

We offer custom metrics, models and samplers.
You can use them within the scikit-learn ecosystem.

```python
# the scikit learn stuff we love
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# the stuff we add
from empulse.metrics import expected_cost_loss
from empulse.models import CSLogitClassifier

set_config(enable_metadata_routing=True)

X, y = make_classification()

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", CSLogitClassifier().set_fit_request(fp_cost=True, fn_cost=True))
])

scorer = make_scorer(
    expected_cost_loss,
    response_method='predict_proba',
    greater_is_better=False,
    fp_cost=1,
    fn_cost=1
)

cross_val_score(
    pipeline,
    X,
    y,
    scoring=scorer,
    params={"fp_cost": 1, "fn_cost": 1}
).mean()
```
