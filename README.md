[![Python Version](https://img.shields.io/pypi/v/empulse)](https://pypi.org/project/empulse/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ShimantoRahman/empulse)
![](https://img.shields.io/pypi/pyversions/empulse)
![Tests](https://github.com/ShimantoRahman/empulse/actions/workflows/tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/654945788.svg)](https://zenodo.org/doi/10.5281/zenodo.11185663)

# Empulse

Empulse is a package aimed to enable value-driven analysis in Python.
The package implements popular value-driven metrics and algorithms in accordance to sci-kit learn conventions.
This allows the measures to seamlessly integrate into existing ML workflows.

## Installation

Install `empulse` via pip with

```bash
pip install empulse
```

## Documentation
You can find the documentation [here](https://shimantorahman.github.io/empulse/).

## Usage

We offer custom metrics and models.
You can use them within the scikit-learn ecosystem.

```python
# the scikit learn stuff we love
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# the stuff we add
from empulse.metrics import empc_score
from empulse.models import ProfLogitClassifier

X, y = make_classification()

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", ProfLogitClassifier())
])

cross_val_score(pipeline, X, y, scoring=make_scorer(empc_score, needs_proba=True))
```
