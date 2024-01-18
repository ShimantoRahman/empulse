# empulse

Empulse is a package aimed to enable value-driven analysis in Python.
The package implements popular value-driven metrics and algorithms in accordance to sci-kit learn conventions.
This allows the measures to seamlessly integrate into existing ML workflows..

## Installation

Install `empulse` via pip with

```bash
pip install empulse
```

## Documentation

TODO: add documentation

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
from sklearn.linear_model import LogisticRegression

# the stuff we add
from empulse.metrics import empc_score

X, y = make_classification()

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])

cross_val_score(pipeline, X, y, scoring=make_scorer(empc_score, needs_proba=True))
```
