"""Numbers the documentation homepage draws, recomputed with Empulse.

The homepage hero is not an illustration but a chart the browser draws from data, so it cannot be
an SVG pair like the rest of ``scripts/figures``. It shares the same rule, though: every number on
it is computed here by the package itself, so the headline claim beside the chart cannot quietly
disagree with the curve under it.

Only bundled datasets are used, so ``just figures`` needs no network and produces the same file on
every machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)
DEFAULT_THRESHOLD = 0.5


def hero_profit_comparison() -> dict[str, Any]:
    """Profit of a standard and a cost-sensitive logistic regression across decision thresholds.

    Both models see the same features and the same split of the TV-subscription churn data; only
    the objective differs. Profit is the retention cost saved compared with contacting nobody,
    priced with each customer's own costs from the dataset's cost matrix.

    Each model's operating point is the threshold that maximises profit on the *training* split, so
    the test-set numbers the page prints are what that choice would actually have earned.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from empulse.datasets import load_churn_tv_subscriptions
    from empulse.metrics import Cost, Metric
    from empulse.models import CSLogitClassifier

    dataset = load_churn_tv_subscriptions(backend=pd)
    features, target = dataset.data, np.asarray(dataset.target)

    # The row indices travel with the split so the per-customer costs can be sliced the same way.
    rows = np.arange(len(target))
    x_train, x_test, y_train, y_test, train_rows, test_rows = train_test_split(
        features, target, rows, test_size=0.3, random_state=42
    )
    costs = {name: np.asarray(values) for name, values in dataset.instance_costs.items()}
    train_costs = {name: values[train_rows] for name, values in costs.items()}
    test_costs = {name: values[test_rows] for name, values in costs.items()}

    expected_cost = Metric(dataset.cost_matrix, Cost())

    def profit_curve(model: Any, x: Any, y: np.ndarray, instance_costs: dict[str, np.ndarray]) -> tuple:
        """Total profit and the number of customers contacted, at every threshold."""
        y_score = model.predict_proba(x)[:, 1]
        n_customers = len(y)
        contact_nobody = expected_cost(y, np.zeros(n_customers), **instance_costs) * n_customers
        profit = [
            contact_nobody - expected_cost(y, (y_score >= t).astype(float), **instance_costs) * n_customers
            for t in THRESHOLDS
        ]
        contacted = [int((y_score >= t).sum()) for t in THRESHOLDS]
        return np.asarray(profit), contacted

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x_train, y_train)
    cost_sensitive = make_pipeline(StandardScaler(), CSLogitClassifier(loss=expected_cost)).fit(
        x_train, y_train, **{f'cslogitclassifier__{name}': values for name, values in train_costs.items()}
    )

    models = {}
    for key, label, model in (
        ('baseline', 'LogisticRegression()', baseline),
        ('cost_sensitive', 'CSLogitClassifier()', cost_sensitive),
    ):
        train_profit, _ = profit_curve(model, x_train, y_train, train_costs)
        test_profit, contacted = profit_curve(model, x_test, y_test, test_costs)
        models[key] = {
            'label': label,
            'profit': [round(float(value)) for value in test_profit],
            'contacted': contacted,
            'tuned_index': int(np.argmax(train_profit)),
        }

    return {
        'dataset': dataset.name,
        'samples': len(y_test),
        'thresholds': [float(t) for t in THRESHOLDS],
        'default_index': int(np.flatnonzero(THRESHOLDS == DEFAULT_THRESHOLD)[0]),
        'models': models,
    }


DATA = {
    'homepage_hero': hero_profit_comparison,
}
