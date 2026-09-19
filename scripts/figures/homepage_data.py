"""Numbers the documentation homepage draws, recomputed with Empulse.

The homepage hero is not an illustration but a chart the browser draws from data, so it cannot be
an SVG pair like the rest of ``scripts/figures``. It shares the same rule, though: every number on
it is computed here by the package itself, so the headline claim beside the chart cannot quietly
disagree with the curve above it.

Only bundled datasets are used, so ``just figures`` needs no network and produces the same file on
every machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def hero_threshold_curve() -> dict[str, Any]:
    """Expected cost per customer as the decision threshold sweeps from 0 to 1.

    One fixed model scored once. Only the cut-off applied to its predictions moves, which is the
    homepage's whole claim: the 0.5 every classifier defaults to is a choice, and on a priced
    problem it is usually the wrong one.
    """
    import pandas as pd
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder

    from empulse.datasets import load_upsell_bank_telemarketing
    from empulse.metrics import Cost, Metric

    dataset = load_upsell_bank_telemarketing(backend=pd)
    features, target = dataset.data, np.asarray(dataset.target)

    # The row indices travel with the split so the per-customer costs can be sliced the same way.
    rows = np.arange(len(target))
    x_train, x_test, y_train, y_test, _, test_rows = train_test_split(
        features, target, rows, test_size=0.4, random_state=42, stratify=target
    )

    # The categorical columns only need to become numbers; which encoding is used does not change
    # what the figure shows, and an ordinal one keeps the pipeline short.
    model = make_pipeline(
        make_column_transformer(
            (
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                make_column_selector(dtype_include=object),
            ),
            remainder='passthrough',
        ),
        HistGradientBoostingClassifier(random_state=42),
    ).fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]

    instance_costs = {name: np.asarray(values)[test_rows] for name, values in dataset.instance_costs.items()}
    expected_cost = Metric(dataset.cost_matrix, Cost())

    thresholds = np.linspace(0.0, 1.0, 101)
    costs = np.array([expected_cost(y_test, (y_score >= t).astype(float), **instance_costs) for t in thresholds])

    best = int(costs.argmin())
    default = int(np.abs(thresholds - 0.5).argmin())

    return {
        'dataset': dataset.name,
        'samples': len(y_test),
        'unit': 'cost per customer',
        'thresholds': [round(float(t), 3) for t in thresholds],
        'costs': [round(float(c), 4) for c in costs],
        'default': {'threshold': round(float(thresholds[default]), 3), 'cost': round(float(costs[default]), 3)},
        'optimal': {'threshold': round(float(thresholds[best]), 3), 'cost': round(float(costs[best]), 3)},
        'reduction': round(float(1 - costs[best] / costs[default]), 4),
    }


DATA = {
    'homepage_hero': hero_threshold_curve,
}
