"""Remote cost-sensitive datasets fetched from OpenML and UCI ML Repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import narwhals as nw

from ._base import Dataset, get_data_home
from ._cost_matrices import churn_retention_cost_matrix, credit_scoring_cost_matrix
from ._io import _fetch_openml, _fetch_uci, load_or_fetch
from ._process import (
    _GIVE_ME_SOME_CREDIT_FEATURE_ORDER,  # noqa: F401 (re-exported for tests/compat)
    process_give_me_some_credit,
    process_iranian_churn,
)

_IRANIAN_CHURN_DESCRIPTION = """\
Iranian Churn Dataset
=====================

A churn dataset from an Iranian telecom company, collected over 12 months.

The goal is to predict whether a customer will churn.
Target variable: 1 = churned, 0 = active.

Each customer has a ``Customer Value`` column representing their
customer lifetime value (CLV), which is used as an instance-dependent
cost driver for the retention cost matrix.

=================   ==============
Classes                          2
Churners                       495
Non-churners                  2755
Samples                       3150
Features                        12
=================   ==============

Source
------
UCI Machine Learning Repository — Iranian Churn Dataset (ID 563).
https://doi.org/10.24432/C5JW3Z

References
----------
.. [1] Jafari-Marandi, R., Denton, J., Idris, A., Smith, B. K., & Keramati, A.
       (2020). Optimum profit-driven churn decision making:
       innovative artificial neural networks in telecom industry.
       Neural Computing and Applications, 32(18), 14929–14962.
"""

_GIVE_ME_SOME_CREDIT_DESCRIPTION = """\
Give Me Some Credit
===================

Originally from the 2011 Kaggle competition hosted by Credit Fusion.
The goal is to predict whether a customer will experience serious
financial distress in the next two years.
Target variable: 1 = defaulted, 0 = no default.

Only customers with positive monthly income and a debt ratio < 1 are kept.

=================   ==============
Classes                          2
Defaulters                    7616
Non-defaulters              105299
Samples                     112915
Features                        10
=================   ==============

Source
------
OpenML — GiveMeSomeCredit.

References
----------
.. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
       "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
       in Proceedings of the International Conference on Machine Learning
       and Applications, 2014.
"""


# ---------------------------------------------------------------------------
# Iranian Churn — UCI ML Repository
# ---------------------------------------------------------------------------


def fetch_iranian_churn(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
    incentive_fraction: float = 0.05,
    contact_fraction: float = 0.01,
    accept_rate: float = 0.3,
) -> Dataset[Any, Any]:
    """
    Fetch the Iranian Churn dataset from the UCI ML Repository (binary classification).

    The goal is to predict whether a telecom customer will churn.
    The target variable is whether the customer churned, 1 = churned, 0 = active.

    Uses the UCI ML Repository API (stdlib only — no extra dependencies).

    For additional information about the dataset,
    consult the :ref:`User Guide <iranian_churn>`.

    =================   ==============
    Classes                          2
    Churners                       495
    Non-churners                  2755
    Samples                       3150
    Features                        12
    =================   ==============

    Parameters
    ----------
    backend : module
        Dataframe library to use for ``data`` and ``target``.
        Pass the library module directly, e.g. ``backend=polars`` or
        ``backend=pandas``.
    data_home : str or Path, optional
        Directory used for caching downloaded data.
        Defaults to ``~/empulse_data`` (or ``$EMPULSE_DATA_HOME``).
    download_if_missing : bool, default=True
        If False, raise an ``OSError`` when the data is not cached locally.
    incentive_fraction : float, default=0.05
        Fraction of CLV offered as retention incentive (:math:`d`).
    contact_fraction : float, default=0.01
        Fraction of CLV spent on contacting the customer (:math:`f`).
    accept_rate : float, default=0.3
        Probability that a churner accepts the retention offer (:math:`\\gamma`).

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'clv': array}`` — the customer lifetime
        value of each customer, extracted from the ``Customer Value`` column.

    Notes
    -----
    Cost matrix (deterministic, :math:`\\gamma` treated as a fixed scalar):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_benefit`` :math:`= \\gamma (CLV_i - d \\cdot CLV_i - f \\cdot CLV_i)
            - (1-\\gamma) f \\cdot CLV_i`
          - ``fp_cost`` :math:`= d \\cdot CLV_i + f \\cdot CLV_i`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= CLV_i`
          - ``tn_cost`` :math:`= 0`

    with :math:`d` = ``incentive_fraction``, :math:`f` = ``contact_fraction``,
    :math:`\\gamma` = ``accept_rate``.

    References
    ----------
    .. [1] Jafari-Marandi, R., Denton, J., Idris, A., Smith, B. K., & Keramati, A.
           (2020). Optimum profit-driven churn decision making.
           Neural Computing and Applications, 32(18), 14929–14962.

    Examples
    --------

    .. code-block:: python

        import pandas as pd
        from empulse.datasets import fetch_iranian_churn
        from empulse.metrics import Metric, Cost

        dataset = fetch_iranian_churn(backend=pd)
        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'iranian_churn.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_uci_iranian_churn_raw,
        download_if_missing=download_if_missing,
        dataset_name='Iranian Churn dataset',
    )

    feature_df, target_series, clv = process_iranian_churn(raw, backend)
    cost_matrix, instance_costs = churn_retention_cost_matrix(
        clv,
        incentive_fraction=incentive_fraction,
        contact_fraction=contact_fraction,
        accept_rate=accept_rate,
    )

    return Dataset(
        data=nw.to_native(feature_df),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feature_df.columns,
        target_names=['no churn', 'churn'],
        name='Iranian Churn',
        DESCR=_IRANIAN_CHURN_DESCRIPTION,
    )


def _fetch_uci_iranian_churn_raw() -> dict[str, list[str | None]]:
    """Download the Iranian Churn dataset from UCI and return a flat string dict."""
    features, targets = _fetch_uci(563)
    return {col: list(arr.astype(str)) for col, arr in {**features, **targets}.items()}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Give Me Some Credit — OpenML
# ---------------------------------------------------------------------------


def fetch_give_me_some_credit(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
    interest_rate: float = 0.0479,
    fund_cost: float = 0.0294,
    max_credit_line: float = 25000,
    loss_given_default: float = 0.75,
    term_length_months: int = 24,
    loan_to_income_ratio: float = 3,
) -> Dataset[Any, Any]:
    """
    Fetch the "Give Me Some Credit" dataset from OpenML (binary classification).

    The goal is to predict whether a customer will default on a loan in the
    next two years.
    Target variable: 1 = defaulted, 0 = no default.

    Downloads data directly from OpenML without requiring scikit-learn.
    Downloaded data is cached locally.

    Only customers with positive monthly income and a debt ratio below 1 are kept.

    For a full data description and additional information about the dataset,
    consult the :ref:`User Guide <give_me_some_credit>`.

    =================   ==============
    Classes                          2
    Defaulters                    7616
    Non-defaulters              105299
    Samples                     112915
    Features                        10
    =================   ==============

    Parameters
    ----------
    backend : module
        Dataframe library to use for ``data`` and ``target``.
        Pass the library module directly, e.g. ``backend=polars`` or
        ``backend=pandas``.
    data_home : str or Path, optional
        Directory used for caching downloaded data.
        Defaults to ``~/empulse_data`` (or ``$EMPULSE_DATA_HOME``).
    download_if_missing : bool, default=True
        If False, raise an ``OSError`` when the data is not cached locally.
    interest_rate : float, default=0.0479
        Annual loan interest rate.
    fund_cost : float, default=0.0294
        Annual cost of funds.
    max_credit_line : float, default=25000
        Maximum credit line per client.
    loss_given_default : float, default=0.75
        Fraction of credit line lost on default.
    term_length_months : int, default=24
        Loan term in months.
    loan_to_income_ratio : float, default=3
        Loan-to-monthly-income ratio.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains:

        - ``'cl'``: estimated credit line per client.
        - ``'fp_cost'``: precomputed FP cost per client.

    Notes
    -----
    Cost matrix

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= 0`
          - ``fp_cost`` (precomputed, depends on dataset-level statistics)
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= Cl_i \\cdot L_{gd}`
          - ``tn_cost`` :math:`= 0`

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning
           and Applications, 2014.

    Examples
    --------

    .. code-block:: python

        import pandas as pd
        from empulse.datasets import fetch_give_me_some_credit
        from empulse.metrics import Metric, Cost

        dataset = fetch_give_me_some_credit(backend=pd)
        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'give_me_some_credit.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_openml_give_me_some_credit_raw,
        download_if_missing=download_if_missing,
        dataset_name='Give Me Some Credit dataset',
    )

    feat, target_series, monthly_income_np, debt_ratio_np, target_np = process_give_me_some_credit(raw, backend)
    cost_matrix, instance_costs = credit_scoring_cost_matrix(
        monthly_income_np,
        debt_ratio=debt_ratio_np,
        target_np=target_np,
        interest_rate=interest_rate,
        fund_cost=fund_cost,
        cl_max=max_credit_line,
        loss_given_default=loss_given_default,
        term_length_months=term_length_months,
        loan_to_income_ratio=loan_to_income_ratio,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['no default', 'default'],
        name='Give Me Some Credit',
        DESCR=_GIVE_ME_SOME_CREDIT_DESCRIPTION,
    )


def _fetch_openml_give_me_some_credit_raw() -> dict[str, list[str | None]]:
    """Download Give Me Some Credit from OpenML and return a flat string dict."""
    try:
        return _fetch_openml(name='GiveMeSomeCredit', version=1)  # type: ignore[return-value]
    except Exception as exc:
        raise OSError(f'Failed to download the GiveMeSomeCredit dataset from OpenML. Original error: {exc}') from exc
