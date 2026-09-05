"""Local cost-sensitive datasets bundled with empulse."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np

from ._base import Dataset
from ._cost_matrices import (
    churn_precomputed_cost_matrix,
    credit_scoring_cost_matrix,
    upsell_bank_cost_matrix,
)
from ._io import _read_csv_gz
from ._process import (
    process_bank_telemarketing,
    process_churn_tv,
    process_credit_scoring_pakdd,
)

if TYPE_CHECKING:
    from narwhals.typing import EagerAllowed, IntoBackend

_DATA_DIR = Path(__file__).parent / 'data'
_DESC_DIR = Path(__file__).parent / 'descriptions'


def _read_description(name: str) -> str:
    with open(_DESC_DIR / name, encoding='utf-8') as fh:
        return fh.read()


def load_churn_tv_subscriptions(*, backend: IntoBackend[EagerAllowed]) -> Dataset[Any, Any]:
    """
    Load the TV Subscription Churn dataset (binary classification).

    The goal is to predict whether a customer will churn or not.
    The target variable is whether the customer churned, 'yes' = 1 and 'no' = 0.

    This dataset is from a TV cable provider containing all 9410 customers
    active during the first semester of 2014.
    Feature names are anonymized to protect the privacy of the customers.

    For additional information about the dataset,
    consult the :ref:`User Guide <churn_tv_subscriptions>`.

    =================   ==============
    Classes                          2
    Churners                       455
    Non-churners                  8955
    Samples                       9410
    Features                        45
    =================   ==============

    Parameters
    ----------
    backend : module
        Dataframe library to use for ``data`` and ``target``.
        Pass the library module directly, e.g. ``backend=polars`` or
        ``backend=pandas``.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains the precomputed per-customer costs:
        ``tp_benefit``, ``fp_cost``, ``tn_benefit``, ``fn_cost``.

    Notes
    -----
    The per-instance costs stored in the dataset were derived from the
    churn retention cost model of Bahnsen et al. (2015).  The individual
    cost drivers (CLV, incentive cost, contact cost, acceptance probability)
    are not stored; only the final computed values are available.

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           `"A novel cost-sensitive framework for customer churn predictive modeling"
           <http://www.decisionanalyticsjournal.com/content/pdf/s40165-015-0014-6.pdf>`__,
           Decision Analytics, 2:5, 2015.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import load_churn_tv_subscriptions
        from empulse.metrics import Metric, Cost

        dataset = load_churn_tv_subscriptions(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    raw = _read_csv_gz(_DATA_DIR / 'churn_tv_subscriptions.csv.gz')
    df = nw.from_dict(raw, backend=backend)
    feature_df, target_series, instance_costs = process_churn_tv(df)
    cost_matrix = churn_precomputed_cost_matrix()

    return Dataset(
        data=nw.to_native(feature_df),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feature_df.columns,
        target_names=['no churn', 'churn'],
        name='Churn TV Subscriptions',
        DESCR=_read_description('churn_tv_subscriptions.rst'),
    )


def load_upsell_bank_telemarketing(*, backend: IntoBackend[EagerAllowed]) -> Dataset[Any, Any]:
    """
    Load the bank telemarketing dataset (binary classification).

    The goal is to predict whether a client will subscribe to a term deposit
    after being called by the bank.
    The target variable is whether the client subscribed, 'yes' = 1 and 'no' = 0.

    Features recorded *after* the contact event are excluded to avoid data leakage.
    Only clients with a positive balance are considered.

    For a full data description and additional information about the dataset,
    consult the :ref:`User Guide <upsell_bank_telemarketing>`.

    =================   ==============
    Classes                          2
    Subscribers                   4787
    Non-subscribers              33144
    Samples                      37931
    Features                        10
    =================   ==============

    Parameters
    ----------
    backend : module
        Dataframe library to use for ``data`` and ``target``.
        Pass the library module directly, e.g. ``backend=polars`` or
        ``backend=pandas``.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'balance': array}`` — the client's
        average yearly balance in euros, which drives the false-negative cost.

    Notes
    -----
    Cost matrix

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= c`
          - ``fp_cost`` :math:`= c`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= \\max(r \\cdot d \\cdot balance_i,\\; c)`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``interest_rate`` (:math:`r`) = 0.02463333
    - ``term_deposit_fraction`` (:math:`d`) = 0.25
    - ``contact_cost`` (:math:`c`) = 1.0

    To override these defaults, pass the desired values when evaluating the metric::

        metric(dataset.target, y_score, interest_rate=0.03, **dataset.instance_costs)

    References
    ----------
    .. [1] Moro, S., Rita, P., & Cortez, P. (2014).
           Bank Marketing [Dataset]. UCI Machine Learning Repository.
           https://doi.org/10.24432/C5K306.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import load_upsell_bank_telemarketing
        from empulse.metrics import Metric, Cost

        dataset = load_upsell_bank_telemarketing(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    raw = _read_csv_gz(_DATA_DIR / 'bankmarketing.csv.gz', delimiter=';')
    df = nw.from_dict(raw, backend=backend)
    feature_df, target_series, balance = process_bank_telemarketing(df)
    cost_matrix, instance_costs = upsell_bank_cost_matrix(
        balance,
        interest_rate=0.02463333,
        term_deposit_fraction=0.25,
        contact_cost=1.0,
    )

    return Dataset(
        data=nw.to_native(feature_df),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feature_df.columns,
        target_names=['no subscription', 'subscription'],
        name='Bank Telemarketing',
        DESCR=_read_description('bankmarketing.rst'),
    )


def load_credit_scoring_pakdd(*, backend: IntoBackend[EagerAllowed]) -> Dataset[Any, Any]:
    """
    Load the credit scoring PAKDD 2009 competition dataset (binary classification).

    The goal is to predict whether a customer will default on a loan in the
    next two years.
    The target variable is whether the customer defaulted, 'yes' = 1 and 'no' = 0.

    Only clients with a personal income between 100 and 10000 are considered.

    For a full data description and additional information about the dataset,
    consult the :ref:`User Guide <credit_scoring_pakdd>`.

    =================   ==============
    Classes                          2
    Defaulters                    7743
    Non-defaulters               31195
    Samples                      38938
    Features                        25
    =================   ==============

    Parameters
    ----------
    backend : module
        Dataframe library to use for ``data`` and ``target``.
        Pass the library module directly, e.g. ``backend=polars`` or
        ``backend=pandas``.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains:

        - ``'cl'``: estimated credit line per client (used in ``fn_cost = cl * lgd``).
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
          - ``fp_cost`` :math:`= r_i - (1 - \\pi_1) \\bar{r} + \\pi_1 \\overline{Cl} L_{gd}`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= Cl_i \\cdot L_{gd}`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``loss_given_default`` (:math:`L_{gd}`) = 0.75

    The following parameters are used to precompute the per-instance credit lines
    and ``fp_cost`` values stored in ``instance_costs``:

    - ``interest_rate`` = 0.63 (annual)
    - ``fund_cost`` = 0.165 (annual)
    - ``max_credit_line`` = 25000
    - ``term_length_months`` = 24
    - ``loan_to_income_ratio`` = 3

    To override the symbolic default, pass the desired value when evaluating the metric::

        metric(dataset.target, y_score, loss_given_default=0.6, **dataset.instance_costs)

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning
           and Applications, 2014.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import load_credit_scoring_pakdd
        from empulse.metrics import Metric, Cost

        dataset = load_credit_scoring_pakdd(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    raw = _read_csv_gz(
        _DATA_DIR / 'creditscoring2.csv.gz',
        delimiter='\t',
        null_values=['N', ''],
    )
    df = nw.from_dict(raw, backend=backend)
    feature_df, target_series, monthly_income = process_credit_scoring_pakdd(df)
    target_np = target_series.to_numpy()

    cost_matrix, instance_costs = credit_scoring_cost_matrix(
        monthly_income,
        debt_ratio=np.zeros(len(target_np)),
        target_np=target_np,
        interest_rate=0.63,
        fund_cost=0.165,
        cl_max=25000 * 0.33,
        loss_given_default=0.75,
        term_length_months=24,
        loan_to_income_ratio=3,
    )

    return Dataset(
        data=nw.to_native(feature_df),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feature_df.columns,
        target_names=['no default', 'default'],
        name='Credit Scoring PAKDD 2009',
        DESCR=_read_description('creditscoring2.rst'),
    )
