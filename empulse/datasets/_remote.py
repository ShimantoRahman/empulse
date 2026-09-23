"""Remote cost-sensitive datasets fetched from OpenML and UCI ML Repository."""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import narwhals as nw
import numpy as np

from ._base import Dataset, get_data_home
from ._cost_matrices import (
    churn_monthly_charges_cost_matrix,
    churn_retention_cost_matrix,
    churn_retention_monthly_cost_matrix,
    credit_scoring_cost_matrix,
    credit_scoring_known_cl_cost_matrix,
    direct_marketing_cost_matrix,
    fraud_detection_cost_matrix,
)
from ._io import _fetch_csv_url, _fetch_openml, _fetch_uci, _fetch_url_bytes, _read_csv_columns, load_or_fetch
from ._process import (
    _GIVE_ME_SOME_CREDIT_FEATURE_ORDER,  # ruff: ignore[unused-import] (re-exported for tests/compat)
    KDD98_ATTRIBUTES,
    process_cell2cell,
    process_credit_card_fraud,
    process_default_credit_card_clients,
    process_give_me_some_credit,
    process_home_equity,
    process_ieee_fraud_detection,
    process_iranian_churn,
    process_kdd98,
    process_kddcup09_churn,
    process_south_german_credit,
    process_telco_customer_churn,
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


def fetch_iranian_churn(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
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
          - ``tp_benefit`` :math:`= \\gamma (CLV_i - d \\cdot CLV_i - f) - (1-\\gamma) f`
          - ``fp_cost`` :math:`= d \\cdot CLV_i + f`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= 0`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``incentive_fraction`` (:math:`d`) = 0.05
    - ``contact_cost`` (:math:`f`) = 1
    - ``accept_rate`` (:math:`\\gamma`) = 0.3

    To override these defaults, pass the desired values when evaluating the metric::

        metric(dataset.target, y_score, accept_rate=0.5, **dataset.instance_costs)

    References
    ----------
    .. [1] Jafari-Marandi, R., Denton, J., Idris, A., Smith, B. K., & Keramati, A.
           (2020). Optimum profit-driven churn decision making.
           Neural Computing and Applications, 32(18), 14929–14962.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_iranian_churn
        from empulse.metrics import Metric, Cost

        dataset = fetch_iranian_churn(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

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
        incentive_fraction=0.05,
        contact_cost=1,
        accept_rate=0.3,
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


def fetch_give_me_some_credit(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
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

    The cost matrix uses symbolic parameters with the following defaults:

    - ``loss_given_default`` (:math:`L_{gd}`) = 0.75

    The following parameters are used to precompute per-instance credit lines
    and ``fp_cost`` values stored in ``instance_costs``:

    - ``interest_rate`` = 0.0479 (annual)
    - ``fund_cost`` = 0.0294 (annual)
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
        from empulse.datasets import fetch_give_me_some_credit
        from empulse.metrics import Metric, Cost

        dataset = fetch_give_me_some_credit(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

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
        interest_rate=0.0479,
        fund_cost=0.0294,
        cl_max=25000,
        loss_given_default=0.75,
        term_length_months=24,
        loan_to_income_ratio=3,
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


_TELCO_CUSTOMER_CHURN_DESCRIPTION = """\
Kaggle Telco Customer Churn
===========================

Customer churn dataset from IBM Sample Data Sets, hosted on Kaggle and OpenML.
The goal is to predict whether a customer will churn.
Target variable: 1 = churned, 0 = active.

=================   ==============
Classes                          2
Churners                      1869
Non-churners                  5163
Samples                       7032
Features                        19
=================   ==============

Source
------
OpenML — telco-customer-churn (ID: 42178).
https://www.openml.org/d/42178

References
----------
.. [1] Petrides, G., & Verbeke, W. (2021). Cost-sensitive ensemble learning:
       a unifying framework. Data Mining and Knowledge Discovery, 1–28.
.. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. Information Sciences, 594, 400–415.
"""


def fetch_telco_customer_churn(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the Telco Customer Churn dataset from OpenML (binary classification).

    The goal is to predict whether a customer will churn or not.
    The target variable is whether the customer churned, 1 = churned, 0 = active.

    For additional information about the dataset,
    consult the :ref:`User Guide <telco_customer_churn>`.

    =================   ==============
    Classes                          2
    Churners                      1869
    Non-churners                  5163
    Samples                       7032
    Features                        19
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'monthly_charges': array}`` — the customer's
        monthly bill amount, which scales the false-negative and false-positive costs.

    Notes
    -----
    Cost matrix (Petrides & Verbeke 2021, Vanderschueren et al. 2022):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= 0`
          - ``fp_cost`` :math:`= fp\\_months \\cdot monthly\\_charges_i`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= fn\\_months \\cdot monthly\\_charges_i`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``fn_months`` = 12.0 (annual revenue lost upon churn)
    - ``fp_months`` = 2.0 (two months of retention offer cost)

    References
    ----------
    .. [1] Petrides, G., & Verbeke, W. (2021). Cost-sensitive ensemble learning:
           a unifying framework. Data Mining and Knowledge Discovery, 1–28.
    .. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
           Predict-then-optimize or predict-and-optimize? An empirical evaluation
           of cost-sensitive learning strategies. Information Sciences, 594, 400–415.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_telco_customer_churn
        from empulse.metrics import Metric, Cost

        dataset = fetch_telco_customer_churn(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'telco_customer_churn.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_openml_telco_customer_churn_raw,
        download_if_missing=download_if_missing,
        dataset_name='Telco Customer Churn dataset',
    )

    feat, target_series, monthly_charges = process_telco_customer_churn(raw, backend)
    cost_matrix, instance_costs = churn_monthly_charges_cost_matrix(
        monthly_charges,
        fn_months=12.0,
        fp_months=2.0,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['no churn', 'churn'],
        name='Telco Customer Churn',
        DESCR=_TELCO_CUSTOMER_CHURN_DESCRIPTION,
    )


def _fetch_openml_telco_customer_churn_raw() -> dict[str, list[str | None]]:
    """Download Telco Customer Churn from OpenML and return a flat string dict."""
    try:
        return _fetch_openml(data_id=42178)  # type: ignore[return-value]
    except Exception as exc:
        msg = f'Failed to download the Telco Customer Churn dataset from OpenML. Original error: {exc}'
        raise OSError(msg) from exc


_DEFAULT_CREDIT_CARD_CLIENTS_DESCRIPTION = """\
Default of Credit Card Clients
==============================

Credit card default dataset from Taiwan (Yeh & Lien, 2009), hosted on the UCI
ML Repository and OpenML.
The goal is to predict whether a customer will default on credit card payments.
Target variable: 1 = default, 0 = no default.

=================   ==============
Classes                          2
Defaulters                    6636
Non-defaulters               23364
Samples                      30000
Features                        23
=================   ==============

Source
------
OpenML — default-of-credit-card-clients (ID: 42477).
https://www.openml.org/d/42477

References
----------
.. [1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
       for the predictive accuracy of probability of default of credit card clients.
       Expert Systems with Applications, 36(2), 2473–2480.
.. [2] Bahnsen, A. C., Aouada, D., & Ottersten, B. (2014).
       Example-dependent cost-sensitive logistic regression for credit scoring.
       In 2014 13th International Conference on Machine Learning and Applications (pp. 263-269).
.. [3] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. Information Sciences, 594, 400–415.
"""


def fetch_default_credit_card_clients(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the Default of Credit Card Clients dataset from OpenML (binary classification).

    The goal is to predict whether a client will default on their credit card payment.
    Target variable: 1 = default, 0 = no default.

    For additional information about the dataset,
    consult the :ref:`User Guide <default_credit_card_clients>`.

    =================   ==============
    Classes                          2
    Defaulters                    6636
    Non-defaulters               23364
    Samples                      30000
    Features                        23
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains:

        - ``'cl'``: credit limit (``limit_bal``) per client.
        - ``'fp_cost'``: precomputed FP cost per client following Bahnsen et al. (2014).

    Notes
    -----
    Cost matrix (Bahnsen et al. 2014, Vanderschueren et al. 2022):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= 0`
          - ``fp_cost`` (precomputed per client)
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= Cl_i \\cdot L_{gd}`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``loss_given_default`` (:math:`L_{gd}`) = 0.75

    References
    ----------
    .. [1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
           for the predictive accuracy of probability of default of credit card clients.
           Expert Systems with Applications, 36(2), 2473–2480.
    .. [2] Bahnsen, A. C., Aouada, D., & Ottersten, B. (2014).
           Example-dependent cost-sensitive logistic regression for credit scoring.
           In 2014 13th International Conference on Machine Learning and Applications (pp. 263-269).

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_default_credit_card_clients
        from empulse.metrics import Metric, Cost

        dataset = fetch_default_credit_card_clients(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'default_of_credit_card_clients.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_openml_default_credit_card_raw,
        download_if_missing=download_if_missing,
        dataset_name='Default of Credit Card Clients dataset',
    )

    feat, target_series, credit_line, target_np = process_default_credit_card_clients(raw, backend)
    cost_matrix, instance_costs = credit_scoring_known_cl_cost_matrix(
        credit_line,
        target_np,
        interest_rate=0.0479,
        fund_cost=0.0294,
        loss_given_default=0.75,
        term_length_months=24,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['no default', 'default'],
        name='Default of Credit Card Clients',
        DESCR=_DEFAULT_CREDIT_CARD_CLIENTS_DESCRIPTION,
    )


def _fetch_openml_default_credit_card_raw() -> dict[str, list[str | None]]:
    """Download Default of Credit Card Clients from OpenML and return a flat string dict."""
    try:
        return _fetch_openml(data_id=42477)  # type: ignore[return-value]
    except Exception as exc:
        msg = f'Failed to download the Default of Credit Card Clients dataset from OpenML. Original error: {exc}'
        raise OSError(msg) from exc


_IEEE_FRAUD_DETECTION_DESCRIPTION = """\
IEEE-CIS Fraud Detection
========================

Real-world e-commerce fraud detection dataset from Vesta Corporation,
originally hosted on Kaggle and available on OpenML.
The goal is to predict whether a transaction is fraudulent.
Target variable: 1 = fraud, 0 = not fraud.

=================   ==============
Classes                          2
Frauds                       20663
Legitimate                  569877
Samples                     590540
Features                       431
=================   ==============

Source
------
OpenML — IEEE-CIS_Fraud_Detection (ID: 46858).
https://www.openml.org/d/46858

References
----------
.. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       European Journal of Operational Research, 297(1), 291–300.
.. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. Information Sciences, 594, 400–415.
"""


def fetch_ieee_fraud_detection(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the IEEE-CIS Fraud Detection dataset from OpenML (binary classification).

    The goal is to predict whether an e-commerce transaction is fraudulent.
    Target variable: 1 = fraud, 0 = not fraud.

    For additional information about the dataset,
    consult the :ref:`User Guide <ieee_fraud_detection>`.

    =================   ==============
    Classes                          2
    Frauds                       20663
    Legitimate                  569877
    Samples                     590540
    Features                       431
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'amount': array}`` — the transaction amount
        representing the loss if a fraudulent transaction is missed (false negative).

    Notes
    -----
    Cost matrix (Höppner et al. 2022, Vanderschueren et al. 2022):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= c_f`
          - ``fp_cost`` :math:`= c_f`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= amount_i`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``investigation_cost`` (:math:`c_f`) = 10.0 (cost of investigating an alert)

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291–300.
    .. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
           Predict-then-optimize or predict-and-optimize? An empirical evaluation
           of cost-sensitive learning strategies. Information Sciences, 594, 400–415.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_ieee_fraud_detection
        from empulse.metrics import Metric, Cost

        dataset = fetch_ieee_fraud_detection(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'ieee_fraud_detection.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_openml_ieee_fraud_detection_raw,
        download_if_missing=download_if_missing,
        dataset_name='IEEE-CIS Fraud Detection dataset',
    )

    feat, target_series, amount = process_ieee_fraud_detection(raw, backend)
    cost_matrix, instance_costs = fraud_detection_cost_matrix(
        amount,
        investigation_cost=10.0,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['legitimate', 'fraud'],
        name='IEEE-CIS Fraud Detection',
        DESCR=_IEEE_FRAUD_DETECTION_DESCRIPTION,
    )


def _fetch_openml_ieee_fraud_detection_raw() -> dict[str, list[str | None]]:
    """Download IEEE-CIS Fraud Detection from OpenML and return a flat string dict."""
    try:
        return _fetch_openml(data_id=46858)  # type: ignore[return-value]
    except Exception as exc:
        msg = f'Failed to download the IEEE-CIS Fraud Detection dataset from OpenML. Original error: {exc}'
        raise OSError(msg) from exc


_KDD98_DESCRIPTION = """\
KDD Cup 1998 Direct Mailing
===========================

Direct marketing dataset from the KDD Cup 1998 competition, hosted on the UCI KDD
Archive. It records a mailing campaign by a veterans' charity asking lapsed donors
for a new donation. The learning and validation sets are combined, as in
Vanderschueren et al. (2022).
The goal is to predict whether a recipient will respond to a donation mailing.
Target variable: 1 = donated, 0 = did not donate.

Only the 22 attributes selected by Petrides & Verbeke (2021) are kept. The donation
amount (``TARGET_D``) is not a feature: it is the false-negative cost.

=================   ==============
Classes                          2
Donors                        9716
Non-donors                  182063
Samples                     191779
Features                        22
=================   ==============

Source
------
UCI KDD Archive — KDD Cup 1998 Data.
https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html

References
----------
.. [1] Zadrozny, B., Langford, J., & Abe, N. (2003). Cost-sensitive learning by
       cost-proportionate example weighting. In Third IEEE International Conference
       on Data Mining (pp. 435-442).
.. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. Information Sciences, 594, 400–415.
"""


def fetch_kdd98(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the KDD Cup 1998 Direct Mailing dataset from the UCI KDD Archive (binary classification).

    The goal is to predict whether a recipient will donate in response to a direct mailing.
    Target variable: 1 = donated, 0 = did not donate.

    The learning (``cup98LRN``) and validation (``cup98VAL`` + ``valtargt``) sets are
    downloaded and combined, as in Vanderschueren et al. (2022). Only the 22 attributes
    selected by Petrides & Verbeke (2021) are kept. The download is about 75 MB and is
    cached locally.

    For additional information about the dataset,
    consult the :ref:`User Guide <kdd98>`.

    =================   ==============
    Classes                          2
    Donors                        9716
    Non-donors                  182063
    Samples                     191779
    Features                        22
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'amount': array}`` — the donation amount
        (``TARGET_D``), which is lost when a donor is not mailed. It is 0 for
        non-donors, for whom the false-negative cost never applies.

    Notes
    -----
    Cost matrix (Zadrozny et al. 2003, Vanderschueren et al. 2022):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= c_f`
          - ``fp_cost`` :math:`= c_f`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= amount_i`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``contact_cost`` (:math:`c_f`) = 0.68 (cost of one mailing, in dollars)

    References
    ----------
    .. [1] Zadrozny, B., Langford, J., & Abe, N. (2003). Cost-sensitive learning by
           cost-proportionate example weighting. In Third IEEE International Conference
           on Data Mining (pp. 435-442).
    .. [2] Petrides, G., & Verbeke, W. (2021). Cost-sensitive ensemble learning:
           a unifying framework. Data Mining and Knowledge Discovery, 1–28.
    .. [3] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
           Predict-then-optimize or predict-and-optimize? An empirical evaluation
           of cost-sensitive learning strategies. Information Sciences, 594, 400–415.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_kdd98
        from empulse.metrics import Metric, Cost

        dataset = fetch_kdd98(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'kdd98.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_uci_kdd98_raw,
        download_if_missing=download_if_missing,
        dataset_name='KDD Cup 1998 Direct Mailing dataset',
    )

    feat, target_series, amount = process_kdd98(raw, backend)
    cost_matrix, instance_costs = direct_marketing_cost_matrix(
        amount,
        contact_cost=0.68,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['no donation', 'donated'],
        name='KDD Cup 1998 Direct Mailing',
        DESCR=_KDD98_DESCRIPTION,
    )


_KDD98_URL = 'https://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/'


def _fetch_uci_kdd98_raw() -> dict[str, list[str | None]]:
    """Download the KDD Cup 1998 learning and validation sets and return a flat string dict."""
    targets = ('TARGET_B', 'TARGET_D')
    try:
        learning = _read_csv_columns(
            _fetch_url_bytes(_KDD98_URL + 'cup98lrn.zip'),
            (*KDD98_ATTRIBUTES, *targets),
        )
        validation = _read_csv_columns(
            _fetch_url_bytes(_KDD98_URL + 'cup98val.zip'),
            ('CONTROLN', *KDD98_ATTRIBUTES),
        )
        validation_targets = _read_csv_columns(
            _fetch_url_bytes(_KDD98_URL + 'valtargt.txt'),
            ('CONTROLN', *targets),
        )
    except (OSError, KeyError) as exc:
        raise OSError(f'Failed to download the KDD Cup 1998 dataset. Original error: {exc}') from exc

    # The validation labels ship in a separate file, keyed on the donor id.
    target_rows = {
        control: (target_b, target_d)
        for control, target_b, target_d in zip(
            validation_targets['CONTROLN'],
            validation_targets['TARGET_B'],
            validation_targets['TARGET_D'],
            strict=True,
        )
    }
    validation_labels = [target_rows[control] for control in validation['CONTROLN']]
    validation['TARGET_B'] = [target_b for target_b, _ in validation_labels]
    validation['TARGET_D'] = [target_d for _, target_d in validation_labels]

    return {col: learning[col] + validation[col] for col in (*KDD98_ATTRIBUTES, *targets)}  # type: ignore[misc]


_CREDIT_CARD_FRAUD_DESCRIPTION = """\
Credit Card Fraud Detection (ULB MLG)
=====================================

Transactions made by European cardholders in September 2013, collected during a
research collaboration between Worldline and the Machine Learning Group (MLG) of ULB.
The dataset is highly imbalanced: 492 frauds out of 284,807 transactions.
Following Hoppner et al. (2022) and Vanderschueren et al. (2022), transactions with
zero amount are filtered out, leaving 282,982 samples (465 frauds).

Target variable: 1 = fraud, 0 = legitimate.

Features:
- ``V1``–``V28``: Principal components obtained with PCA (due to confidentiality).
- ``Amount``: Transaction amount in euros.

=================   ==============
Classes                          2
Frauds                         465
Legitimate                  282517
Samples                     282982
Features                        29
=================   ==============

Source
------
OpenML — creditcard (ID: 1597).
https://www.openml.org/d/1597

References
----------
.. [1] Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
       Calibrating probability with undersampling for unbalanced classification.
       In 2015 IEEE Symposium Series on Computational Intelligence (pp. 159-166).
.. [2] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       European Journal of Operational Research, 297(1), 291–300.
.. [3] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. Information Sciences, 594, 400–415.
.. [4] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
       Robust instance-dependent cost-sensitive classification.
       Advances in Data Analysis and Classification, 1–23.
"""


def fetch_credit_card_fraud(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the Kaggle Credit Card Fraud (ULB MLG) dataset from OpenML (binary classification).

    The goal is to predict whether a credit card transaction is fraudulent.
    Target variable: 1 = fraud, 0 = legitimate.

    Transactions with zero amount are filtered out following Höppner et al. (2022)
    and Vanderschueren et al. (2022), resulting in 282,982 samples.

    For additional information about the dataset,
    consult the :ref:`User Guide <credit_card_fraud>`.

    =================   ==============
    Classes                          2
    Frauds                         465
    Legitimate                  282517
    Samples                     282982
    Features                        29
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'amount': array}`` — the transaction amount
        representing the loss if a fraudulent transaction is missed (false negative).

    Notes
    -----
    Cost matrix (Höppner et al. 2022, Vanderschueren et al. 2022):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= c_f`
          - ``fp_cost`` :math:`= c_f`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= amount_i`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``investigation_cost`` (:math:`c_f`) = 10.0 (cost of investigating an alert)

    References
    ----------
    .. [1] Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
           Calibrating probability with undersampling for unbalanced classification.
           In 2015 IEEE Symposium Series on Computational Intelligence (pp. 159-166).
    .. [2] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291–300.
    .. [3] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
           Predict-then-optimize or predict-and-optimize? An empirical evaluation
           of cost-sensitive learning strategies. Information Sciences, 594, 400–415.
    .. [4] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
           Robust instance-dependent cost-sensitive classification.
           Advances in Data Analysis and Classification, 1–23.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_credit_card_fraud
        from empulse.metrics import Metric, Cost

        dataset = fetch_credit_card_fraud(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'credit_card_fraud.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_openml_credit_card_fraud_raw,
        download_if_missing=download_if_missing,
        dataset_name='Credit Card Fraud Detection dataset',
    )

    feat, target_series, amount = process_credit_card_fraud(raw, backend)
    cost_matrix, instance_costs = fraud_detection_cost_matrix(
        amount,
        investigation_cost=10.0,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['legitimate', 'fraud'],
        name='Credit Card Fraud Detection',
        DESCR=_CREDIT_CARD_FRAUD_DESCRIPTION,
    )


def _fetch_openml_credit_card_fraud_raw() -> dict[str, list[str | None]]:
    """Download Credit Card Fraud Detection from OpenML and return a flat string dict."""
    try:
        return _fetch_openml(data_id=1597)  # type: ignore[return-value]
    except Exception as exc:
        msg = f'Failed to download the Credit Card Fraud Detection dataset from OpenML. Original error: {exc}'
        raise OSError(msg) from exc


_HOME_EQUITY_DESCRIPTION = """\
Home Equity (HMEQ) Dataset
==========================

Baseline and loan performance information for 5,960 recent home equity loans,
used to automate loan approval decisions following the Equal Credit Opportunity Act.
The goal is to predict whether an applicant will default or be seriously delinquent
(target: 1 = default, 0 = good credit).

=================   ==============
Classes                          2
Defaults                      1189
Non-defaults                  4771
Samples                       5960
Features                        12
=================   ==============

Source
------
OpenML — HMEQ_Data (ID: 43337).
https://www.openml.org/d/43337

References
----------
.. [1] Baesens, B., Roesch, D., & Scheule, H. (2016). Credit risk analytics:
       Measurement techniques, applications, and examples in SAS. John Wiley & Sons.
.. [2] Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability
       of model explanations in instance-dependent cost-sensitive credit scoring.
       European Journal of Operational Research, 326(2), 630–640.
"""


def fetch_home_equity(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the Home Equity (HMEQ) dataset from OpenML (binary classification).

    The goal is to predict whether a home equity loan applicant will default
    or be seriously delinquent.
    Target variable: 1 = default, 0 = no default.

    For additional information about the dataset,
    consult the :ref:`User Guide <home_equity>`.

    =================   ==============
    Classes                          2
    Defaults                      1189
    Non-defaults                  4771
    Samples                       5960
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains:

        - ``'cl'``: loan amount per borrower (``LOAN``).
        - ``'fp_cost'``: precomputed false positive cost per borrower.

    Notes
    -----
    Cost matrix (Bahnsen et al. 2014, Ballegeer et al. 2025):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= 0`
          - ``fp_cost`` (precomputed per borrower)
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= Cl_i \\cdot L_{gd}`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``loss_given_default`` (:math:`L_{gd}`) = 0.75

    References
    ----------
    .. [1] Baesens, B., Roesch, D., & Scheule, H. (2016). Credit risk analytics:
           Measurement techniques, applications, and examples in SAS. John Wiley & Sons.
    .. [2] Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability
           of model explanations in instance-dependent cost-sensitive credit scoring.
           European Journal of Operational Research, 326(2), 630–640.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_home_equity
        from empulse.metrics import Metric, Cost

        dataset = fetch_home_equity(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'home_equity.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_openml_home_equity_raw,
        download_if_missing=download_if_missing,
        dataset_name='Home Equity (HMEQ) dataset',
    )

    feat, target_series, amounts, target_np = process_home_equity(raw, backend)
    cost_matrix, instance_costs = credit_scoring_known_cl_cost_matrix(
        amounts,
        target_np,
        interest_rate=0.0479,
        fund_cost=0.0294,
        loss_given_default=0.75,
        term_length_months=24,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['no default', 'default'],
        name='Home Equity (HMEQ)',
        DESCR=_HOME_EQUITY_DESCRIPTION,
    )


def _fetch_openml_home_equity_raw() -> dict[str, list[str | None]]:
    """Download Home Equity from OpenML and return a flat string dict."""
    try:
        return _fetch_openml(data_id=43337)  # type: ignore[return-value]
    except Exception as exc:
        msg = f'Failed to download the Home Equity dataset from OpenML. Original error: {exc}'
        raise OSError(msg) from exc


_SOUTH_GERMAN_CREDIT_DESCRIPTION = """\
South German Credit
===================

Credit scoring dataset of 1,000 consumer loans from a southern German bank
(1973–1975), in the corrected version published by Grömping (2019). The widely used
Statlog "German Credit" data has the same origin but wrongly coded variables;
this dataset replaces it.
The goal is to predict whether a loan is a bad credit risk (target: 1 = bad, 0 = good).
Bad credits are oversampled in the data, so the 30% default rate is not the
bank's actual default rate.

Every feature except ``duration``, ``amount`` and ``age`` is an integer category
code; the code tables are in Grömping (2019).

=================   ==============
Classes                          2
Bad credit                     300
Good credit                    700
Samples                       1000
Features                        20
=================   ==============

Source
------
UCI Machine Learning Repository — South German Credit (ID 522).
https://archive.ics.uci.edu/dataset/522/south+german+credit

References
----------
.. [1] Grömping, U. (2019). South German Credit Data: Correcting a widely used data set.
       Reports in Mathematics, Physics and Chemistry, Report 4/2019, Department II,
       Beuth University of Applied Sciences Berlin.
.. [2] Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability
       of model explanations in instance-dependent cost-sensitive credit scoring.
       European Journal of Operational Research, 326(2), 630–640.
"""


def fetch_south_german_credit(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the South German Credit dataset from the UCI ML Repository (binary classification).

    The goal is to predict whether a loan is a bad credit risk.
    Target variable: 1 = bad credit, 0 = good credit.

    This is the corrected version of the Statlog "German Credit" data published by
    Grömping (2019), as used by Ballegeer et al. (2025). Every feature except
    ``duration``, ``amount`` and ``age`` is an integer category code; the code tables
    are in Grömping (2019).

    For additional information about the dataset,
    consult the :ref:`User Guide <south_german_credit>`.

    =================   ==============
    Classes                          2
    Bad credit                     300
    Good credit                    700
    Samples                       1000
    Features                        20
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains:

        - ``'cl'``: credit amount per borrower (``amount``, in Deutsche Mark).
        - ``'fp_cost'``: precomputed false positive cost per borrower.

    Notes
    -----
    Cost matrix (Ballegeer et al. 2025, following Bahnsen et al. 2014):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= 0`
          - ``fp_cost`` (precomputed per borrower)
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= Cl_i \\cdot L_{gd}`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``loss_given_default`` (:math:`L_{gd}`) = 0.75

    The false positive cost is precomputed with an annual interest rate of 4.79%,
    an annual cost of funds of 2.94% and a 24-month term.

    References
    ----------
    .. [1] Grömping, U. (2019). South German Credit Data: Correcting a widely used data set.
           Reports in Mathematics, Physics and Chemistry, Report 4/2019, Department II,
           Beuth University of Applied Sciences Berlin.
    .. [2] Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability
           of model explanations in instance-dependent cost-sensitive credit scoring.
           European Journal of Operational Research, 326(2), 630–640.
    .. [3] Bahnsen, A. C., Aouada, D., & Ottersten, B. (2014). Example-dependent
           cost-sensitive logistic regression for credit scoring. In 2014 13th International
           Conference on Machine Learning and Applications (pp. 263–269).

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_south_german_credit
        from empulse.metrics import Metric, Cost

        dataset = fetch_south_german_credit(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'south_german_credit.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_uci_south_german_credit_raw,
        download_if_missing=download_if_missing,
        dataset_name='South German Credit dataset',
    )

    feat, target_series, amounts, target_np = process_south_german_credit(raw, backend)
    cost_matrix, instance_costs = credit_scoring_known_cl_cost_matrix(
        amounts,
        target_np,
        interest_rate=0.0479,
        fund_cost=0.0294,
        loss_given_default=0.75,
        term_length_months=24,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['good credit', 'bad credit'],
        name='South German Credit',
        DESCR=_SOUTH_GERMAN_CREDIT_DESCRIPTION,
    )


_SOUTH_GERMAN_CREDIT_URL = 'https://archive.ics.uci.edu/static/public/522/south+german+credit.zip'


def _fetch_uci_south_german_credit_raw() -> dict[str, list[str | None]]:
    """Download South German Credit from the UCI archive and return a flat string dict."""
    try:
        archive_bytes = _fetch_url_bytes(_SOUTH_GERMAN_CREDIT_URL)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            lines = archive.read('SouthGermanCredit.asc').decode('latin-1').split('\n')
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise OSError(f'Failed to download the South German Credit dataset. Original error: {exc}') from exc

    rows = [line.split() for line in lines if line.strip()]
    header, body = rows[0], rows[1:]
    return {col: [row[i] for row in body] for i, col in enumerate(header)}


_KDDCUP09_CHURN_DESCRIPTION = """\
KDD Cup 2009 / Orange Customer Churn
====================================

Customer relationship management data provided by the French telecom operator
Orange for the ACM KDD Cup 2009 (small version).
Contains 50,000 customers described by 230 anonymised variables: Var1–Var190 are
numeric and Var191–Var229 are nominal (Var209 and Var230 are empty).
The goal is to predict customer churn (target: 1 = churn, 0 = no churn).

The variables are anonymised and include no revenue or customer value, so the cost
matrix uses the average customer lifetime value of Verbeke et al. (2012) and is the
same for every customer.

=================   ==============
Classes                          2
Churners                      3672
Non-churners                 46328
Samples                      50000
Features                       230
=================   ==============

Source
------
OpenML — KDDCup09_churn (ID: 1112).
https://www.openml.org/d/1112

References
----------
.. [1] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).
       New insights into churn prediction in the telecommunication sector:
       A profit driven data mining approach.
       European Journal of Operational Research, 218(1), 211–229.
.. [2] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B., & Snoeck, M. (2018).
       Profit maximizing logistic model for customer churn prediction using genetic algorithms.
       Swarm and Evolutionary Computation, 40, 116–130.
"""


def fetch_kddcup09_churn(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """Fetch the KDD Cup 2009 / Orange Customer Churn dataset from OpenML (binary classification).

    The goal is to predict customer churn (target: 1 = churn, 0 = no churn).

    The 230 variables are anonymised and include no revenue or customer value, so the
    cost matrix uses the average customer lifetime value of Verbeke et al. (2012) and
    is the same for every customer.

    For additional information about the dataset,
    consult the :ref:`User Guide <kddcup09_churn>`.

    =================   ==============
    Classes                          2
    Churners                      3672
    Non-churners                 46328
    Samples                      50000
    Features                       230
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'clv': array}`` — the customer lifetime value,
        200 for every customer.

    Notes
    -----
    Cost matrix (Verbraken et al. 2013, with the parameters of Verbeke et al. 2012;
    :math:`\\gamma` treated as a fixed scalar):

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_benefit`` :math:`= \\gamma (CLV_i - d \\cdot CLV_i - f) - (1-\\gamma) f`
          - ``fp_cost`` :math:`= d \\cdot CLV_i + f`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= 0`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``incentive_fraction`` (:math:`d`) = 0.05 (an incentive of 10 on a CLV of 200)
    - ``contact_cost`` (:math:`f`) = 1
    - ``accept_rate`` (:math:`\\gamma`) = 0.3 (the mean of the Beta(6, 14)
      distribution used by the expected maximum profit measure)

    To override these defaults, pass the desired values when evaluating the metric::

        metric(dataset.target, y_score, accept_rate=0.5, **dataset.instance_costs)

    References
    ----------
    .. [1] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).
           New insights into churn prediction in the telecommunication sector:
           A profit driven data mining approach.
           European Journal of Operational Research, 218(1), 211–229.
    .. [2] Verbraken, T., Verbeke, W., & Baesens, B. (2013). A novel profit maximizing
           metric for measuring classification performance of customer churn prediction
           models. IEEE Transactions on Knowledge and Data Engineering, 25(5), 961–973.
    .. [3] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B., & Snoeck, M. (2018).
           Profit maximizing logistic model for customer churn prediction using genetic algorithms.
           Swarm and Evolutionary Computation, 40, 116–130.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_kddcup09_churn
        from empulse.metrics import Metric, Cost

        dataset = fetch_kddcup09_churn(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'kddcup09_churn.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_openml_kddcup09_churn_raw,
        download_if_missing=download_if_missing,
        dataset_name='KDD Cup 2009 / Orange Customer Churn dataset',
    )

    feat, target_series = process_kddcup09_churn(raw, backend)
    cost_matrix, instance_costs = churn_retention_cost_matrix(
        np.full(len(target_series), 200.0),
        incentive_fraction=0.05,
        contact_cost=1,
        accept_rate=0.3,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['no churn', 'churn'],
        name='KDD Cup 2009 Orange Churn',
        DESCR=_KDDCUP09_CHURN_DESCRIPTION,
    )


def _fetch_openml_kddcup09_churn_raw() -> dict[str, list[str | None]]:
    """Download KDD Cup 2009 Churn from OpenML and return a flat string dict."""
    try:
        return _fetch_openml(data_id=1112)  # type: ignore[return-value]
    except Exception as exc:
        msg = f'Failed to download the KDD Cup 2009 Churn dataset from OpenML. Original error: {exc}'
        raise OSError(msg) from exc


_CELL2CELL_DESCRIPTION = """\
Cell2Cell Customer Churn
========================

Customer churn data of the US wireless operator Cell2Cell, released for the churn
modelling tournament of the Teradata Center for Customer Relationship Management
at Duke University. It is the "Duke" data used throughout the
profit-driven churn literature (e.g. Verbeke et al. 2012; Höppner et al. 2020;
Maldonado et al. 2020).
The goal is to predict whether a customer will churn.
Target variable: 1 = churned, 0 = active.

The Duke center no longer distributes the data. This loader downloads the
``cell2celltrain.csv`` file from a public GitHub mirror, pinned to a fixed commit,
and drops the 156 customers without a ``MonthlyRevenue``.

The data records monthly revenue but not customer lifetime value, so the CLV of the
churn cost matrix is approximated as a number of months of revenue.

=================   ==============
Classes                          2
Churners                     14641
Non-churners                 36250
Samples                      50891
Features                        56
=================   ==============

Source
------
Teradata Center for Customer Relationship Management, Duke University,
via https://github.com/janebunr/Cell2Cell.

References
----------
.. [1] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).
       New insights into churn prediction in the telecommunication sector:
       A profit driven data mining approach.
       European Journal of Operational Research, 218(1), 211–229.
.. [2] Verbraken, T., Verbeke, W., & Baesens, B. (2013). A novel profit maximizing
       metric for measuring classification performance of customer churn prediction
       models. IEEE Transactions on Knowledge and Data Engineering, 25(5), 961–973.
.. [3] Höppner, S., Stripling, E., Baesens, B., vanden Broucke, S., & Verdonck, T. (2020).
       Profit driven decision trees for churn prediction.
       European Journal of Operational Research, 284(3), 920–933.
.. [4] Maldonado, S., López, J., & Vairetti, C. (2020). Profit-based churn
       prediction based on Minimax Probability Machines. European Journal of
       Operational Research, 284(1), 273–284.
"""


def fetch_cell2cell(
    *,
    backend: Any,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Dataset[Any, Any]:
    """
    Fetch the Cell2Cell Customer Churn dataset (binary classification).

    The goal is to predict whether a customer will churn or not.
    The target variable is whether the customer churned: 1 = churned, 0 = active.

    This is the "Duke" churn data of the Teradata Center for Customer Relationship
    Management tournament. The Duke center no longer distributes it, so it is
    downloaded from a public GitHub mirror pinned to a fixed commit. The 156 customers
    without a ``MonthlyRevenue`` are dropped.

    For additional information about the dataset,
    consult the :ref:`User Guide <cell2cell>`.

    =================   ==============
    Classes                          2
    Churners                     14641
    Non-churners                 36250
    Samples                      50891
    Features                        56
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

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset`
        ``instance_costs`` contains ``{'monthly_revenue': array}`` — each customer's
        average monthly revenue, from which their lifetime value is approximated. The 3
        negative revenues are set to 0: a customer who costs money has no value to retain.

    Notes
    -----
    Cost matrix (Verbraken et al. 2013, :math:`\\gamma` treated as a fixed scalar).
    The data has no customer lifetime value, so it is approximated by
    :math:`CLV_i = m \\cdot MonthlyRevenue_i`:

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_benefit`` :math:`= \\gamma (CLV_i - d \\cdot CLV_i - f) - (1-\\gamma) f`
          - ``fp_cost`` :math:`= d \\cdot CLV_i + f`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= 0`
          - ``tn_cost`` :math:`= 0`

    The cost matrix uses symbolic parameters with the following defaults:

    - ``clv_months`` (:math:`m`) = 12 (months of revenue counted as lifetime value)
    - ``incentive_fraction`` (:math:`d`) = 0.05 (as an incentive of 10 on a CLV of 200
      in Verbeke et al. 2012)
    - ``contact_cost`` (:math:`f`) = 1
    - ``accept_rate`` (:math:`\\gamma`) = 0.3 (the mean of the Beta(6, 14)
      distribution used by the expected maximum profit measure)

    To override these defaults, pass the desired values when evaluating the metric::

        metric(dataset.target, y_score, clv_months=24, **dataset.instance_costs)

    References
    ----------
    .. [1] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).
           New insights into churn prediction in the telecommunication sector:
           A profit driven data mining approach.
           European Journal of Operational Research, 218(1), 211–229.
    .. [2] Verbraken, T., Verbeke, W., & Baesens, B. (2013). A novel profit maximizing
           metric for measuring classification performance of customer churn prediction
           models. IEEE Transactions on Knowledge and Data Engineering, 25(5), 961–973.
    .. [3] Höppner, S., Stripling, E., Baesens, B., vanden Broucke, S., & Verdonck, T. (2020).
           Profit driven decision trees for churn prediction.
           European Journal of Operational Research, 284(3), 920–933.
    .. [4] Maldonado, S., López, J., & Vairetti, C. (2020). Profit-based churn
           prediction based on Minimax Probability Machines. European Journal of
           Operational Research, 284(1), 273–284.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from empulse.datasets import fetch_cell2cell
        from empulse.metrics import Metric, Cost

        dataset = fetch_cell2cell(backend=pd)

        # replace with your own model's predicted probabilities
        y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

        metric = Metric(dataset.cost_matrix, Cost())
        score = metric(dataset.target, y_score, **dataset.instance_costs)
    """
    cache_file = get_data_home(data_home) / 'cell2cell.csv.gz'
    raw = load_or_fetch(
        cache_file,
        _fetch_cell2cell_raw,
        download_if_missing=download_if_missing,
        dataset_name='Cell2Cell Customer Churn dataset',
    )

    feat, target_series, monthly_revenue = process_cell2cell(raw, backend)
    cost_matrix, instance_costs = churn_retention_monthly_cost_matrix(
        # 9 customers have a zero or negative revenue (net credits); they have no value to retain
        np.maximum(monthly_revenue, 0.0),
        clv_months=12,
        incentive_fraction=0.05,
        contact_cost=1,
        accept_rate=0.3,
    )

    return Dataset(
        data=nw.to_native(feat),
        target=nw.to_native(target_series),
        cost_matrix=cost_matrix,
        instance_costs=instance_costs,
        feature_names=feat.columns,
        target_names=['no churn', 'churn'],
        name='Cell2Cell Customer Churn',
        DESCR=_CELL2CELL_DESCRIPTION,
    )


_CELL2CELL_URL = (
    'https://raw.githubusercontent.com/janebunr/Cell2Cell/'
    'd646177d0c23e7dd7cde532ee7a5162c768fe963/Data%20files/cell2celltrain.csv'
)


def _fetch_cell2cell_raw() -> dict[str, list[str | None]]:
    """Download the Cell2Cell training data from a pinned GitHub mirror."""
    try:
        return _fetch_csv_url(_CELL2CELL_URL, timeout=60)
    except Exception as exc:
        msg = f'Failed to download the Cell2Cell dataset. Original error: {exc}'
        raise OSError(msg) from exc
