.. _kdd98:

===========================
KDD Cup 1998 Direct Mailing
===========================

Summary
=======

The KDD Cup 1998 data comes from the Paralyzed Veterans of America (PVA), a not-for-profit
organisation that raises money through direct mail [1]_. It covers all 191,779 "lapsed" donors, who
last gave between 13 and 24 months earlier, and who received PVA's June 1997 renewal mailing. The
task is to decide whom to mail: each piece costs $0.68, and only about one in twenty recipients
donates.

The competition split the donors into a learning and a validation set. Empulse combines them, as
Vanderschueren et al. [2]_ do, and keeps the 22 donor attributes selected by Petrides &
Verbeke [3]_ out of the original 479. The donation amount itself is not a feature: it is the
false-negative cost.

=================   ==============
Classes                          2
Donors                        9716
Non-donors                  182063
Samples                     191779
Features                        22
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_kdd98`. The original files are
downloaded from the UCI KDD Archive on first use (about 75 MB) and cached under ``~/empulse_data``
(override with ``$EMPULSE_DATA_HOME`` or the ``data_home`` argument), so later calls work offline.

It returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'amount'``, the donation)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_kdd98

    dataset = fetch_kdd98(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

The data mixes numeric attributes with flags and dates, and has missing values in both, so it needs
imputation and encoding before a linear model can use it:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler, TargetEncoder

    numeric = X.select_dtypes(include=['number']).columns
    categorical = X.select_dtypes(exclude=['number']).columns

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
            ('cat', make_pipeline(
                SimpleImputer(strategy='constant', fill_value='missing'),
                TargetEncoder(),
            ), categorical),
        ])),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    pipeline.fit(X, y, model__amount=dataset.instance_costs['amount'])

Cost Matrix
===========

Mailing a donor costs :math:`c_f`, whether or not they respond. A donor who is not mailed does not
donate, which costs the donation :math:`A_i` they would have made [2]_ [4]_.

.. list-table::

    * -
      - Actual donor :math:`y_i = 1`
      - Actual non-donor :math:`y_i = 0`
    * - Predicted donor :math:`\hat{y}_i = 1`
      - ``tp_cost`` :math:`= c_f`
      - ``fp_cost`` :math:`= c_f`
    * - Predicted non-donor :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= A_i`
      - ``tn_cost`` :math:`= 0`

:math:`A_i` is the donation in response to the June 1997 mailing (``TARGET_D`` in the original
data). It is only known for donors, and is 0 for non-donors, for whom the false-negative cost
never applies.

The mailing cost is a symbolic parameter with the default :math:`c_f = 0.68`, the cost per piece
mailed stated in the competition documentation. It is exposed under the alias ``contact_cost``.
Override it by passing the alias when evaluating the metric:

.. code-block:: python

    y_score = pipeline.predict_proba(X)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_cost = cost(y, y_score, **dataset.instance_costs)
    expensive_mailing = cost(y, y_score, contact_cost=2.0, **dataset.instance_costs)

Data Description
================

Descriptions follow the competition's data dictionary; the original column name is given in
brackets. Dates are ``YYMM`` codes and are kept as categories.

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Feature
     - Description
     - Type
   * - ``mail_code`` (``MAILCODE``)
     - ``B`` if the donor's address is bad
     - categorical
   * - ``no_exchange`` (``NOEXCH``)
     - Whether the donor may not be exchanged in list rentals
     - categorical
   * - ``age`` (``AGE``)
     - Age of the donor
     - numeric
   * - ``home_owner`` (``HOMEOWNR``)
     - ``H`` for a home owner, ``U`` for unknown
     - categorical
   * - ``n_children`` (``NUMCHLD``)
     - Number of children
     - numeric
   * - ``income`` (``INCOME``)
     - Household income band
     - numeric
   * - ``gender`` (``GENDER``)
     - ``M``, ``F``, ``U`` (unknown), ``J`` (joint account), or a few codes the dictionary does
       not define
     - categorical
   * - ``wealth_rating`` (``WEALTH1``)
     - Wealth rating
     - numeric
   * - ``collectables`` (``COLLECT1``)
     - ``Y`` if the donor collects collectables
     - categorical
   * - ``n_card_promotions`` (``CARDPROM``)
     - Lifetime number of card promotions received
     - numeric
   * - ``last_promotion_date`` (``MAXADATE``)
     - Date of the most recent promotion received
     - categorical
   * - ``n_card_gifts`` (``CARDGIFT``)
     - Lifetime number of gifts to card promotions
     - numeric
   * - ``min_gift_amount`` (``MINRAMNT``)
     - Amount of the smallest gift to date, in dollars
     - numeric
   * - ``min_gift_date`` (``MINRDATE``)
     - Date of the smallest gift
     - categorical
   * - ``max_gift_amount`` (``MAXRAMNT``)
     - Amount of the largest gift to date, in dollars
     - numeric
   * - ``max_gift_date`` (``MAXRDATE``)
     - Date of the largest gift
     - categorical
   * - ``last_gift_amount`` (``LASTGIFT``)
     - Amount of the most recent gift, in dollars
     - numeric
   * - ``last_gift_date`` (``LASTDATE``)
     - Date of the most recent gift
     - categorical
   * - ``first_gift_date`` (``FISTDATE``)
     - Date of the first gift
     - categorical
   * - ``second_gift_date`` (``NEXTDATE``)
     - Date of the second gift
     - categorical
   * - ``months_first_to_second_gift`` (``TIMELAG``)
     - Months between the first and second gift
     - numeric
   * - ``avg_gift_amount`` (``AVGGIFT``)
     - Average amount of the gifts to date, in dollars
     - numeric
   * - donated (target)
     - Whether the donor responded to the June 1997 mailing (1 = yes, 0 = no)
     - binary

References
==========

.. [1] KDD Cup 1998 Data. UCI KDD Archive.
       https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html

.. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. *Information Sciences*, 594, 400–415.

.. [3] Petrides, G., & Verbeke, W. (2021). Cost-sensitive ensemble learning: a unifying framework.
       *Data Mining and Knowledge Discovery*, 1–28.

.. [4] Zadrozny, B., Langford, J., & Abe, N. (2003). Cost-sensitive learning by
       cost-proportionate example weighting. In *Third IEEE International Conference on Data
       Mining* (pp. 435–442).
