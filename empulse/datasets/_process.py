"""Per-dataset data-cleaning and feature-engineering functions.

Each ``process_*`` function takes raw data (either a narwhals DataFrame for
local datasets, or a raw ``dict`` + backend for remote datasets) and returns
cleaned, typed narwhals objects ready for cost-matrix computation and
final Dataset assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw

from ._io import _find_column, _sanitize_column_name

if TYPE_CHECKING:
    from .._types import FloatNDArray


def process_churn_tv(
    df: nw.DataFrame[Any],
) -> tuple[nw.DataFrame[Any], nw.Series[Any], dict[str, FloatNDArray]]:
    """Process the Churn TV Subscriptions dataset.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series  (name='churn', dtype Int8)
    instance_costs : dict  — pre-computed per-instance costs from the CSV
    """
    all_cols = df.columns
    # Layout: [index_col, *features, target, C_TP, C_FP, C_TN, C_FN]
    feature_cols = all_cols[1:-5]

    instance_costs: dict[str, FloatNDArray] = {
        'tp_benefit': df['C_TP'].cast(nw.Float64).to_numpy(),
        'fp_cost': df['C_FP'].cast(nw.Float64).to_numpy(),
        'tn_benefit': df['C_TN'].cast(nw.Float64).to_numpy(),
        'fn_cost': df['C_FN'].cast(nw.Float64).to_numpy(),
    }
    feature_df = df.select(nw.col(feature_cols).cast(nw.Float64))
    target_series = df.select(nw.col('target').cast(nw.Float64).cast(nw.Int8).alias('churn'))['churn']
    return feature_df, target_series, instance_costs


def process_bank_telemarketing(
    df: nw.DataFrame[Any],
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the Bank Telemarketing dataset.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series  (name='subscription', dtype Int8)
    balance : numpy array  — per-client balance for cost computation
    """
    df = df.filter(nw.col('balance').cast(nw.Int32) > 0)

    balance: FloatNDArray = df['balance'].cast(nw.Float64).to_numpy()
    target_series = df.select((nw.col('y') == 'yes').cast(nw.Int8).alias('subscription'))['subscription']

    feature_df = df.select(
        nw.col('age').cast(nw.UInt8),
        nw.col('balance').cast(nw.Int32),
        nw.col('previous').cast(nw.Float64).cast(nw.UInt16),
        nw.col('job'),
        nw.col('marital'),
        nw.col('education'),
        nw
        .when(nw.col('default') == 'yes')
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.UInt8)
        .alias('has_credit_in_default'),
        nw
        .when(nw.col('housing') == 'yes')
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.UInt8)
        .alias('has_housing_loan'),
        nw.when(nw.col('loan') == 'yes').then(nw.lit(1)).otherwise(nw.lit(0)).cast(nw.UInt8).alias('has_personal_loan'),
        nw.col('poutcome').alias('previous_outcome'),
    )
    return feature_df, target_series, balance


def process_credit_scoring_pakdd(
    df: nw.DataFrame[Any],
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the PAKDD 2009 credit scoring dataset.

    Expects a narwhals DataFrame where all columns are strings (built via
    ``_read_csv_gz`` → ``nw.from_dict``).

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series  (name='default', dtype Int64)
    monthly_income : numpy array  — income scaled by 0.33 for credit line computation
    """
    target_col = 'TARGET_LABEL_BAD=1'
    income_col = 'PERSONAL_NET_INCOME'

    # Replace sentinel strings with null, then cast and filter
    df = df.with_columns(
        nw
        .when(nw.col(income_col).is_in(['N', '']))
        .then(nw.lit(None))
        .otherwise(nw.col(income_col))
        .cast(nw.Float64)
        .alias(income_col)
    ).filter(~nw.col(target_col).is_null() & (nw.col(income_col) > 100.0) & (nw.col(income_col) < 10000.0))

    target_series = df.select(nw.col(target_col).cast(nw.Int64).alias('default'))['default']

    # Feature columns: all except the target and the last (trailing) column
    all_cols = df.columns
    feature_col_names = [c for c in all_cols if c != target_col][:-1]
    feat = df.select(feature_col_names)

    # 1. Encode FLAG columns (Y → 1, anything else → 0)
    flag_cols = [c for c in feat.columns if c.upper().startswith('FLAG')]
    feat = feat.with_columns(*[
        nw.when(nw.col(col) == 'Y').then(nw.lit(1)).otherwise(nw.lit(0)).cast(nw.UInt8).alias(col) for col in flag_cols
    ])

    # 2. Normalise column names
    def _norm(name: str) -> str:
        return (
            name
            .lower()
            .replace('#', '')
            .replace('quant', 'n')
            .replace('_in_the_application', '')
            .replace('residencial', 'residential')
        )

    feat = feat.rename({col: _norm(col) for col in feat.columns})

    # 3. Value mappings
    feat = feat.with_columns(
        nw
        .when(nw.col('sex') == 'M')
        .then(nw.lit(1))
        .when(nw.col('sex') == 'F')
        .then(nw.lit(0))
        .otherwise(nw.lit(1))
        .cast(nw.UInt8)
        .alias('sex'),
        nw
        .when(nw.col('marital_status') == 'S')
        .then(nw.lit('single'))
        .when(nw.col('marital_status') == 'M')
        .then(nw.lit('married'))
        .when(nw.col('marital_status') == 'D')
        .then(nw.lit('divorced'))
        .when(nw.col('marital_status') == 'W')
        .then(nw.lit('widowed'))
        .otherwise(nw.lit('other'))
        .alias('marital_status'),
        nw
        .when(nw.col('residence_type') == 'P')
        .then(nw.lit('owned'))
        .when(nw.col('residence_type') == 'A')
        .then(nw.lit('rented'))
        .when(nw.col('residence_type') == 'C')
        .then(nw.lit('parents'))
        .otherwise(nw.lit('other'))
        .alias('residence_type'),
    )

    # 4. Rename to descriptive names
    pakdd_renames = {
        'flag_residence_town_eq_working_town': 'lives_in_work_town',
        'flag_residence_state_eq_working_state': 'lives_in_work_state',
        'flag_residential_address_eq_postal_address': 'has_same_postal_address',
        'flag_residential_phone': 'has_residential_phone',
        'sex': 'is_male',
        'flag_mothers_name': 'filled_in_mothers_name',
        'flag_fathers_name': 'filled_in_fathers_name',
        'mate_income': 'partner_income',
        'flag_other_card': 'has_other_card',
        'flag_mobile_phone': 'has_mobile_phone',
        'flag_contact_phone': 'has_contact_phone',
        'cod_application_booth': 'application_booth_code',
        'flag_card_insurance_option': 'has_card_insurance',
        'id_shop': 'shop_code',
    }
    feat = feat.rename({k: v for k, v in pakdd_renames.items() if k in feat.columns})

    # 5. Canonical column order
    pakdd_order = [
        'age',
        'personal_net_income',
        'partner_income',
        'months_in_residence',
        'months_in_the_job',
        'payment_day',
        'n_banking_accounts',
        'n_additional_cards',
        'is_male',
        'has_residential_phone',
        'has_mobile_phone',
        'has_contact_phone',
        'has_same_postal_address',
        'has_other_card',
        'lives_in_work_town',
        'lives_in_work_state',
        'filled_in_mothers_name',
        'filled_in_fathers_name',
        'shop_rank',
        'marital_status',
        'residence_type',
        'area_code_residential_phone',
        'shop_code',
        'application_booth_code',
        'profession_code',
    ]
    feat = feat.select([col for col in pakdd_order if col in feat.columns])

    # 6. Type casting
    feat = feat.with_columns(
        nw.col('age').cast(nw.UInt8),
        nw.col('personal_net_income').cast(nw.Float32),
        nw.col('partner_income').cast(nw.Float32),
        nw.col('months_in_residence').cast(nw.UInt16),
        nw.col('months_in_the_job').cast(nw.UInt16),
        nw.col('payment_day').cast(nw.UInt8),
        nw.col('n_banking_accounts').cast(nw.UInt8),
        nw.col('n_additional_cards').cast(nw.UInt8),
    )

    monthly_income: FloatNDArray = feat['personal_net_income'].cast(nw.Float64).to_numpy() * 0.33
    return feat, target_series, monthly_income


def process_iranian_churn(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the Iranian Churn raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or UCI API).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series  (dtype Int8)
    clv : numpy array
    """
    clv_col = _find_column(raw, ('Customer Value', 'CustomerValue', 'customer_value'))
    target_col = _find_column(raw, ('Churn', 'churn', 'Class', 'class'))
    feature_cols = [c for c in raw if c not in {clv_col, target_col}]

    # Detect yes/no columns from raw strings BEFORE building the frame
    yes_no_norm: set[str] = {
        _sanitize_column_name(col)
        for col in feature_cols
        if {str(v).strip().lower() for v in raw[col] if v is not None and str(v).strip()} <= {'yes', 'no'}
    }

    # Build feature dict with sanitised column names and string values
    feat_dict: dict[str, list[str | None]] = {
        _sanitize_column_name(col): [str(v) if v is not None else None for v in raw[col]] for col in feature_cols
    }
    df = nw.from_dict(feat_dict, backend=backend)

    numeric_cols = [c for c in df.columns if c not in yes_no_norm]
    df = df.with_columns(
        *[
            nw.when(nw.col(c) == 'yes').then(nw.lit(1)).otherwise(nw.lit(0)).cast(nw.UInt8).alias(c)
            for c in yes_no_norm
            if c in df.columns
        ],
        *[nw.col(c).cast(nw.Float64).alias(c) for c in numeric_cols],
    )

    clv_vals: list[str | None] = [str(v) if v is not None else None for v in raw[clv_col]]
    clv: FloatNDArray = nw.from_dict({clv_col: clv_vals}, backend=backend)[clv_col].cast(nw.Float64).to_numpy()

    target_vals: list[str | None] = [str(v) if v is not None else None for v in raw[target_col]]
    target_series = nw.from_dict({target_col: target_vals}, backend=backend)[target_col].cast(nw.Int8)

    return df, target_series, clv


_GIVE_ME_SOME_CREDIT_COL_MAP: dict[str, str] = {
    'SeriousDlqin2yrs': 'target',
    'seriousdlqin2yrs': 'target',
    'FinancialDistressNextTwoYears': 'target',
    'financialdistressnexttwoyears': 'target',
    'RevolvingUtilizationOfUnsecuredLines': 'revolving_utilization',
    'revolvingutilizationofunsecuredlines': 'revolving_utilization',
    'age': 'age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'n_times_late_30_59_days',
    'numberoftimes30-59dayspastduenotworse': 'n_times_late_30_59_days',
    'NumberOfTimes90DaysLate': 'n_times_late_over_90_days',
    'numberoftimes90dayslate': 'n_times_late_over_90_days',
    'DebtRatio': 'debt_ratio',
    'debtratio': 'debt_ratio',
    'MonthlyIncome': 'monthly_income',
    'monthlyincome': 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans': 'n_open_credit_lines',
    'numberofopencrditlinesandloans': 'n_open_credit_lines',
    'NumberRealEstateLoansOrLines': 'n_real_estate_loans',
    'numberrealestatelloansandlines': 'n_real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'n_times_late_60_89_days',
    'numberoftimes60-89dayspastduenotworse': 'n_times_late_60_89_days',
    'NumberOfDependents': 'n_dependents',
    'numberofdependents': 'n_dependents',
}

_GIVE_ME_SOME_CREDIT_FEATURE_ORDER = [
    'monthly_income',
    'debt_ratio',
    'revolving_utilization',
    'age',
    'n_dependents',
    'n_open_credit_lines',
    'n_real_estate_loans',
    'n_times_late_30_59_days',
    'n_times_late_60_89_days',
    'n_times_late_over_90_days',
]


def process_give_me_some_credit(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray, FloatNDArray, FloatNDArray]:
    """Process the Give Me Some Credit raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series  (name='default', dtype Int64)
    monthly_income_np : numpy array
    debt_ratio_np : numpy array
    target_np : numpy array
    """
    # Normalise column names
    renamed: dict[str, list[Any]] = {}
    for col, vals in raw.items():
        stripped = col.strip()
        new_name = _GIVE_ME_SOME_CREDIT_COL_MAP.get(
            stripped, _GIVE_ME_SOME_CREDIT_COL_MAP.get(stripped.lower(), stripped)
        )
        renamed[new_name] = vals

    target_col = _find_column(
        renamed,
        (
            'target',
            'SeriousDlqin2yrs',
            'FinancialDistressNextTwoYears',
            'financialdistressnexttwoyears',
            'class',
            'Class',
        ),
        fallback_prefix='target',
    )

    df = nw.from_dict(renamed, backend=backend)

    # Handle missing values and cast income/debt
    df = df.with_columns(
        nw
        .when(nw.col('monthly_income').is_in(['?', 'NA', '']))
        .then(nw.lit(None))
        .otherwise(nw.col('monthly_income'))
        .cast(nw.Float64)
        .alias('monthly_income'),
        nw
        .when(nw.col('debt_ratio').is_in(['?', 'NA', '']))
        .then(nw.lit(None))
        .otherwise(nw.col('debt_ratio'))
        .cast(nw.Float64)
        .alias('debt_ratio'),
    ).filter(
        ~nw.col('monthly_income').is_null()
        & (nw.col('monthly_income') > 0)
        & ~nw.col('debt_ratio').is_null()
        & (nw.col('debt_ratio') < 1)
    )

    # Encode target (handles numeric strings and Yes/No variants)
    target_as_int = (
        nw
        .when(nw.col(target_col).is_in(['1', '1.0', 'Yes', 'yes']))
        .then(nw.lit(1))
        .when(nw.col(target_col).is_in(['0', '0.0', 'No', 'no']))
        .then(nw.lit(0))
        .otherwise(nw.lit(None))
    )
    target_series = df.select(target_as_int.cast(nw.Int64).alias('default'))['default']

    monthly_income_np: FloatNDArray = df['monthly_income'].cast(nw.Float64).to_numpy()
    debt_ratio_np: FloatNDArray = df['debt_ratio'].cast(nw.Float64).to_numpy()
    target_np: FloatNDArray = target_series.to_numpy()

    # Select and type-cast feature columns
    available_features = [c for c in _GIVE_ME_SOME_CREDIT_FEATURE_ORDER if c in df.columns]
    float_cols = {'monthly_income', 'debt_ratio', 'revolving_utilization'}
    feat = df.select(*[
        nw.col(c).cast(nw.Float64).alias(c)
        if c in float_cols
        else (
            nw
            .when(nw.col(c).is_in(['?', 'NA', '']))
            .then(nw.lit(None))
            .otherwise(nw.col(c))
            .cast(nw.Float64)
            .cast(nw.UInt8)
            .alias(c)
        )
        for c in available_features
    ])

    return feat, target_series, monthly_income_np, debt_ratio_np, target_np
