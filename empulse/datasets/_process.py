"""Per-dataset data-cleaning and feature-engineering functions.

Each ``process_*`` function takes raw data (either a narwhals DataFrame for
local datasets, or a raw ``dict`` + backend for remote datasets) and returns
cleaned, typed narwhals objects ready for cost-matrix computation and
final Dataset assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw

from ._io import _find_column, _sanitize_column_name, _snake_case_column_name

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
        'tp_cost': df['C_TP'].cast(nw.Float64).to_numpy(),
        'fp_cost': df['C_FP'].cast(nw.Float64).to_numpy(),
        'tn_cost': df['C_TN'].cast(nw.Float64).to_numpy(),
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


def process_telco_customer_churn(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the Kaggle Telco Customer Churn raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='churn', dtype Int8)
    monthly_charges : numpy array
    """
    df = nw.from_dict(raw, backend=backend)

    # Filter out empty/blank TotalCharges (dropping 11 rows with missing charges, matching the paper's N=7032)
    df = df.filter(~nw.col('TotalCharges').is_null() & ~nw.col('TotalCharges').is_in(['', ' ']))

    # Target
    target_series = df.select(
        nw
        .when(nw.col('Churn').is_in(['Yes', 'yes', '1', '1.0']))
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.Int8)
        .alias('churn')
    )['churn']

    monthly_charges: FloatNDArray = df['MonthlyCharges'].cast(nw.Float64).to_numpy()

    # Drop customerID and Churn from features
    feature_cols = [c for c in df.columns if c not in {'customerID', 'CustomerID', 'customer_id', 'Churn', 'churn'}]

    numeric_cols = {'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'}
    feat = df.select([
        nw
        .when(nw.col(c).is_in(['?', 'NA', '', ' ']) | nw.col(c).is_null())
        .then(nw.lit(None))
        .otherwise(nw.col(c))
        .cast(nw.Float64)
        .alias(_snake_case_column_name(c))
        if c in numeric_cols
        else nw.col(c).alias(_snake_case_column_name(c))
        for c in feature_cols
    ])

    return feat, target_series, monthly_charges


_DEFAULT_CREDIT_CARD_MAP: dict[str, str] = {
    'x1': 'limit_bal',
    'x2': 'sex',
    'x3': 'education',
    'x4': 'marriage',
    'x5': 'age',
    'x6': 'pay_0',
    'x7': 'pay_2',
    'x8': 'pay_3',
    'x9': 'pay_4',
    'x10': 'pay_5',
    'x11': 'pay_6',
    'x12': 'bill_amt1',
    'x13': 'bill_amt2',
    'x14': 'bill_amt3',
    'x15': 'bill_amt4',
    'x16': 'bill_amt5',
    'x17': 'bill_amt6',
    'x18': 'pay_amt1',
    'x19': 'pay_amt2',
    'x20': 'pay_amt3',
    'x21': 'pay_amt4',
    'x22': 'pay_amt5',
    'x23': 'pay_amt6',
}


def process_default_credit_card_clients(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray, FloatNDArray]:
    """Process the UCI Default of Credit Card Clients raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='default', dtype Int8)
    credit_line : numpy array (LIMIT_BAL)
    target_np : numpy array
    """
    df = nw.from_dict(raw, backend=backend)

    # Map column names if present
    col_map = {}
    for c in df.columns:
        c_low = c.lower().strip()
        if c_low in _DEFAULT_CREDIT_CARD_MAP:
            col_map[c] = _DEFAULT_CREDIT_CARD_MAP[c_low]
        elif c_low in {'y', 'default payment next month'}:
            col_map[c] = 'target'
        else:
            col_map[c] = _sanitize_column_name(c)

    df = df.rename(col_map)

    target_col = 'target' if 'target' in df.columns else _find_column(dict.fromkeys(df.columns), ('y', 'default'))
    target_series = df.select(
        nw
        .when(nw.col(target_col).is_in(['1', '1.0', 'Yes', 'yes']))
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.Int8)
        .alias('default')
    )['default']
    target_np: FloatNDArray = target_series.to_numpy()

    cl_col = (
        'limit_bal'
        if 'limit_bal' in df.columns
        else _find_column(dict.fromkeys(df.columns), ('limit_bal', 'x1', 'LIMIT_BAL'))
    )
    credit_line: FloatNDArray = df[cl_col].cast(nw.Float64).to_numpy()

    drop_cols = {target_col, 'id', 'ID'}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    feat = df.select([nw.col(c).cast(nw.Float64).alias(c) for c in feature_cols])

    return feat, target_series, credit_line, target_np


#: The categorical IEEE-CIS attributes, as listed by the competition host and used by
#: Vanderschueren et al. (2022). Several are numeric codes (``card1``, ``addr1``, ``id_13``, ...).
IEEE_FRAUD_CATEGORICAL_ATTRIBUTES: frozenset[str] = frozenset({
    'ProductCD',
    *(f'card{i}' for i in range(1, 7)),
    'addr1',
    'addr2',
    'P_emaildomain',
    'R_emaildomain',
    *(f'M{i}' for i in range(1, 10)),
    'DeviceType',
    'DeviceInfo',
    *(f'id_{i}' for i in range(12, 39)),
})


def _ieee_column_name(name: str) -> str:
    # `emaildomain` is two words the CamelCase split cannot see
    return _snake_case_column_name(name).replace('emaildomain', 'email_domain')


def process_ieee_fraud_detection(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the Kaggle IEEE-CIS Fraud Detection raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='fraud', dtype Int8)
    amount : numpy array (TransactionAmt)
    """
    df = nw.from_dict(raw, backend=backend)

    target_col = _find_column(dict.fromkeys(df.columns), ('isFraud', 'isfraud', 'class', 'Class'))
    amt_col = _find_column(dict.fromkeys(df.columns), ('TransactionAmt', 'transactionamt', 'amount', 'Amount'))

    target_series = df.select(
        nw
        .when(nw.col(target_col).is_in(['1', '1.0', 'Yes', 'yes']))
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.Int8)
        .alias('fraud')
    )['fraud']

    amount: FloatNDArray = df[amt_col].cast(nw.Float64).to_numpy()

    drop_cols = {target_col, 'TransactionID', 'transactionid', 'TransactionDT', 'transactiondt'}
    # The OpenML copy adds the Kaggle test set's hyphenated identity columns (id-01, ...), which are
    # empty for these training transactions; the underscore versions (id_01, ...) hold the data.
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith('id-')]

    def _missing_to_null(c: str) -> nw.Expr:
        return nw.when(nw.col(c).is_in(['?', ''])).then(nw.lit(None)).otherwise(nw.col(c))

    feat = df.select([
        _missing_to_null(c).alias(_ieee_column_name(c))
        if c in IEEE_FRAUD_CATEGORICAL_ATTRIBUTES
        else _missing_to_null(c).cast(nw.Float64).alias(_ieee_column_name(c))
        for c in feature_cols
    ])
    return feat, target_series, amount


def process_credit_card_fraud(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the Kaggle Credit Card Fraud (ULB MLG) raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='fraud', dtype Int8)
    amount : numpy array (transaction amount)
    """
    df = nw.from_dict(raw, backend=backend)

    target_col = _find_column(dict.fromkeys(df.columns), ('Class', 'class', 'target', 'fraud'))
    amt_col = _find_column(dict.fromkeys(df.columns), ('Amount', 'amount', 'transactionamt'))

    # Filter out zero/negative amount transactions (following Hoppner et al., 2022 and Vanderschueren et al., 2022)
    amt_expr = nw.col(amt_col).cast(nw.Float64)
    df = df.filter(~amt_expr.is_null() & (amt_expr > 0))

    target_series = df.select(
        nw
        .when(nw.col(target_col).is_in(['1', '1.0', 'Yes', 'yes']))
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.Int8)
        .alias('fraud')
    )['fraud']

    amount: FloatNDArray = df[amt_col].cast(nw.Float64).to_numpy()

    # Drop non-predictive sequential Time index and target Class
    drop_cols = {target_col, 'Time', 'time'}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    feat = df.select([nw.col(c).cast(nw.Float64).alias(_sanitize_column_name(c)) for c in feature_cols])
    return feat, target_series, amount


#: The 22 KDD Cup 1998 attributes selected by Petrides & Verbeke (2021) and Vanderschueren et al. (2022),
#: mapped to descriptive snake_case names following the competition's data dictionary.
KDD98_ATTRIBUTES: dict[str, str] = {
    'MAILCODE': 'mail_code',
    'NOEXCH': 'no_exchange',
    'AGE': 'age',
    'HOMEOWNR': 'home_owner',
    'NUMCHLD': 'n_children',
    'INCOME': 'income',
    'GENDER': 'gender',
    'WEALTH1': 'wealth_rating',
    'COLLECT1': 'collectables',
    'CARDPROM': 'n_card_promotions',
    'MAXADATE': 'last_promotion_date',
    'CARDGIFT': 'n_card_gifts',
    'MINRAMNT': 'min_gift_amount',
    'MINRDATE': 'min_gift_date',
    'MAXRAMNT': 'max_gift_amount',
    'MAXRDATE': 'max_gift_date',
    'LASTGIFT': 'last_gift_amount',
    'LASTDATE': 'last_gift_date',
    'FISTDATE': 'first_gift_date',
    'NEXTDATE': 'second_gift_date',
    'TIMELAG': 'months_first_to_second_gift',
    'AVGGIFT': 'avg_gift_amount',
}
#: Flags and YYMM date codes, treated as categorical following Vanderschueren et al. (2022).
KDD98_CATEGORICAL_ATTRIBUTES: frozenset[str] = frozenset({
    'MAILCODE',
    'NOEXCH',
    'HOMEOWNR',
    'GENDER',
    'COLLECT1',
    'MAXADATE',
    'MINRDATE',
    'MAXRDATE',
    'LASTDATE',
    'FISTDATE',
    'NEXTDATE',
})


def process_kdd98(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the KDD Cup 1998 Direct Mailing raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='donated', dtype Int8)
    amount : numpy array (``TARGET_D``, the donation amount; 0 for non-donors)
    """
    df = nw.from_dict(raw, backend=backend)

    target_series = df.select(nw.col('TARGET_B').cast(nw.Float64).cast(nw.Int8).alias('donated'))['donated']
    amount: FloatNDArray = df['TARGET_D'].cast(nw.Float64).to_numpy()

    # The raw files mark missing values with blanks, which survive the CSV cache as whitespace.
    def _blank_to_null(c: str) -> nw.Expr:
        return nw.when(nw.col(c).str.strip_chars() == '').then(nw.lit(None)).otherwise(nw.col(c).str.strip_chars())

    feat = df.select([
        _blank_to_null(c).alias(KDD98_ATTRIBUTES[c])
        if c in KDD98_CATEGORICAL_ATTRIBUTES
        else _blank_to_null(c).cast(nw.Float64).alias(KDD98_ATTRIBUTES[c])
        for c in KDD98_ATTRIBUTES
    ])

    return feat, target_series, amount


def process_vub_credit_scoring(
    df: nw.DataFrame[Any],
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the VUB Credit Scoring dataset.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='default', dtype Int8)
    amounts : numpy array — shifted positive loan amounts for credit line computation
    """
    target_series = df.select(nw.col('Default_45').cast(nw.Float64).cast(nw.Int8).alias('default'))['default']

    loan_amt_s = df['Loan_amount'].cast(nw.Float64)
    min_amt = float(loan_amt_s.to_numpy().min())
    amounts: FloatNDArray = (loan_amt_s - min_amt + 1e-9).to_numpy()

    drop_cols = {
        'ID',
        'id',
        'Test_set1',
        'Test_set2',
        'Test_set3',
        'Default_45',
        'Days_late',
        'Expected_loss',
        'Expected_profit',
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    numeric_cols = {
        'Loan_amount',
        'Monthly_income',
        'Age',
        'Gearing_coefficient',
        'Max_gearing_ratio',
    }

    exprs = []
    for c in feature_cols:
        c_clean = _sanitize_column_name(c)
        if c == 'FICO_Score':
            exprs.append(
                nw
                .when(nw.col(c).is_null() | nw.col(c).is_in(['', ' ']))
                .then(nw.lit(0.0))
                .otherwise(nw.col(c))
                .cast(nw.Float64)
                .alias(c_clean)
            )
        elif c == 'Has_FICO':
            exprs.append(nw.col(c).cast(nw.UInt8).alias(c_clean))
        elif c in numeric_cols:
            exprs.append(nw.col(c).cast(nw.Float64).alias(c_clean))
        else:
            # Nominal / categorical features: V1-V8, Business_channel
            exprs.append(nw.col(c).alias(c_clean))

    feat = df.select(exprs)
    return feat, target_series, amounts


#: HMEQ's abbreviated column names mapped to descriptive snake_case names.
HOME_EQUITY_COLUMNS: dict[str, str] = {
    'loan': 'loan_amount',
    'mortdue': 'mortgage_due',
    'value': 'property_value',
    'reason': 'reason',
    'job': 'job',
    'yoj': 'years_at_job',
    'derog': 'n_derogatory_reports',
    'delinq': 'n_delinquent_credit_lines',
    'clage': 'oldest_credit_line_age',
    'ninq': 'n_recent_credit_inquiries',
    'clno': 'n_credit_lines',
    'debtinc': 'debt_to_income',
}


def process_home_equity(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray, FloatNDArray]:
    """Process the Home Equity (HMEQ) raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='default', dtype Int8)
    loan_amount : numpy array
    target_np : numpy array
    """
    df = nw.from_dict(raw, backend=backend)

    target_col = _find_column(dict.fromkeys(df.columns), ('BAD', 'bad', 'target', 'default'))
    amt_col = _find_column(dict.fromkeys(df.columns), ('LOAN', 'loan', 'amount', 'Amount'))

    target_series = df.select(
        nw
        .when(nw.col(target_col).is_in(['1', '1.0', 'Yes', 'yes']))
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.Int8)
        .alias('default')
    )['default']
    target_np: FloatNDArray = target_series.to_numpy()

    loan_amount: FloatNDArray = df[amt_col].cast(nw.Float64).to_numpy()

    categorical_cols = {'reason', 'job'}
    feature_cols = [c for c in df.columns if c != target_col]

    def _name(c: str) -> str:
        name = _sanitize_column_name(c)
        return HOME_EQUITY_COLUMNS.get(name, name)

    feat = df.select([
        nw
        .when(nw.col(c).is_in(['?', 'NA', '']) | nw.col(c).is_null())
        .then(nw.lit(None))
        .otherwise(nw.col(c))
        .alias(_name(c))
        if _sanitize_column_name(c) in categorical_cols
        else (
            nw
            .when(nw.col(c).is_in(['?', 'NA', '']) | nw.col(c).is_null())
            .then(nw.lit(None))
            .otherwise(nw.col(c))
            .cast(nw.Float64)
            .alias(_name(c))
        )
        for c in feature_cols
    ])

    return feat, target_series, loan_amount, target_np


#: German column names of the South German Credit data mapped to the English names of Grömping (2019).
SOUTH_GERMAN_CREDIT_COLUMNS: dict[str, str] = {
    'laufkont': 'status',
    'laufzeit': 'duration',
    'moral': 'credit_history',
    'verw': 'purpose',
    'hoehe': 'amount',
    'sparkont': 'savings',
    'beszeit': 'employment_duration',
    'rate': 'installment_rate',
    'famges': 'personal_status_sex',
    'buerge': 'other_debtors',
    'wohnzeit': 'present_residence',
    'verm': 'property',
    'alter': 'age',
    'weitkred': 'other_installment_plans',
    'wohn': 'housing',
    'bishkred': 'number_credits',
    'beruf': 'job',
    'pers': 'people_liable',
    'telef': 'telephone',
    'gastarb': 'foreign_worker',
    'kredit': 'credit_risk',
}
_SOUTH_GERMAN_CREDIT_NUMERIC = frozenset({'duration', 'amount', 'age'})


def process_south_german_credit(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray, FloatNDArray]:
    """Process the South German Credit raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values with the original German column names
        (from cache or the UCI archive).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
        Features with the English names of Grömping (2019). Every feature except
        ``duration``, ``amount`` and ``age`` is an integer category code.
    target_series : narwhals Series (name='default', dtype Int8)
        1 = bad credit risk, 0 = good credit risk.
    credit_amount : numpy array
    target_np : numpy array
    """
    df = nw.from_dict(raw, backend=backend).rename(SOUTH_GERMAN_CREDIT_COLUMNS)

    # `kredit` / credit_risk is coded 1 = good, 0 = bad; the positive class is the bad risk.
    target_series = df.select((1 - nw.col('credit_risk').cast(nw.Float64)).cast(nw.Int8).alias('default'))['default']
    target_np: FloatNDArray = target_series.to_numpy()

    credit_amount: FloatNDArray = df['amount'].cast(nw.Float64).to_numpy()

    feat = df.select([
        nw.col(c).cast(nw.Float64) if c in _SOUTH_GERMAN_CREDIT_NUMERIC else nw.col(c).cast(nw.Float64).cast(nw.UInt8)
        for c in df.columns
        if c != 'credit_risk'
    ])

    return feat, target_series, credit_amount, target_np


def process_kddcup09_churn(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any]]:
    """Process the KDD Cup 2009 / Orange Churn raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or OpenML).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='churn', dtype Int8)
    """
    df = nw.from_dict(raw, backend=backend)

    target_col = _find_column(
        dict.fromkeys(df.columns),
        ('CHURN', 'churn', 'target', 'class'),
    )

    target_series = df.select(
        nw
        .when(nw.col(target_col).is_in(['1', '1.0', 'Yes', 'yes', 'True', 'true']))
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.Int8)
        .alias('churn')
    )['churn']

    # Var191 to Var229 are nominal; OpenML types the all-missing Var209 as numeric, which makes no difference
    nominal_set = {f'var{i}' for i in range(191, 230)}

    feature_cols = [c for c in df.columns if c != target_col]
    feat = df.select([
        nw
        .when(nw.col(c).is_in(['?', 'NA', '']) | nw.col(c).is_null())
        .then(nw.lit(None))
        .otherwise(nw.col(c))
        .alias(_sanitize_column_name(c))
        if _sanitize_column_name(c) in nominal_set
        else (
            nw
            .when(nw.col(c).is_in(['?', 'NA', '']) | nw.col(c).is_null())
            .then(nw.lit(None))
            .otherwise(nw.col(c))
            .cast(nw.Float64)
            .alias(_sanitize_column_name(c))
        )
        for c in feature_cols
    ])

    return feat, target_series


def process_cell2cell(
    raw: dict[str, list[Any]],
    backend: Any,
) -> tuple[nw.DataFrame[Any], nw.Series[Any], FloatNDArray]:
    """Process the Cell2Cell Customer Churn raw dict.

    Parameters
    ----------
    raw : dict
        Column-oriented dict of raw string values (from cache or remote).
    backend : module
        Narwhals-compatible dataframe backend.

    Returns
    -------
    feature_df : narwhals DataFrame
    target_series : narwhals Series (name='churn', dtype Int8)
    monthly_revenue : numpy array
    """
    df = nw.from_dict(raw, backend=backend)

    # Filter out rows with missing MonthlyRevenue (156 rows, 0.31%), matching standard practice
    rev_col = _find_column(dict.fromkeys(df.columns), ('MonthlyRevenue', 'monthly_revenue', 'monthlyrevenue'))
    df = df.filter(~nw.col(rev_col).is_null() & ~nw.col(rev_col).is_in(['', 'NA', '?', 'None']))

    target_col = _find_column(dict.fromkeys(df.columns), ('Churn', 'churn', 'target', 'class'))
    target_series = df.select(
        nw
        .when(nw.col(target_col).is_in(['1', '1.0', 'Yes', 'yes', 'True', 'true']))
        .then(nw.lit(1))
        .otherwise(nw.lit(0))
        .cast(nw.Int8)
        .alias('churn')
    )['churn']

    monthly_revenue: FloatNDArray = df[rev_col].cast(nw.Float64).to_numpy()

    # Drop ID and target columns
    id_cols = {'CustomerID', 'customerid', 'customer_id'}
    feature_cols = [c for c in df.columns if c != target_col and c not in id_cols]

    # Categorical features in Cell2Cell
    categorical_cols = {
        _snake_case_column_name(col)
        for col in (
            'ServiceArea',
            'ChildrenInHH',
            'HandsetRefurbished',
            'HandsetWebCapable',
            'TruckOwner',
            'RVOwner',
            'Homeownership',
            'BuysViaMailOrder',
            'RespondsToMailOffers',
            'OptOutMailings',
            'NonUSTravel',
            'OwnsComputer',
            'HasCreditCard',
            'NewCellphoneUser',
            'NotNewCellphoneUser',
            'OwnsMotorcycle',
            'HandsetPrice',
            'MadeCallToRetentionTeam',
            'CreditRating',
            'PrizmCode',
            'Occupation',
            'MaritalStatus',
        )
    }

    feat = df.select([
        nw
        .when(nw.col(c).is_in(['?', 'NA', '', 'None', 'null']) | nw.col(c).is_null())
        .then(nw.lit(None))
        .otherwise(nw.col(c))
        .alias(_snake_case_column_name(c))
        if _snake_case_column_name(c) in categorical_cols
        else (
            nw
            .when(nw.col(c).is_in(['?', 'NA', '', 'None', 'null', 'Unknown']) | nw.col(c).is_null())
            .then(nw.lit(None))
            .otherwise(nw.col(c))
            .cast(nw.Float64)
            .alias(_snake_case_column_name(c))
        )
        for c in feature_cols
    ])

    return feat, target_series, monthly_revenue
