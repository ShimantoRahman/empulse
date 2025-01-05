from dataclasses import dataclass
from os.path import dirname
from os.path import join
from typing import Generic, TypeVar, overload, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Frame = TypeVar('Frame', pd.DataFrame, NDArray)
Series = TypeVar('Series', pd.Series, NDArray)


@dataclass(frozen=True)
class Dataset(Generic[Frame, Series]):
    """
    Container object for datasets returned by the load functions.

    Attributes
    ----------
    data : :class:`pandas:pandas.DataFrame` or :class:`numpy:numpy.ndarray`
        Features of the dataset.
    target : :class:`pandas:pandas.Series` or :class:`numpy:numpy.ndarray`
        The classification labels.
    tp_cost : :class:`pandas:pandas.Series`, :class:`numpy:numpy.ndarray` or float
        The cost of true positives.
    tn_cost : :class:`pandas:pandas.Series`, :class:`numpy:numpy.ndarray` or float
        The cost of true negatives.
    fp_cost : :class:`pandas:pandas.Series`, :class:`numpy:numpy.ndarray` or float
        The cost of false positives.
    fn_cost : :class:`pandas:pandas.Series`, :class:`numpy:numpy.ndarray` or float
        The cost of false negatives.
    feature_names : :class:`pandas:pandas.Series` or :class:`numpy:numpy.ndarray`
        The meaning of the features.
    target_names : :class:`pandas:pandas.Series` or :class:`numpy:numpy.ndarray`
        The meaning of the labels.
    name : str
        The name of the dataset.
    DESCR : str
        The full description of the dataset
    """

    data: Frame
    target: Series
    tp_cost: Series | float
    tn_cost: Series | float
    fp_cost: Series | float
    fn_cost: Series | float
    feature_names: Series
    target_names: Series
    name: str
    DESCR: str


@overload
def load_churn_tv_subscriptions(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[False] = False,
) -> Dataset[pd.DataFrame, pd.Series]:
    ...


@overload
def load_churn_tv_subscriptions(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[False] = False,
) -> Dataset[NDArray, NDArray]:
    ...


@overload
def load_churn_tv_subscriptions(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[True] = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ...


@overload
def load_churn_tv_subscriptions(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[True] = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    ...


def load_churn_tv_subscriptions(
        as_frame: bool = False,
        return_X_y_costs: bool = False
) -> Dataset | tuple:
    """
    Load the TV Subscription Churn dataset (binary classification).

    The goal is to predict whether a customer will churn or not.
    The target variable is whether the customer churned, 'yes' = 1 and 'no' = 0.

    This dataset is from a TV cable provider containing all 9410 customers active during the first semester of 2014.
    Features names are anonymized to protect the privacy of the customers.

    For additional information about the dataset, consult the :ref:`User Guide <churn_tv_subscriptions>`.

    =================   ==============
    Classes                          2
    Churners                       455
    Non-churners                  8955
    Samples                       9410
    Features                        45
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the output will be a pandas DataFrames or Series instead of numpy arrays.
    return_X_y_costs : bool, default=False
        If True, return (data, target, tp_cost, fp_cost, tn_cost, fn_cost) instead of a Dataset object.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset` or tuple of (data, target, tp_cost, fp_cost, tn_cost, fn_cost)
        Returns a Dataset object if `return_X_y_costs=False` (default), otherwise a tuple.

    Notes
    -----

    Cost matrix

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= \\gamma_i d_i + (1 - \\gamma_i) (CLV_i + c_i)`
          - ``fp_cost`` :math:`= d_i + c_i`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= CLV_i`
          - ``tn_cost`` :math:`= 0`

    with
        - :math:`\\gamma_i` : probability of the customer accepting the retention offer
        - :math:`CLV_i` : customer lifetime value of the retained customer
        - :math:`d_i` : cost of incentive offered to the customer
        - :math:`c_i` : cost of contacting the customer

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           `"A novel cost-sensitive framework for customer churn predictive modeling"
           <http://www.decisionanalyticsjournal.com/content/pdf/s40165-015-0014-6.pdf>`__,
           Decision Analytics, 2:5, 2015.

    Examples
    --------

    .. code-block:: python

        from empulse.datasets import load_churn_tv_subscriptions
        from sklearn.model_selection import train_test_split

        dataset = load_churn_tv_subscriptions()
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            random_state=42
        )
    """
    module_path = dirname(__file__)
    raw_data = pd.read_csv(
        join(module_path, 'data', 'churn_tv_subscriptions.csv.gz'),
        delimiter=',',
        compression='gzip'
    )
    description = open(join(module_path, 'descriptions', 'churn_tv_subscriptions.rst')).read()
    data = raw_data.iloc[:, 1:-5]

    if return_X_y_costs:
        if as_frame:
            return data, raw_data.target, raw_data['C_TP'], raw_data['C_FP'], raw_data['C_TN'], raw_data['C_FN']
        else:
            return (
                data.values,
                raw_data.target.values.astype(np.int8),
                raw_data['C_TP'].values,
                raw_data['C_FP'].values,
                raw_data['C_TN'].values,
                raw_data['C_FN'].values
            )
    else:
        return Dataset(
            data=data.values if not as_frame else data,
            target=raw_data.target.values.astype(np.int8) if not as_frame else raw_data['target'],
            tp_cost=raw_data.C_TP.values if not as_frame else raw_data['C_TP'],
            fp_cost=raw_data.C_FP.values if not as_frame else raw_data['C_FP'],
            tn_cost=raw_data.C_TN.values if not as_frame else raw_data['C_TN'],
            fn_cost=raw_data.C_FN.values if not as_frame else raw_data['C_FN'],
            feature_names=data.columns.values if not as_frame else data.columns,
            target_names=np.array(['no churn', 'churn']) if not as_frame else pd.Series(['no churn', 'churn']),
            name='Churn TV subscriptions',
            DESCR=description,
        )


@overload
def load_upsell_bank_telemarketing(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[False] = False,
        interest_rate: float = 0.02463333,
        term_deposit_fraction: float = 0.25,
        contact_cost: float = 1,
) -> Dataset[pd.DataFrame, pd.Series]:
    ...


@overload
def load_upsell_bank_telemarketing(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[False] = False,
        interest_rate: float = 0.02463333,
        term_deposit_fraction: float = 0.25,
        contact_cost: float = 1,
) -> Dataset[NDArray, NDArray]:
    ...


@overload
def load_upsell_bank_telemarketing(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[True] = False,
        interest_rate: float = 0.02463333,
        term_deposit_fraction: float = 0.25,
        contact_cost: float = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ...


@overload
def load_upsell_bank_telemarketing(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[True] = False,
        interest_rate: float = 0.02463333,
        term_deposit_fraction: float = 0.25,
        contact_cost: float = 1,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    ...


def load_upsell_bank_telemarketing(
        as_frame: bool = False,
        return_X_y_costs: bool = False,
        interest_rate: float = 0.02463333,
        term_deposit_fraction: float = 0.25,
        contact_cost: float = 1,
) -> Dataset | tuple:
    """
    Load the bank telemarketing dataset (binary classification).

    The goal is to predict whether a client will subscribe to a term deposit after being called by the bank.
    The target variable is whether the client subscribed to the term deposit, 'yes' = 1 and 'no' = 0.

    The dataset is related to a direct marketing campaigns (phone calls) of a Portuguese banking institution.
    The marketing campaigns were based on phone calls.
    Often, more than one contact to the same client was required,
    in order to access if the product (bank term deposit) would be or not subscribed.

    Features recorded before the contact event are removed from the original dataset [1]_ to avoid data leakage.
    Only clients with a positive balance are considered, since clients in debt are not eligible for term deposits.

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
    as_frame : bool, default=False
        If True, the output will be a pandas DataFrames or Series instead of numpy arrays.
    return_X_y_costs : bool, default=False
        If True, return (data, target, tp_cost, fp_cost, tn_cost, fn_cost) instead of a Dataset object.
    interest_rate : float, default=0.02463333
        Interest rate of the term deposit.
    term_deposit_fraction : float, default=0.25
        Fraction of the client's balance that is deposited in the term deposit.
    contact_cost : float, default=1
        Cost of contacting the client.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset` or tuple of (data, target, tp_cost, fp_cost, tn_cost, fn_cost)
        Returns a Dataset object if `return_X_y_costs=False` (default), otherwise a tuple.

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
          - ``fn_cost`` :math:`= r \\, d_i \\, b_i`
          - ``tn_cost`` :math:`= 0`

    with
        - :math:`c` : cost of contacting the client
        - :math:`r` : interest rate of the term deposit
        - :math:`d_i` : fraction of the client's balance that is deposited in the term deposit
        - :math:`b_i` : client's balance

    Using default parameters, it is assumed that :math:`c = 1`, :math:`r = 0.02463333`, :math:`d_i = 0.25`
    for all clients.

    References
    ----------
    .. [1] Moro, S., Rita, P., & Cortez, P. (2014).
           Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

    .. [2] S. Moro, R. Laureano and P. Cortez.
           Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.
           In P. Novais et al. (Eds.),
           Proceedings of the European Simulation and Modelling Conference
           - ESM'2011, pp. 117-121, Guimaraes, Portugal, October, 2011. EUROSIS. [bank.zip]

    .. [3] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities"
           <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__,
           in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    Examples
    --------

    .. code-block:: python

        from empulse.datasets import load_upsell_bank_telemarketing
        from sklearn.model_selection import train_test_split

        dataset = load_upsell_bank_telemarketing()
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            random_state=42
        )

    """
    module_path = dirname(__file__)
    raw_data = pd.read_csv(join(module_path, 'data', 'bankmarketing.csv.gz'), delimiter=';', compression='gzip')
    description = open(join(module_path, 'descriptions', 'bankmarketing.rst')).read()

    # only use features pre-contact:
    # 1 - age (numeric)
    # 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur",
    #                        "student","blue-collar","self-employed","retired","technician","services")
    # 3 - marital : marital status (categorical: "married","divorced","single";
    #                               note: "divorced" means divorced or widowed)
    # 4 - education (categorical: "unknown","secondary","primary","tertiary")
    # 5 - default: has credit in default? (binary: "yes","no")
    # 6 - balance: average yearly balance, in euros (numeric)
    # 7 - housing: has housing loan? (binary: "yes","no")
    # 8 - loan: has personal loan? (binary: "yes","no")
    # 15 - previous: number of contacts performed before this campaign and for this client (numeric)
    # 16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

    # Following features exclude because are collected after the contact event
    # # related with the last contact of the current campaign:
    # 9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
    # 10 - day: last contact day of the month (numeric)
    # 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
    # 12 - duration: last contact duration, in seconds (numeric)
    # # other attributes:
    # 13 - campaign: number of contacts performed during this campaign and for this client
    # 14 - pdays: number of days that passed by after the client was last contacted from a
    #       previous campaign (numeric, -1 means client was not previously contacted)

    # Filter if balance>0
    raw_data = raw_data.loc[raw_data['balance'] > 0]
    target = (raw_data.y.values == 'yes').astype(np.int8)
    data = raw_data[['age', 'balance', 'previous', 'job', 'marital',
                     'education', 'default', 'housing', 'loan', 'poutcome']]

    fp_cost = contact_cost
    fn_cost = np.maximum(data['balance'].values * interest_rate * term_deposit_fraction, contact_cost)
    tp_cost = contact_cost
    tn_cost = 0.0

    data.loc[:, 'default'] = data['default'].map({'yes': 1, 'no': 0})
    data.loc[:, 'housing'] = data['housing'].map({'yes': 1, 'no': 0})
    data.loc[:, 'loan'] = data['loan'].map({'yes': 1, 'no': 0})

    data = data.astype({
        'age': np.uint8,
        'balance': np.int32,
        'previous': np.uint8,
        'job': 'category',
        'marital': 'category',
        'education': 'category',
        'default': np.uint8,
        'housing': np.uint8,
        'loan': np.uint8,
        'poutcome': 'category'
    })

    data = data.rename(columns={
        'poutcome': 'previous_outcome',
        'loan': 'has_personal_loan',
        'housing': 'has_housing_loan',
        'default': 'has_credit_in_default',
    })

    if return_X_y_costs:
        if as_frame:
            return (
                data,
                pd.Series(target, name='subscription'),
                tp_cost,
                fp_cost,
                tn_cost,
                pd.Series(fn_cost, name='fn_cost')
            )
        else:
            return (
                data.values,
                target,
                tp_cost,
                fp_cost,
                tn_cost,
                fn_cost
            )
    else:
        target_names = ['no subscription', 'subscription']
        return Dataset(
            data=data.values if not as_frame else data,
            target=target if not as_frame else pd.Series(target, name='subscription'),
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost if not as_frame else pd.Series(fn_cost, name='fn_cost'),
            feature_names=data.columns.values if not as_frame else data.columns,
            target_names=np.array(target_names) if not as_frame else pd.Series(target_names, name='target'),
            name='Bank Telemarketing',
            DESCR=description,
        )


@overload
def load_give_me_some_credit(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[False] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> Dataset[pd.DataFrame, pd.Series]:
    ...


@overload
def load_give_me_some_credit(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[False] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> Dataset[NDArray, NDArray]:
    ...


@overload
def load_give_me_some_credit(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[True] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ...


@overload
def load_give_me_some_credit(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[True] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    ...


def load_give_me_some_credit(
        as_frame: bool = False,
        return_X_y_costs: bool = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> Dataset | tuple:
    """
    Load the "Give Me Some Credit" Kaggle credit scoring competition dataset (binary classification).

    The goal is to predict whether a customer will default on a loan in the next two years.
    The target variable is whether the customer defaulted, 'yes' = 1 and 'no' = 0.

    Only customers with a positive monthly income and a debt ratio less than 1 are considered.

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
    as_frame : bool, default=False
        If True, the output will be a pandas DataFrames or Series instead of numpy arrays.
    return_X_y_costs : bool, default=False
        If True, return (data, target, tp_cost, fp_cost, tn_cost, fn_cost) instead of a Dataset object.
    interest_rate : float, default=0.02463333
        Annual interest rate of the term deposit.
    fund_cost : float, default=0.0294
        Annual cost of funds.
    max_credit_line : float, default=25000
        The maximum amount a client can borrow.
    loss_given_default : float, default=0.75
        The fraction of the loan amount which is lost if the client defaults.
    term_length_months : int, default=24
        The length of the loan term in months.
    loan_to_income_ratio : float, default=3
        The ratio of the loan amount to the client's income.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset` or tuple of (data, target, tp_cost, fp_cost, tn_cost, fn_cost)
        Returns a Dataset object if `return_X_y_costs=False` (default), otherwise a tuple.

    Notes
    -----

    Cost matrix

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= 0`
          - ``fp_cost`` :math:`= r_i + -\\bar{r} \\cdot \\pi_0 + \\bar{Cl} \\cdot L_{gd} \\cdot \\pi_1`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= Cl_i \\cdot L_{gd}`
          - ``tn_cost`` :math:`= 0`

    with
        - :math:`r_i` : loss in profit by rejecting what would have been a good loan
        - :math:`\\bar{r}` : average loss in profit by rejecting what would have been a good loan
        - :math:`\\pi_0` : percentage of defaulters
        - :math:`\\pi_1` : percentage of non-defaulters
        - :math:`Cl_i` : credit line of the client
        - :math:`\\bar{Cl}` : average credit line
        - :math:`L_{gd}` : the fraction of the loan amount which is lost if the client defaults

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications, 2014.

    Examples
    --------

    .. code-block:: python

        from empulse.datasets import load_give_me_some_credit
        from sklearn.model_selection import train_test_split

        dataset = load_give_me_some_credit()
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            random_state=42
        )

    """
    module_path = dirname(__file__)
    raw_data = pd.read_csv(join(module_path, 'data', 'creditscoring1.csv.gz'), delimiter=',', compression='gzip')
    description = open(join(module_path, 'descriptions', 'creditscoring1.rst')).read()

    # Exclude MonthlyIncome = nan or =0 or DebtRatio >1
    raw_data = raw_data.dropna()
    raw_data = raw_data.loc[(raw_data['MonthlyIncome'] > 0)]
    raw_data = raw_data.loc[(raw_data['DebtRatio'] < 1)]

    target = raw_data['SeriousDlqin2yrs'].values.astype(np.int64)

    data = raw_data.drop(['SeriousDlqin2yrs', 'id'], axis=1)

    # Calculate cost_mat (see[1])
    cost_mat_parameters = {'int_r': interest_rate / 12,
                           'int_cf': fund_cost / 12,
                           'cl_max': max_credit_line,
                           'n_term': term_length_months,
                           'k': loan_to_income_ratio,
                           'lgd': loss_given_default}

    pi_1 = target.mean()
    # cost_mat[FP,FN,TP,TN]
    cost_mat = _creditscoring_costmat(data['MonthlyIncome'].values, data['DebtRatio'].values, pi_1, cost_mat_parameters)

    # unroll into separate costs
    fp_cost = cost_mat[:, 0]
    fn_cost = cost_mat[:, 1]

    # normalize feature names
    column_mapping = {
        'RevolvingUtilizationOfUnsecuredLines': 'revolving_utilization',
        'age': 'age',
        'NumberOfTime30-59DaysPastDueNotWorse': 'n_times_late_30_59_days',
        'DebtRatio': 'debt_ratio',
        'MonthlyIncome': 'monthly_income',
        'NumberOfOpenCreditLinesAndLoans': 'n_open_credit_lines',
        'NumberOfTimes90DaysLate': 'n_times_late_over_90_days',
        'NumberRealEstateLoansOrLines': 'n_real_estate_loans',
        'NumberOfTime60-89DaysPastDueNotWorse': 'n_times_late_60_89_days',
        'NumberOfDependents': 'n_dependents'
    }

    data = data.rename(columns=column_mapping)

    new_order = [
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

    # Reorder columns
    data = data.reindex(columns=new_order)

    data = data.astype({
        'monthly_income': np.float64,
        'debt_ratio': np.float64,
        'revolving_utilization': np.float64,
        'age': np.uint8,
        'n_dependents': np.uint8,
        'n_open_credit_lines': np.uint8,
        'n_real_estate_loans': np.uint8,
        'n_times_late_30_59_days': np.uint8,
        'n_times_late_60_89_days': np.uint8,
        'n_times_late_over_90_days': np.uint8,
    })

    if return_X_y_costs:
        if as_frame:
            return (
                data,
                pd.Series(target, name='default'),
                0.0,
                pd.Series(fp_cost, name='fp_cost'),
                0.0,
                pd.Series(fn_cost, name='fn_cost')
            )
        else:
            return (
                data.values,
                target,
                0.0,
                fp_cost,
                0.0,
                fn_cost
            )
    else:
        target_names = ['no default', 'default']
        return Dataset(
            data=data.values if not as_frame else data,
            target=target if not as_frame else pd.Series(target, name='default'),
            tp_cost=0.0,
            fp_cost=fp_cost if not as_frame else pd.Series(fp_cost, name='fp_cost'),
            tn_cost=0.0,
            fn_cost=fn_cost if not as_frame else pd.Series(fn_cost, name='fn_cost'),
            feature_names=data.columns.values if not as_frame else data.columns,
            target_names=np.array(target_names) if not as_frame else pd.Series(target_names, name='target'),
            name='Give Me Some Credit',
            DESCR=description,
        )


@overload
def load_credit_scoring_pakdd(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[False] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> Dataset[pd.DataFrame, pd.Series]:
    ...


@overload
def load_credit_scoring_pakdd(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[False] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> Dataset[NDArray, NDArray]:
    ...


@overload
def load_credit_scoring_pakdd(
        as_frame: Literal[True] = False,
        return_X_y_costs: Literal[True] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ...


@overload
def load_credit_scoring_pakdd(
        as_frame: Literal[False] = False,
        return_X_y_costs: Literal[True] = False,
        interest_rate: float = 0.0479,
        fund_cost: float = 0.0294,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    ...


def load_credit_scoring_pakdd(
        as_frame: bool = False,
        return_X_y_costs: bool = False,
        interest_rate: float = 0.63,
        fund_cost: float = 0.165,
        max_credit_line: float = 25000,
        loss_given_default: float = 0.75,
        term_length_months: int = 24,
        loan_to_income_ratio: float = 3,
):
    """
    Load the credit scoring PAKDD 2009 competition dataset (binary classification).

    The goal is to predict whether a customer will default on a loan in the next two years.
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
    as_frame : bool, default=False
        If True, the output will be a pandas DataFrames or Series instead of numpy arrays.
    return_X_y_costs : bool, default=False
        If True, return (data, target, tp_cost, fp_cost, tn_cost, fn_cost) instead of a Dataset object.
    interest_rate : float, default=0.63
        Annual interest rate of the term deposit.
    fund_cost : float, default=0.165
        Annual cost of funds.
    max_credit_line : float, default=25000
        The maximum amount a client can borrow.
    loss_given_default : float, default=0.75
        The amount of the loan amount which is lost if the client defaults.
    term_length_months : int, default=24
        The length of the loan term in months.
    loan_to_income_ratio : float, default=3
        The ratio of the loan amount to the client's income.

    Returns
    -------
    dataset : :class:`~empulse.datasets.Dataset` or tuple of (data, target, tp_cost, fp_cost, tn_cost, fn_cost)
        Returns a Dataset object if `return_X_y_costs=False` (default), otherwise a tuple.

    Notes
    -----

    Cost matrix

    .. list-table::

        * -
          - Actual positive :math:`y_i = 1`
          - Actual negative :math:`y_i = 0`
        * - Predicted positive :math:`\\hat{y}_i = 1`
          - ``tp_cost`` :math:`= 0`
          - ``fp_cost`` :math:`= r_i + -\\bar{r} \\cdot \\pi_0 + \\bar{Cl} \\cdot L_{gd} \\cdot \\pi_1`
        * - Predicted negative :math:`\\hat{y}_i = 0`
          - ``fn_cost`` :math:`= Cl_i \\cdot L_{gd}`
          - ``tn_cost`` :math:`= 0`

    with
        - :math:`r_i` : loss in profit by rejecting what would have been a good loan
        - :math:`\\bar{r}` : average loss in profit by rejecting what would have been a good loan
        - :math:`\\pi_0` : percentage of defaulters
        - :math:`\\pi_1` : percentage of non-defaulters
        - :math:`Cl_i` : credit line of the client
        - :math:`\\bar{Cl}` : average credit line
        - :math:`L_{gd}` : the fraction of the loan amount which is lost if the client defaults

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Examples
    --------

    .. code-block:: python

        from empulse.datasets import load_credit_scoring_pakdd
        from sklearn.model_selection import train_test_split

        dataset = load_credit_scoring_pakdd()
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            random_state=42
        )

    """
    module_path = dirname(__file__)
    raw_data = pd.read_csv(join(module_path, 'data', 'creditscoring2.csv.gz'), delimiter='\t', compression='gzip')
    description = open(join(module_path, 'descriptions', 'creditscoring2.rst')).read()

    # Exclude TARGET_LABEL_BAD=1 == 'N'
    raw_data = raw_data.loc[raw_data['TARGET_LABEL_BAD=1'] != 'N']

    # Exclude 100<PERSONAL_NET_INCOME<10000
    raw_data = raw_data.loc[(raw_data['PERSONAL_NET_INCOME'].values.astype(np.float64) > 100)]
    raw_data = raw_data.loc[(raw_data['PERSONAL_NET_INCOME'].values.astype(np.float64) < 10000)]

    target = raw_data['TARGET_LABEL_BAD=1'].values.astype(np.int64)
    data = raw_data.drop(['TARGET_LABEL_BAD=1'], axis=1)

    # drop the last column
    data = data.iloc[:, :-1]

    continuous_columns = [
        'MATE_INCOME', 'PERSONAL_NET_INCOME'
    ]
    integer_columns = [
        'ID_SHOP', 'AGE', 'AREA_CODE_RESIDENCIAL_PHONE', 'PAYMENT_DAY', 'SHOP_RANK',
        'MONTHS_IN_RESIDENCE', 'MONTHS_IN_THE_JOB', 'PROFESSION_CODE',
        'QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION'
    ]
    data[continuous_columns] = data[continuous_columns].astype('float64')
    data[integer_columns] = data[integer_columns].astype('int32')

    # Calculate cost_mat (see[1])
    cost_mat_parameters = {
        'int_r': interest_rate / 12,
        'int_cf': fund_cost / 12,
        'cl_max': max_credit_line * 0.33,
        'n_term': term_length_months,
        'k': loan_to_income_ratio,
        'lgd': loss_given_default
    }

    n_samples = data.shape[0]
    pi_1 = target.mean()
    monthly_income = data['PERSONAL_NET_INCOME'].values * 0.33
    cost_mat = _creditscoring_costmat(monthly_income, np.zeros(n_samples), pi_1, cost_mat_parameters)

    # unroll into separate costs
    fp_cost = cost_mat[:, 0]
    fn_cost = cost_mat[:, 1]

    # convert all columns which start with flag to integers
    flag_columns = [col for col in data.columns if col.startswith('FLAG')]
    data[flag_columns] = (data[flag_columns] == 'Y').astype(np.int8)

    # normalize feature names
    data.columns = data.columns.str.lower().str.replace('#', '').str.replace('quant', 'n')
    data.columns = data.columns.str.replace('_in_the_application', '').str.replace('residencial', 'residential')
    data.loc[:, 'sex'] = data['sex'].map({'M': 1, 'F': 0})
    # fill in single  missing value as male
    data.loc[:, 'sex'] = data.sex.fillna('1')

    column_mapping = {
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
    data = data.rename(columns=column_mapping)

    # remap values of matiral_status to more readable values
    data.loc[:, 'marital_status'] = data['marital_status'].map({
        'S': 'single',
        'M': 'married',
        'D': 'divorced',
        'W': 'widowed',
        'O': 'other'
    })
    # remap values of residence_type to more readable values
    data.loc[:, 'residence_type'] = data['residence_type'].map({
        'P': 'owned',
        'A': 'rented',
        'C': 'parents',
        'O': 'other'
    })

    # Desired column order
    new_order = [
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
        'has_card_insurance',
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

    # Reorder columns
    data = data.reindex(columns=new_order)

    data = data.astype({
        'age': np.uint8,
        'personal_net_income': np.float32,
        'partner_income': np.float32,
        'months_in_residence': np.uint8,
        'months_in_the_job': np.uint8,
        'payment_day': np.uint8,
        'n_banking_accounts': np.uint8,
        'n_additional_cards': np.uint8,
        'is_male': np.uint8,
        'has_residential_phone': np.uint8,
        'has_mobile_phone': np.uint8,
        'has_contact_phone': np.uint8,
        'has_same_postal_address': np.uint8,
        'has_other_card': np.uint8,
        'has_card_insurance': np.uint8,
        'lives_in_work_town': np.uint8,
        'lives_in_work_state': np.uint8,
        'filled_in_mothers_name': np.uint8,
        'filled_in_fathers_name': np.uint8,
        'shop_rank': 'category',
        'marital_status': 'category',
        'residence_type': 'category',
        'area_code_residential_phone': 'category',
        'shop_code': 'category',
        'application_booth_code': 'category',
        'profession_code': 'category',
    })

    # remove flag_card_insurance_option
    data = data.drop(['has_card_insurance'], axis=1)

    if return_X_y_costs:
        if as_frame:
            return (
                data,
                pd.Series(target, name='default'),
                0.0,
                pd.Series(fp_cost, name='fp_cost'),
                0.0,
                pd.Series(fn_cost, name='fn_cost')
            )
        else:
            return (
                data.values,
                target,
                0.0,
                fp_cost,
                0.0,
                fn_cost
            )
    else:
        target_names = ['no default', 'default']
        return Dataset(
            data=data.values if not as_frame else data,
            target=target if not as_frame else pd.Series(target, name='default'),
            tp_cost=0.0,
            fp_cost=fp_cost if not as_frame else pd.Series(fp_cost, name='fp_cost'),
            tn_cost=0.0,
            fn_cost=fn_cost if not as_frame else pd.Series(fn_cost, name='fn_cost'),
            feature_names=data.columns.values if not as_frame else data.columns,
            target_names=np.array(target_names) if not as_frame else pd.Series(target_names, name='target'),
            name='Credit Scoring PAKDD 2009',
            DESCR=description,
        )


def _creditscoring_costmat(income, debt, pi_1, cost_mat_parameters):
    """ Private function to calculate the cost matrix of credit scoring models.

    Parameters
    ----------
    income : array of shape = [n_samples]
        Monthly income of each example

    debt : array of shape = [n_samples]
        Debt ratio each example

    pi_1 : float
        Percentage of positives in the training set

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Returns
    -------
    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.
    """

    def calculate_a(cl_i, int_, n_term):
        return cl_i * ((int_ * (1 + int_) ** n_term) / ((1 + int_) ** n_term - 1))

    def calculate_pv(a, int_, n_term):
        return a / int_ * (1 - 1 / (1 + int_) ** n_term)

    # Calculate credit line Cl
    def calculate_cl(k, inc_i, cl_max, debt_i, int_r, n_term):
        cl_k = k * inc_i
        A = calculate_a(cl_k, int_r, n_term)
        Cl_debt = calculate_pv(inc_i * min(A / inc_i, 1 - debt_i), int_r, n_term)
        return min(cl_k, cl_max, Cl_debt)

    # calculate costs
    def calculate_cost_fn(cl_i, lgd):
        return cl_i * lgd

    def calculate_cost_fp(cl_i, int_r, n_term, int_cf, pi_1, lgd, cl_avg):
        a = calculate_a(cl_i, int_r, n_term)
        pv = calculate_pv(a, int_cf, n_term)
        r = pv - cl_i
        r_avg = calculate_pv(calculate_a(cl_avg, int_r, n_term), int_cf, n_term) - cl_avg
        cost_fp = r - (1 - pi_1) * r_avg + pi_1 * calculate_cost_fn(cl_avg, lgd)
        return max(0, cost_fp)

    v_calculate_cost_fp = np.vectorize(calculate_cost_fp)
    v_calculate_cost_fn = np.vectorize(calculate_cost_fn)

    v_calculate_cl = np.vectorize(calculate_cl)

    # Parameters
    k = cost_mat_parameters['k']
    int_r = cost_mat_parameters['int_r']
    n_term = cost_mat_parameters['n_term']
    int_cf = cost_mat_parameters['int_cf']
    lgd = cost_mat_parameters['lgd']
    cl_max = cost_mat_parameters['cl_max']

    cl = v_calculate_cl(k, income, cl_max, debt, int_r, n_term)
    cl_avg = cl.mean()

    n_samples = income.shape[0]
    cost_mat = np.zeros((n_samples, 4))  # cost_mat[FP,FN,TP,TN]
    cost_mat[:, 0] = v_calculate_cost_fp(cl, int_r, n_term, int_cf, pi_1, lgd, cl_avg)
    cost_mat[:, 1] = v_calculate_cost_fn(cl, lgd)
    cost_mat[:, 2] = 0.0
    cost_mat[:, 3] = 0.0

    return cost_mat
