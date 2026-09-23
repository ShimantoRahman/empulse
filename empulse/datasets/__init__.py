from ._base import Dataset, get_data_home
from ._remote import (
    fetch_cell2cell,
    fetch_credit_card_fraud,
    fetch_default_credit_card_clients,
    fetch_give_me_some_credit,
    fetch_home_equity,
    fetch_ieee_fraud_detection,
    fetch_iranian_churn,
    fetch_kdd98,
    fetch_kddcup09_churn,
    fetch_south_german_credit,
    fetch_telco_customer_churn,
)
from .datasets import (
    load_churn_tv_subscriptions,
    load_credit_scoring_pakdd,
    load_upsell_bank_telemarketing,
    load_vub_credit_scoring,
)

__all__ = [
    'Dataset',
    'fetch_cell2cell',
    'fetch_credit_card_fraud',
    'fetch_default_credit_card_clients',
    'fetch_give_me_some_credit',
    'fetch_home_equity',
    'fetch_ieee_fraud_detection',
    'fetch_iranian_churn',
    'fetch_kdd98',
    'fetch_kddcup09_churn',
    'fetch_south_german_credit',
    'fetch_telco_customer_churn',
    'get_data_home',
    'load_churn_tv_subscriptions',
    'load_credit_scoring_pakdd',
    'load_upsell_bank_telemarketing',
    'load_vub_credit_scoring',
]
