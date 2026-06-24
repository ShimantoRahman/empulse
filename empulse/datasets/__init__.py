from ._base import Dataset, get_data_home
from ._remote import fetch_give_me_some_credit, fetch_iranian_churn
from .datasets import (
    load_churn_tv_subscriptions,
    load_credit_scoring_pakdd,
    load_upsell_bank_telemarketing,
)

__all__ = [
    'Dataset',
    'fetch_give_me_some_credit',
    'fetch_iranian_churn',
    'get_data_home',
    'load_churn_tv_subscriptions',
    'load_credit_scoring_pakdd',
    'load_upsell_bank_telemarketing',
]
