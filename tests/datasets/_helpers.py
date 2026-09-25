"""
Helpers shared by the dataset tests.

``BACKENDS`` was previously defined in four modules, and the cell2cell fixture and the Bahnsen
false positive cost were only reachable from the module that happened to define them.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

BACKENDS = [
    pytest.param(pd, id='pandas'),
    pytest.param(pl, id='polars'),
]


def bahnsen_fp_cost(cl, pi_1, *, interest_rate=0.0479, fund_cost=0.0294, lgd=0.75, n=24):
    """Independent restatement of the Bahnsen et al. (2014) false positive cost."""
    int_r, int_cf = interest_rate / 12, fund_cost / 12

    def lost_profit(credit_line):
        installment = credit_line * int_r * (1 + int_r) ** n / ((1 + int_r) ** n - 1)
        present_value = installment / int_cf * (1 - (1 + int_cf) ** -n)
        return present_value - credit_line

    cl_avg = np.mean(cl)
    return np.maximum(0, lost_profit(cl) - (1 - pi_1) * lost_profit(cl_avg) + pi_1 * cl_avg * lgd)


# A four-row cell2cell extract in the raw string form the fetcher caches; the third row has no
# monthly revenue, so processing drops it.
MOCK_RAW_CELL2CELL = {
    'CustomerID': ['3000001', '3000002', '3000003', '3000004'],
    'Churn': ['Yes', 'No', 'Yes', 'No'],
    'MonthlyRevenue': ['50.0', '100.0', '', '75.5'],
    'MonthlyMinutes': ['200', '500', '150', '300'],
    'TotalRecurringCharge': ['40.0', '80.0', '30.0', '60.0'],
    'DirectorAssistedCalls': ['0.5', '1.0', '0.0', '0.25'],
    'OverageMinutes': ['10', '0', '5', '20'],
    'RoamingCalls': ['0', '2', '0', '1'],
    'PercChangeMinutes': ['-10', '5', '0', '15'],
    'PercChangeRevenues': ['-5', '10', '0', '8'],
    'DroppedCalls': ['1.0', '0.5', '2.0', '0.0'],
    'BlockedCalls': ['0.0', '1.0', '0.0', '0.5'],
    'UnansweredCalls': ['5.0', '10.0', '2.0', '3.0'],
    'CustomerCareCalls': ['0', '1', '0', '2'],
    'ThreewayCalls': ['0', '0', '1', '0'],
    'ReceivedCalls': ['50', '100', '30', '80'],
    'OutboundCalls': ['20', '40', '10', '30'],
    'InboundCalls': ['5', '10', '2', '8'],
    'PeakCallsInOut': ['30', '60', '20', '50'],
    'OffPeakCallsInOut': ['20', '40', '10', '30'],
    'DroppedBlockedCalls': ['1.0', '1.5', '2.0', '0.5'],
    'CallForwardingCalls': ['0', '0', '0', '0'],
    'CallWaitingCalls': ['1', '2', '0', '1'],
    'MonthsInService': ['12', '24', '6', '36'],
    'UniqueSubs': ['1', '2', '1', '1'],
    'ActiveSubs': ['1', '1', '1', '1'],
    'ServiceArea': ['SEAPOR503', 'OKCTUL918', 'SEAPOR503', 'MILMIL414'],
    'Handsets': ['1', '2', '1', '3'],
    'HandsetModels': ['1', '1', '1', '2'],
    'CurrentEquipmentDays': ['300', '600', '150', '900'],
    'AgeHH1': ['45', '50', '30', '60'],
    'AgeHH2': ['0', '48', '0', '55'],
    'ChildrenInHH': ['No', 'Yes', 'No', 'No'],
    'HandsetRefurbished': ['No', 'No', 'Yes', 'No'],
    'HandsetWebCapable': ['Yes', 'Yes', 'Yes', 'Yes'],
    'TruckOwner': ['No', 'Yes', 'No', 'No'],
    'RVOwner': ['No', 'No', 'No', 'Yes'],
    'Homeownership': ['Known', 'Known', 'Unknown', 'Known'],
    'BuysViaMailOrder': ['Yes', 'Yes', 'No', 'Yes'],
    'RespondsToMailOffers': ['Yes', 'No', 'No', 'Yes'],
    'OptOutMailings': ['No', 'No', 'No', 'No'],
    'NonUSTravel': ['No', 'No', 'No', 'Yes'],
    'OwnsComputer': ['Yes', 'Yes', 'No', 'Yes'],
    'HasCreditCard': ['Yes', 'Yes', 'No', 'Yes'],
    'RetentionCalls': ['0', '1', '0', '0'],
    'RetentionOffersAccepted': ['0', '1', '0', '0'],
    'NewCellphoneUser': ['No', 'No', 'Yes', 'No'],
    'NotNewCellphoneUser': ['Yes', 'Yes', 'No', 'Yes'],
    'ReferralsMadeBySubscriber': ['0', '1', '0', '0'],
    'IncomeGroup': ['4', '6', '2', '8'],
    'OwnsMotorcycle': ['No', 'No', 'No', 'No'],
    'AdjustmentsToCreditRating': ['0', '0', '0', '1'],
    'HandsetPrice': ['30', '100', 'Unknown', '150'],
    'MadeCallToRetentionTeam': ['No', 'Yes', 'No', 'No'],
    'CreditRating': ['1-Highest', '2-High', '3-Good', '1-Highest'],
    'PrizmCode': ['Suburban', 'Town', 'Rural', 'Other'],
    'Occupation': ['Professional', 'Professional', 'Other', 'Crafts'],
    'MaritalStatus': ['No', 'Yes', 'Unknown', 'Yes'],
}
