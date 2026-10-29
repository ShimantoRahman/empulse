"""Run the code examples and benchmarks from the Empulse JSS paper.

Run with: python replication_jss.py
Adjust the EMP dimension limit and training settings in main().
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threadpoolctl import threadpool_limits

from jss_benchmark import emp_integration, figures, gradient, timing
from jss_benchmark.data import N_GRID


def run_examples() -> None:
    """Execute the paper's examples and Section 9 case study in manuscript order."""
    import warnings
    import pandas as pd

    np.random.seed(42)
    np.set_printoptions(precision=6, threshold=20)
    # ## Section 5: Defining costs: The CostMatrix class
    #
    # ### Section 5.1 Symbolic foundations and the builder API
    #
    # The customer churn cost matrix: a contacted churner accepts the retention offer with probability
    # $\gamma$, which costs the incentive $\delta \cdot clv$ and the contact cost $f$; a contacted
    # non-churner costs the same.

    print('\n' + 'Section 5: Defining costs: The CostMatrix class')

    import sympy

    from empulse.metrics import CostMatrix

    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * ((1 - delta) * clv - f))
        .add_tp_benefit(-(1 - gamma) * f)
        .add_fp_cost(delta * clv + f)
    )
    print(cost_matrix)

    # ### Section 5.2 Modeling stochastic costs and uncertainty
    #
    # The acceptance rate becomes a Beta-distributed random variable. The incentive is written as a fixed
    # amount $d$ here, as in the EMP literature.

    print('\n' + 'Section 5.2 Modeling stochastic costs and uncertainty')

    from sympy.stats import Beta

    alpha, beta, d = sympy.symbols('alpha beta d')
    gamma = Beta('gamma', alpha, beta)
    stochastic_cost_matrix = (
        CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(d + f)
    )
    print(stochastic_cost_matrix)

    # ### Section 5.3 Configuration utilities

    print('\n' + 'Section 5.3 Configuration utilities')

    cost_matrix.alias({
        'incentive_fraction': 'delta',
        'contact_cost': 'f',
        'accept_rate': 'gamma',
    })
    cost_matrix.set_default(incentive_fraction=0.05, contact_cost=1, accept_rate=0.3)

    # ## Section 6: Evaluation metrics
    #
    # ### Section 6.1 The unified metric API
    #
    # *Replication setup:* the paper assumes a trained model and a metric. Here, a logistic
    # regression is trained on the churn data of Section 9, with the same split as the case study, and
    # the metric is the Expected Maximum Profit of Section 6.4 on the Section 5 cost matrix
    # (`Metric(cost_matrix, MaxProfit())`).

    print('\n' + 'Section 6: Evaluation metrics')

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import MaxProfit, Metric

    churn = fetch_iranian_churn(backend=pd)
    X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
        churn.data,
        churn.target,
        churn.instance_costs['clv'],
        test_size=0.3,
        random_state=42,
        stratify=churn.target,
    )
    clv_array = clv_test
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    metric = Metric(cost_matrix, MaxProfit())

    # Assume a trained probabilistic model and an instantiated metric
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Compute the expected business impact
    print(metric.score(y_test, y_proba, clv=clv_array))
    # or through the equivalent __call__
    print(metric(y_test, y_proba, clv=clv_array))

    # 2. Find the optimal probability cutoff for decision making
    print(metric.optimal_threshold(y_test, y_proba, clv=clv_array))

    # 3. Find the optimal target volume (e.g., target the top 15%)
    print(metric.optimal_rate(y_test, y_proba, clv=clv_array))

    # ### Section 6.2 Expected cost
    #
    # *Replication setup:* four illustrative customers, matching the four instance-dependent
    # customer lifetime values of the example.

    print('\n' + 'Section 6.2 Expected cost')

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.2, 0.8, 0.6, 0.4])

    from empulse.metrics import Cost, Metric

    cost_loss = Metric(cost_matrix=cost_matrix, strategy=Cost())
    result = cost_loss(
        y_true,
        y_pred,
        accept_rate=0.3,
        clv=[20, 10, 15, 5],  # Instance-dependent costs
    )
    print(result)

    # ### Section 6.3 Expected savings
    #
    # *Replication setup:* the baseline scores come from a logistic regression, as in the paper,
    # fitted here on four illustrative feature vectors for the four customers above.

    print('\n' + 'Section 6.3 Expected savings')

    X = np.array([[0.1, 1.2], [1.5, 0.3], [0.9, 0.8], [0.4, 1.0]])
    logistic_regression = LogisticRegression().fit(X, y_true)

    from empulse.metrics import Savings

    savings_metric = Metric(cost_matrix=cost_matrix, strategy=Savings())
    y_pred_baseline = logistic_regression.predict_proba(X)[:, 1]
    custom_savings = savings_metric(
        y_true,
        y_pred,
        accept_rate=0.3,
        clv=[20, 10, 15, 5],
        baseline=y_pred_baseline,  # User-defined general baseline S^b
    )
    print(custom_savings)

    # ### Section 6.4 Expected maximum profit

    print('\n' + 'Section 6.4 Expected maximum profit')

    from empulse.metrics import MaxProfit

    # The strategy automatically selects analytical or numerical integration
    emp_score = Metric(cost_matrix=stochastic_cost_matrix, strategy=MaxProfit(integration_method='auto'))
    result = emp_score(y_true, y_pred, alpha=6, beta=14, clv=200, d=10, f=1)
    print(result)

    # The EMP integration comparison of Section 6.4 is run with the Section 10 benchmarks.

    # ## Section 7: Estimators and scikit-learn integration
    #
    # ### Section 7.1 The estimator workflow and contract
    #
    # `metric` is the Expected Maximum Profit metric of Section 6.1.

    print('\n' + 'Section 7: Estimators and scikit-learn integration')

    from empulse.models import CSBoostClassifier

    model = CSBoostClassifier(loss=metric)
    model.fit(X_train, y_train, clv=clv_train)

    # ### Section 7.2 The optimizers module
    #
    # *Replication setup:* the features are standardized first, as gradient-based optimizers expect.

    print('\n' + 'Section 7.2 The optimizers module')

    X_train_scaled = StandardScaler().fit_transform(X_train)

    from empulse.models import CSLogitClassifier
    from empulse.optimizers import Adam

    model = CSLogitClassifier(optimizer=Adam(lr=0.01, max_iter=100))
    model.fit(X_train_scaled, y_train, fp_cost=5, fn_cost=1)

    # ### Section 7.3 The samplers module

    print('\n' + 'Section 7.3 The samplers module')

    from empulse.samplers import CostSensitiveSampler

    sampler = CostSensitiveSampler(loss=cost_loss, method='oversampling', random_state=42)
    X_re, y_re = sampler.fit_resample(X_train, y_train, clv=clv_train, accept_rate=0.3)

    # ### Section 7.4 scikit-learn integration and metadata routing
    #
    # `clv_array` holds the customer lifetime values of the training instances here.

    print('\n' + 'Section 7.4 scikit-learn integration and metadata routing')

    clv_array = clv_train

    import sklearn
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    from empulse.models import CSBoostClassifier

    sklearn.set_config(enable_metadata_routing=True)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'model',
            CSBoostClassifier(XGBClassifier(n_jobs=1, random_state=42), loss=cost_loss).set_fit_request(clv=True),
        ),
    ])
    scorer = make_scorer(cost_loss, response_method='predict_proba', greater_is_better=False)
    scorer.set_score_request(clv=True)
    search = GridSearchCV(pipeline, param_grid={'model__estimator__n_estimators': [50, 100]}, scoring=scorer)
    search.fit(X_train, y_train, clv=clv_array)

    # ## Section 8: Benchmark datasets
    #
    # *Replication setup:* `y_score` holds out-of-fold churn probabilities of a logistic regression.

    print('\n' + 'Section 8: Benchmark datasets')

    from sklearn.model_selection import cross_val_predict

    y_score = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        churn.data,
        churn.target,
        cv=5,
        method='predict_proba',
    )[:, 1]

    import pandas as pd

    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Metric, Profit

    dataset = fetch_iranian_churn(backend=pd)
    metric = Metric(dataset.cost_matrix, Profit())
    profit = metric(dataset.target, y_score, **dataset.instance_costs)
    print(profit)
    print(dataset)

    # ## Section 9: Case study: Customer churn prediction
    #
    # The code below is the paper's case study, cell for cell. Its output matches the output printed in
    # the paper.
    #
    # ### Section 9.1 Evaluating a standard baseline

    print('\n' + 'Section 9: Case study: Customer churn prediction')

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Metric, Profit

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']
    profit_score = Metric(dataset.cost_matrix, Profit())

    X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
        X, y, clv, test_size=0.3, random_state=42, stratify=y
    )

    baseline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(X_train, y_train)

    y_score_base = baseline.predict_proba(X_test)[:, 1]

    print(f'AUC: {roc_auc_score(y_test, y_score_base):.3f}')
    print(f'Accuracy: {accuracy_score(y_test, (y_score_base >= 0.5)):.3f}')
    profit = profit_score(y_test, y_score_base, clv=clv_test)
    print(f'Expected profit: {profit:.2f}')
    print(f'Fraction contacted: {(y_score_base >= 0.5).mean():.1%}')

    # ### Section 9.2 Training a cost-sensitive linear model

    print('\n' + 'Section 9.2 Training a cost-sensitive linear model')

    from sklearn import set_config

    from empulse.models import CSLogitClassifier

    set_config(enable_metadata_routing=True)

    model = CSLogitClassifier(loss=profit_score).set_fit_request(clv=True)
    cslogit = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ]).fit(X_train, y_train, clv=clv_train)

    y_score_cs = cslogit.predict_proba(X_test)[:, 1]

    print(f'AUC: {roc_auc_score(y_test, y_score_cs):.3f}')
    print(f'Accuracy: {accuracy_score(y_test, (y_score_cs >= 0.5)):.3f}')
    profit = profit_score(y_test, y_score_cs, clv=clv_test)
    print(f'Expected profit: {profit:.2f}')
    print(f'Fraction contacted: {(y_score_cs >= 0.5).mean():.1%}')

    # ### Section 9.3 Choosing who to contact
    #
    # `optimal_threshold` warns when a break-even threshold falls outside $[0, 1]$ and is clipped, which
    # happens for a few customers; the warning is silenced here, as it is not part of the paper's output.

    print('\n' + 'Section 9.3 Choosing who to contact')

    warnings.filterwarnings('ignore', message='.*threshold.*', category=UserWarning, module='empulse')

    # Compute individualized break-even thresholds
    base_thresholds = profit_score.optimal_threshold(y_test, y_score_base, clv=clv_test)
    cs_thresholds = profit_score.optimal_threshold(y_test, y_score_cs, clv=clv_test)

    base_contact = y_score_base >= base_thresholds
    cs_contact = y_score_cs >= cs_thresholds
    profit_base_opt = profit_score(y_test, base_contact.astype(float), clv=clv_test)
    profit_cs_opt = profit_score(y_test, cs_contact.astype(float), clv=clv_test)

    print(f'Baseline profit (per-customer threshold): {profit_base_opt:.2f}')
    print(f'Baseline fraction contacted: {base_contact.mean():.1%}')
    print(f'CSLogit profit (per-customer threshold): {profit_cs_opt:.2f}')
    print(f'CSLogit fraction contacted: {cs_contact.mean():.1%}')

    # ### Section 9.4 Cross-validated evaluation

    print('\n' + 'Section 9.4 Cross-validated evaluation')

    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score

    scorer = make_scorer(profit_score, response_method='predict_proba', greater_is_better=True).set_score_request(
        clv=True
    )

    scores = cross_val_score(cslogit, X, y, cv=5, scoring=scorer, params={'clv': clv})
    print(f'CSLogit 5-fold CV Mean profit: {scores.mean():.2f} (+/- {scores.std():.2f})')


def run_emp_benchmark(max_stochastic_variables: int = 2) -> None:
    """Run the EMP benchmark with full size sweeps and a configurable dimension limit."""
    print('\nEMP integration comparison (Section 6.4)')
    size = emp_integration.run_size_sweep()
    dimensions = emp_integration.run_dimension_sweep(dimensions=tuple(range(1, max_stochastic_variables + 1)))
    for name, table in figures.summarize_emp_integration(size, dimensions).items():
        print(f'\nEMP integration: {name}')
        print(table.to_string())
    figures.plot_emp_integration(size, dimensions)


def run_training_times(max_samples: int, repetitions: int = 1) -> None:
    """Run the limited training-time and analytical-gradient comparisons from Section 10."""
    print('\nTraining-time comparison (Section 10)')
    n_grid = [n for n in N_GRID if n <= max_samples] or [max_samples]
    results = pd.concat(
        [timing.run_timing(impl, n_grid, range(repetitions)) for impl in ('current', 'legacy')], ignore_index=True
    )

    print('\nAnalytical-gradient comparison (Section 10)')
    comparison = gradient.run_gradient_comparison(min(max_samples, 100_000), range(repetitions))
    print(comparison.to_string(index=False))
    print(gradient.summarize_gradient_comparison(comparison).to_string())

    summary = figures.summarize_timing(results)
    print('\nTraining times (seconds)')
    print(summary.to_string(index=False))
    print('\nSpeedup (reference / Empulse)')
    print(figures.speedup_table(summary).to_string(index=False))
    figures.plot_scaling(summary)


def main() -> None:
    max_emp_variables = 2
    max_training_samples = 3_000
    training_repetitions = 1

    with threadpool_limits(limits=1):
        run_examples()
        run_emp_benchmark(max_emp_variables)
        run_training_times(max_training_samples, training_repetitions)
    plt.show()


if __name__ == '__main__':
    main()
