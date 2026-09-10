"""The data figures.

Each one runs the same computation as the page it illustrates, so a figure cannot quietly disagree
with the numbers in the prose next to it.
"""

from __future__ import annotations

import io
import re
import warnings
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt
from palette import rc_params


def _save(fig: Any) -> str:
    """Serialise a figure to a scalable SVG string."""
    buffer = io.StringIO()
    fig.savefig(
        buffer,
        format='svg',
        bbox_inches='tight',
        pad_inches=0.08,
        transparent=True,
        metadata={'Date': None, 'Creator': None},
    )
    plt.close(fig)
    svg = buffer.getvalue()

    # Drop the XML/doctype preamble so the file drops into a page the way the diagrams do.
    svg = svg[svg.index('<svg') :]

    # Let the viewBox drive sizing rather than the absolute point size matplotlib writes. The
    # existing width and height have to be replaced rather than prepended, or the tag ends up with
    # two width attributes and the SVG stops being well-formed XML.
    opening_end = svg.index('>')
    opening = re.sub(r'\s(?:width|height)="[^"]*"', '', svg[:opening_end])
    return f'{opening} width="100%">{svg[opening_end + 1 :]}'


# ---------------------------------------------------------------------------------------------
# three_strategies
# ---------------------------------------------------------------------------------------------


def three_strategies(theme: dict[str, str]) -> str:
    """One set of predictions, three questions.

    The input panel is deliberately identical across all three: only the question changes.
    """
    from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric, Savings

    rng = np.random.default_rng(7)
    n = 600
    y_true = rng.binomial(1, 0.3, n)
    y_score = np.clip(rng.beta(2, 5, n) + 0.65 * y_true, 0.001, 0.999)

    matrix = CostMatrix().add_fp_cost('c_fp').add_fn_cost('c_fn').set_default(c_fp=1.0, c_fn=5.0)
    cost = Metric(matrix, Cost())(y_true, y_score)
    savings = Metric(matrix, Savings())(y_true, y_score)

    benefit, contact = 20.0, 2.0
    profit_matrix = CostMatrix().add_tp_benefit(benefit).add_fp_cost(contact)
    max_profit = Metric(profit_matrix, MaxProfit())(y_true, y_score)

    def caption(ax: Any, value: str, question: str, colour: str) -> None:
        """Put the answer under the panel, so it never sits on top of the data."""
        ax.text(0.5, -0.62, value, transform=ax.transAxes, ha='center', color=colour, fontsize=11, fontweight='medium')
        ax.text(
            0.5,
            -0.80,
            question,
            transform=ax.transAxes,
            ha='center',
            color=theme['ink-muted'],
            fontsize=9,
            style='italic',
        )

    with plt.rc_context(rc_params(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6))

        # -- panel 1: expected cost ----------------------------------------------------------
        ax = axes[0]
        ax.hist(
            y_score, bins=np.linspace(0, 1, 22), color=theme['blue-soft'], edgecolor=theme['blue-strong'], linewidth=0.8
        )
        ax.set_title('Cost', color=theme['blue-strong'])
        ax.set_xlabel('predicted probability')
        ax.set_yticks([])
        caption(ax, f'{cost:.2f} per instance', 'what does it cost?', theme['ink'])

        # -- panel 2: savings against a baseline ---------------------------------------------
        ax = axes[1]
        ax.barh([0], [1.0], color=theme['surface'], edgecolor=theme['rule'], height=0.42)
        ax.barh([0], [savings], color=theme['blue-soft'], edgecolor=theme['blue-strong'], height=0.42)
        ax.plot([savings, savings], [-0.21, 0.21], color=theme['blue-strong'], linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0\nnaive', '1\nperfect'])
        ax.set_title('Savings', color=theme['blue-strong'])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(length=0)
        caption(ax, f'{savings:.2f} of the way there', 'how much better than doing nothing?', theme['ink'])

        # -- panel 3: profit at the best cut-off ---------------------------------------------
        ax = axes[2]
        order = np.argsort(-y_score)
        labels = y_true[order]
        fractions = np.arange(1, n + 1) / n
        profits = (np.cumsum(labels) * benefit - np.cumsum(1 - labels) * contact) / n
        peak = int(np.argmax(profits))
        ax.plot(fractions, profits, color=theme['purple-strong'])
        ax.plot([fractions[peak]], [profits[peak]], 'o', color=theme['purple-strong'], markersize=6)
        ax.axhline(0, color=theme['rule'], linewidth=0.8)
        ax.set_xlabel('fraction targeted')
        ax.set_yticks([])
        ax.set_title('MaxProfit', color=theme['purple-strong'])
        caption(ax, f'{max_profit:.2f} at the peak', 'what is the best it could earn?', theme['ink'])

        # -- footer: what each panel needs from y_score --------------------------------------
        for ax, note, colour in (
            (axes[0], 'needs calibrated probabilities', theme['blue-strong']),
            (axes[1], 'needs calibrated probabilities', theme['blue-strong']),
            (axes[2], 'needs only a ranking', theme['purple-strong']),
        ):
            ax.text(
                0.5, -1.06, note, transform=ax.transAxes, ha='center', color=colour, fontsize=8.5, fontweight='medium'
            )

        fig.subplots_adjust(bottom=0.50, top=0.88, left=0.06, right=0.98, wspace=0.24)
        return _save(fig)


# ---------------------------------------------------------------------------------------------
# profit_curve
# ---------------------------------------------------------------------------------------------


def profit_curve(theme: dict[str, str]) -> str:
    """Three ranking metrics read off one curve: the peak, the area, and the hull."""
    rng = np.random.default_rng(3)
    n = 250
    y_true = rng.binomial(1, 0.30, n)
    y_score = np.clip(rng.beta(2, 5, n) + 0.30 * y_true, 0.001, 0.999)

    benefit, cost = 20.0, 2.0
    order = np.argsort(-y_score)
    labels = y_true[order]
    fractions = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    profits = np.concatenate([[0.0], (np.cumsum(labels) * benefit - np.cumsum(1 - labels) * cost) / n])
    # The oracle contacts every true positive first; it is what AUEPC normalises against.
    oracle_labels = np.sort(y_true)[::-1]
    oracle = np.concatenate([[0.0], (np.cumsum(oracle_labels) * benefit - np.cumsum(1 - oracle_labels) * cost) / n])

    peak = int(np.argmax(profits))

    # The ROC convex hull turns the ragged empirical curve into the achievable frontier.
    hull_idx = [0]
    for i in range(1, len(fractions)):
        while len(hull_idx) >= 2:
            a, b = hull_idx[-2], hull_idx[-1]
            cross = (fractions[b] - fractions[a]) * (profits[i] - profits[a]) - (profits[b] - profits[a]) * (
                fractions[i] - fractions[a]
            )
            if cross >= 0:
                hull_idx.pop()
            else:
                break
        hull_idx.append(i)

    with plt.rc_context(rc_params(theme)):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))

        ax.fill_between(fractions, 0, np.maximum(profits, 0), color=theme['purple-soft'], linewidth=0, zorder=1)
        ax.plot(fractions, oracle, color=theme['rule'], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2, label='oracle')
        ax.plot(
            fractions[hull_idx],
            profits[hull_idx],
            color=theme['blue-strong'],
            linewidth=1.4,
            linestyle=(0, (1, 2)),
            zorder=3,
            label='convex hull',
        )
        ax.plot(fractions, profits, color=theme['purple-strong'], zorder=4, label='profit')

        ax.axvline(fractions[peak], color=theme['purple-strong'], linewidth=1.0, linestyle=(0, (3, 3)), zorder=5)
        ax.plot([fractions[peak]], [profits[peak]], 'o', color=theme['purple-strong'], markersize=7, zorder=6)
        ax.axhline(0, color=theme['rule'], linewidth=0.8, zorder=0)

        top = profits[peak]
        ax.annotate(
            'MaxProfit\nthe highest point',
            xy=(fractions[peak], top),
            xytext=(fractions[peak] + 0.10, top + 0.95),
            color=theme['purple-strong'],
            fontsize=9,
            fontweight='medium',
            arrowprops={'arrowstyle': '-', 'color': theme['purple-strong'], 'linewidth': 0.9},
        )
        ax.annotate(
            f'optimal_rate = {fractions[peak]:.0%}',
            xy=(fractions[peak] + 0.015, 0.35),
            color=theme['purple-strong'],
            fontsize=8.5,
            family='monospace',
        )
        ax.annotate(
            'AUEPC\nthe whole area',
            xy=(0.62, top * 0.30),
            color=theme['purple-strong'],
            fontsize=9,
            fontweight='medium',
            ha='center',
        )
        ax.annotate(
            'EmpiricalMaxProfit\nrides the convex hull',
            xy=(0.24, np.interp(0.24, fractions[hull_idx], profits[hull_idx])),
            xytext=(0.04, 0.86),
            textcoords='axes fraction',
            color=theme['blue-strong'],
            fontsize=9,
            fontweight='medium',
            arrowprops={'arrowstyle': '-', 'color': theme['blue-strong'], 'linewidth': 0.9},
        )
        ax.annotate(
            'a perfect ranking',
            xy=(0.13, np.interp(0.13, fractions, oracle)),
            xytext=(0.20, top + 1.30),
            color=theme['ink-muted'],
            fontsize=8.5,
            style='italic',
            arrowprops={'arrowstyle': '-', 'color': theme['rule'], 'linewidth': 0.9},
        )

        ax.set_xlabel('fraction of the population targeted')
        ax.set_ylabel('profit per instance')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        fig.tight_layout()
        return _save(fig)


# ---------------------------------------------------------------------------------------------
# calibration_effect
# ---------------------------------------------------------------------------------------------


def calibration_effect(theme: dict[str, str]) -> str:
    """Calibration leaves the ranking alone and moves the money.

    Runs the identical experiment as ``docs/guide/deciding/calibration.rst`` so the annotated
    numbers are the page's own.
    """
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric

    X, y = make_classification(n_samples=4000, n_informative=6, weights=[0.85], random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    def forest() -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = forest().fit(X_train, y_train)
        cal = CalibratedClassifierCV(forest(), method='sigmoid', cv=3).fit(X_train, y_train)

    score_raw = raw.predict_proba(X_test)[:, 1]
    score_cal = cal.predict_proba(X_test)[:, 1]

    matrix = CostMatrix().add_tp_benefit(100).add_fp_cost(10)
    expected_cost = Metric(matrix, Cost())
    max_profit = Metric(matrix, MaxProfit())

    # Sign is irrelevant here: what matters is how far each measure moved, in its own units.
    measures = [
        ('ROC AUC', roc_auc_score(y_test, score_raw), roc_auc_score(y_test, score_cal), '{:.3f}'),
        ('max profit', max_profit(y_test, score_raw), max_profit(y_test, score_cal), '{:.2f}'),
        (
            'expected cost',
            expected_cost(y_test, score_raw),
            expected_cost(y_test, score_cal),
            '{:.2f}',
        ),
    ]

    with plt.rc_context(rc_params(theme)):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={'width_ratios': [1, 1.15]})

        # -- reliability ------------------------------------------------------------------
        # Quantile bins put both models on the same grid, so the curves are comparable.
        lim = 0.9
        ax_left.plot([0, lim], [0, lim], color=theme['rule'], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        for scores, colour, label in (
            (score_raw, theme['ink-muted'], 'uncalibrated'),
            (score_cal, theme['blue-strong'], 'calibrated'),
        ):
            observed, predicted = calibration_curve(y_test, scores, n_bins=8, strategy='quantile')
            ax_left.plot(predicted, observed, 'o-', color=colour, markersize=4, label=label, zorder=2)
        ax_left.annotate(
            'perfectly calibrated',
            xy=(0.72, 0.72),
            xytext=(0.40, 0.84),
            color=theme['ink-muted'],
            fontsize=8,
            style='italic',
            ha='center',
            arrowprops={'arrowstyle': '-', 'color': theme['rule'], 'linewidth': 0.8},
        )
        ax_left.set_xlim(0, lim)
        ax_left.set_ylim(0, lim)
        ax_left.set_xlabel('predicted probability')
        ax_left.set_ylabel('observed frequency')
        ax_left.set_title('The probabilities change')
        ax_left.legend(loc='lower right')

        # -- how far each measure moved ---------------------------------------------------
        positions = np.arange(len(measures))[::-1]
        changes = [abs(after - before) / abs(before) for _, before, after, _ in measures]
        biggest = max(changes)
        for pos, (_name, before, after, fmt), change in zip(positions, measures, changes, strict=True):
            moves = change > 0.05
            colour = theme['purple-strong'] if moves else theme['ink-muted']
            ax_right.barh([pos], [change], height=0.42, color=colour, edgecolor=colour, linewidth=0)
            ax_right.annotate(
                f'{change:+.1%}   ' + fmt.format(before) + ' → ' + fmt.format(after),
                xy=(change + biggest * 0.035, pos),
                va='center',
                color=theme['ink'],
                fontsize=9,
                fontweight='medium' if moves else 'normal',
            )

        ax_right.set_yticks(positions)
        ax_right.set_yticklabels([m[0] for m in measures])
        ax_right.set_xticks([])
        ax_right.set_xlim(0, biggest * 1.85)
        ax_right.set_ylim(-0.6, len(measures) - 0.4)
        ax_right.set_title('Only one of them moves')
        ax_right.set_xlabel('change after calibrating')
        ax_right.spines['bottom'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.tick_params(length=0)

        fig.tight_layout()
        return _save(fig)


PLOTS = {
    'three_strategies': three_strategies,
    'profit_curve': profit_curve,
    'calibration_effect': calibration_effect,
}
