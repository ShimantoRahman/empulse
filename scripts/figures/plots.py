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
    'calibration_effect': calibration_effect,
}
