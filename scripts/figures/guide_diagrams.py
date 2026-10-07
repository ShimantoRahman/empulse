"""Schematic illustrations used by the user guide."""

from __future__ import annotations

import numpy as np
from svg import Canvas


def heading(c: Canvas, title: str, subtitle: str) -> None:
    """Introduce the question an illustration answers."""
    c.text(28, 30, title, size=24, weight='600', anchor='start')
    c.text(28, 65, subtitle, size=16, fill='ink-muted', anchor='start')


def fold_aligned_costs(theme: dict[str, str]) -> str:
    """One fit request delivers the matching training costs to every cross-validation clone."""
    from sklearn.model_selection import StratifiedKFold

    c = Canvas(960, 690, theme)
    heading(
        c,
        'Request once, receive training costs in every fold',
        "The cross-validator slices clv with X and y before calling each cloned model's fit.",
    )
    rows = [('A', 40), ('B', 200), ('C', 90), ('D', 350), ('E', 60), ('F', 180)]
    c.box(28, 104, 904, 56, fill='blue-soft', stroke=None, radius=8)
    c.text(52, 132, 'model.set_fit_request(clv=True)', size=22, mono=True, anchor='start')
    c.text(28, 189, 'Full data: rows of X and y, with one clv value per row', size=19, anchor='start')
    for index, (row, value) in enumerate(rows):
        x = 28 + index * 154
        c.box(x, 214, 134, 66, fill='surface', radius=5)
        c.text(x + 67, 234, f'Row {row}', size=18, weight='600')
        c.text(x + 67, 260, f'clv={value}', size=17, mono=True, fill='blue-strong')
    c.bracket(28, 932, 295, below=True)
    c.arrow(480, 302, 480, 326, stroke='purple-strong', width=2)
    c.box(28, 334, 904, 58, fill='purple-soft', stroke='purple-strong', radius=8)
    c.text(480, 363, "cross_val_score(model, X, y, params={'clv': clv}, cv=3)", size=20, mono=True)
    c.line(480, 400, 480, 432, stroke='purple-strong', width=2)
    c.line(160, 432, 800, 432, stroke='purple-strong', width=2)
    folds = StratifiedKFold(n_splits=3).split(np.zeros((6, 1)), [0, 1, 0, 1, 0, 1])
    for index, (train, validation) in enumerate(folds):
        x, centre = 28 + index * 320, 160 + index * 320
        c.arrow(centre, 432, centre, 462, stroke='purple-strong', width=2)
        c.box(x, 470, 264, 164, fill='paper', radius=8)
        c.text(centre, 494, f'Fold {index + 1}', size=21, weight='600')
        held_out = ', '.join(rows[i][0] for i in validation)
        training = ', '.join(rows[i][0] for i in train)
        values = ','.join(str(rows[i][1]) for i in train)
        c.text(centre, 525, f'Hold out {held_out}', size=17, fill='ink-muted')
        c.box(x + 12, 549, 240, 70, fill='blue-soft', stroke=None, radius=5)
        c.text(centre, 570, f'fit on {training}', size=18, weight='600')
        c.text(centre, 598, f'clv=[{values}]', size=15, mono=True)
    c.text(
        480, 670, 'With metadata routing enabled, each fit receives only its training rows and matching costs.', size=17
    )
    return c.render()


def instance_thresholds(theme: dict[str, str]) -> str:
    """Equal probabilities can justify opposite actions when the missed value differs."""
    from empulse.metrics import Cost, CostMatrix, Metric

    c = Canvas(880, 458, theme)
    heading(
        c,
        'Same probability, different decision',
        'Correct decisions cost 0. A false positive costs 10 for both customers.',
    )
    probability = 0.25
    missed = np.array([20.0, 100.0])
    metric = Metric(CostMatrix().add_fp_cost(10).add_fn_cost('missed_value'), Cost())
    thresholds = metric.optimal_threshold([0, 1], [probability, probability], missed_value=missed)
    for x, value, threshold in zip((28, 468), missed, thresholds, strict=True):
        act_cost = (1 - probability) * 10
        no_act_cost = probability * value
        act = probability >= threshold
        c.box(x, 104, 384, 270, fill='surface', radius=8)
        c.text(x + 24, 133, f'False negative costs {value:.0f}', size=20, weight='600', anchor='start')
        c.text(x + 24, 174, f'p = {probability:.2f}    threshold = {threshold:.3f}', size=16, mono=True, anchor='start')
        c.text(x + 24, 222, f'Act:       0.75 x 10 = {act_cost:.2f}', size=16, mono=True, anchor='start')
        c.text(x + 24, 254, f'Do nothing: 0.25 x {value:.0f} = {no_act_cost:.2f}', size=16, mono=True, anchor='start')
        c.box(x + 24, 297, 336, 50, fill='purple-soft', stroke='purple-strong', radius=5)
        c.text(x + 192, 322, 'Act' if act else 'Do nothing', size=22, weight='600', fill='purple-strong')
    c.text(440, 405, 'Threshold = fp_cost / (fp_cost + fn_cost)', size=17, mono=True)
    c.text(440, 437, 'Choose the action with the lower expected cost.', size=18)
    return c.render()


def churn_outcomes(theme: dict[str, str]) -> str:
    """Derive the true-positive expression from two possible responses to an offer."""
    c = Canvas(880, 482, theme)
    heading(
        c,
        'What does one retention contact earn?',
        'Values are relative to running no campaign. gamma is the acceptance rate.',
    )
    c.box(28, 108, 208, 188, fill='surface', radius=8)
    c.text(132, 143, 'Contact a churner', size=19, weight='600')
    c.text(132, 180, 'True positive', size=17, fill='ink-muted')
    c.text(132, 224, 'Always pay f', size=18, mono=True)
    for y, title, amount, chance in (
        (108, 'Accepts and stays', 'CLV - d - f', 'gamma'),
        (220, 'Declines and leaves', '-f', '1 - gamma'),
    ):
        c.box(432, y, 418, 88, fill='blue-soft', stroke=None, radius=8)
        c.text(453, y + 27, title, size=19, weight='600', anchor='start')
        c.text(453, y + 60, amount, size=20, mono=True, anchor='start')
        c.elbow_arrow(246, 202, 420, y + 44, stroke='purple-strong', width=2)
        c.text(331, y + (25 if y == 108 else 76), chance, size=15, mono=True, fill='purple-strong')
    c.text(440, 343, 'tp_benefit = gamma x (CLV - d - f) + (1 - gamma) x (-f)', size=17, mono=True)
    c.line(28, 378, 850, 378, width=1)
    c.text(28, 411, 'Contact a loyal customer', size=19, weight='600', anchor='start')
    c.text(850, 411, 'fp_cost = d + f', size=19, mono=True, anchor='end')
    c.text(28, 452, 'Do not contact either customer', size=19, weight='600', anchor='start')
    c.text(850, 452, 'fn_cost = tn_cost = 0', size=19, mono=True, anchor='end')
    return c.render()


def threshold_rate_populations(theme: dict[str, str]) -> str:
    """The same threshold selects different fractions on different score distributions."""
    c = Canvas(880, 445, theme)
    heading(
        c,
        'A threshold and a rate agree on one population',
        'Keep the same model and threshold. Change the population, and the rate can change.',
    )
    scores = (
        [0.10, 0.18, 0.22, 0.29, 0.40, 0.52, 0.62, 0.68, 0.80, 0.90],
        [0.12, 0.16, 0.21, 0.25, 0.31, 0.36, 0.42, 0.49, 0.65, 0.86],
    )
    left, width, threshold = 238, 490, 0.6
    for y, values, title in zip((170, 294), scores, ('Population A', 'Population B'), strict=True):
        c.text(28, y, title, size=20, weight='600', anchor='start')
        c.line(left, y, left + width, y, width=2)
        threshold_x = left + width * threshold
        c.line(threshold_x, y - 27, threshold_x, y + 27, stroke='purple-strong', width=2)
        for value in values:
            c.circle(left + width * value, y, 7, fill='purple-strong' if value >= threshold else 'ink-muted')
        count = sum(value >= threshold for value in values)
        c.text(850, y, f'{count}/10 act', size=18, mono=True, fill='purple-strong', anchor='end')
        c.text(threshold_x, y + 47, 'threshold 0.60', size=16, mono=True, fill='purple-strong')
        c.text(left, y + 30, '0', size=14, mono=True)
        c.text(left + width, y + 30, '1', size=14, mono=True)
    c.text(440, 378, 'A fixed rate selects a fraction of each new batch.', size=18)
    c.text(440, 411, "Converting that rate to a threshold needs that batch's scores.", size=18)
    return c.render()


def peak_and_area(theme: dict[str, str]) -> str:
    """Compute the peak and the oracle-relative area for a small ranking with Empulse."""
    from empulse.metrics import AUEPC, CostMatrix, EmpiricalMaxProfit, Metric

    c = Canvas(880, 462, theme)
    heading(
        c,
        'A peak and an area describe different strengths',
        'One illustrative profit curve. The horizontal axis is the fraction targeted.',
    )
    delta = np.array([5, -2, 4, -1, -1, -2, -2, -3], dtype=float)
    labels, scores, amounts = (delta > 0).astype(int), np.arange(8, 0, -1), np.abs(delta)
    matrix = CostMatrix().add_tp_benefit('amount').add_fp_cost('amount')
    peak_value = Metric(matrix, EmpiricalMaxProfit())(labels, scores, amount=amounts)
    area_value = Metric(matrix, AUEPC())(labels, scores, amount=amounts)
    profit, oracle = np.cumsum(delta), np.cumsum(np.sort(delta)[::-1])
    stop = int(np.flatnonzero(oracle <= 0)[0])
    ratio = profit[:stop] / oracle[:stop]

    def curve(
        x: float,
        fractions: np.ndarray,
        values: np.ndarray,
        scale: float,
        *,
        shade: bool = False,
        dash: str | None = None,
        colour: str = 'blue-strong',
    ) -> tuple[np.ndarray, np.ndarray]:
        px, py = x + fractions * 320, 334 - values * scale
        points = ' '.join(f'L {a:.1f} {b:.1f}' for a, b in zip(px[1:], py[1:], strict=True))
        if shade:
            c.path(
                f'M {px[0]} 334 L {px[0]} {py[0]} ' + points + f' L {px[-1]} 334 Z',
                fill='blue-soft',
                stroke='none',
                width=0,
            )
        c.path(f'M {px[0]} {py[0]} ' + points, stroke=colour, width=2.5, dash=dash)
        return px, py

    fractions = np.arange(1, 9) / 8
    for x, title in ((58, 'EmpiricalMaxProfit'), (493, 'AUEPC')):
        c.text(x + 160, 112, title, size=21, weight='600')
        c.line(x, 174, x, 370)
        c.line(x, 334, x + 320, 334)
        for value in (0, 0.5, 1):
            c.text(x - 12, 334 - value * 140, f'{value:g}', size=13, mono=True, anchor='end', fill='ink-muted')
        for fraction, label in ((0, '0%'), (0.5, '50%'), (1, '100%')):
            c.text(x + 320 * fraction, 389, label, size=14, mono=True)
    curve(58, np.r_[0, fractions], np.r_[0, oracle / 8], 140, dash='5 5', colour='ink-muted')
    px, py = curve(58, np.r_[0, fractions], np.r_[0, profit / 8], 140)
    peak = int(profit.argmax()) + 1
    c.circle(px[peak], py[peak], 6, fill='purple-strong')
    c.text(58, 156, 'profit per customer', size=15, mono=True, anchor='start')
    c.text(58 + 320, 156, 'dashed = oracle', size=15, anchor='end', fill='ink-muted')
    c.text(493, 156, 'profit / oracle profit', size=15, mono=True, anchor='start')
    curve(493, fractions[:stop], ratio, 140, shade=True)
    c.line(493 + stop / 8 * 320, 174, 493 + stop / 8 * 320, 367, dash='4 4', stroke='purple-strong')
    c.text(493 + 320, 358, 'stop', size=15, fill='purple-strong', anchor='end')
    c.text(218, 418, f'Peak = {peak_value:.3f} per customer', size=17, mono=True)
    c.text(653, 418, f'Normalized area = {area_value:.3f}', size=17, mono=True)
    c.text(
        440,
        450,
        'AUEPC integrates the profit ratio until the oracle profit is no longer positive.',
        size=16,
        fill='ink-muted',
    )
    return c.render()


def uncertainty_order(theme: dict[str, str]) -> str:
    """Show which threshold profits enter the expectation over uncertain acceptance."""
    from empulse.metrics import Cost, CostMatrix, Metric

    c = Canvas(960, 477, theme)
    heading(
        c,
        'What gets averaged in expected maximum profit?',
        'Two possible offer acceptance rates, each with probability 50%. Profits are per customer.',
    )
    labels, scores = np.array([1, 0, 1]), np.array([0.9, 0.5, 0.4])
    thresholds, acceptance = (0.7, 0.3), (0.2, 0.6)
    metric = Metric(CostMatrix().add_tp_benefit('accept_rate * 90').add_fp_cost(30), Cost())
    profits = np.array([
        [-metric(labels, (scores >= threshold).astype(float), accept_rate=rate) for rate in acceptance]
        for threshold in thresholds
    ])
    for col, (x, rate) in enumerate(zip((28, 508), acceptance, strict=True)):
        c.box(x, 104, 424, 206, fill='surface', radius=8)
        c.text(x + 24, 134, f'{rate:.0%} of churners accept', size=22, weight='600', anchor='start')
        for row, threshold in enumerate(thresholds):
            y = 181 + row * 39
            c.text(x + 24, y, f'Threshold {threshold:.2f}', size=19, mono=True, anchor='start')
            c.text(x + 400, y, f'profit {profits[row, col]:.0f}', size=19, mono=True, anchor='end')
        winner = int(profits[:, col].argmax())
        c.box(x + 20, 248, 384, 42, fill='purple-soft', stroke='purple-strong', radius=5)
        c.text(
            x + 212,
            269,
            f'Choose {thresholds[winner]:.2f}: profit {profits[winner, col]:.0f}',
            size=20,
            mono=True,
            fill='purple-strong',
        )
        c.arrow(x + 212, 318, 428 if col == 0 else 532, 346, stroke='purple-strong', width=2)
    best_by_scenario = profits.max(axis=0)
    c.box(28, 354, 904, 65, fill='purple-soft', stroke='purple-strong', radius=8)
    c.text(52, 386, 'Expected maximum profit', size=23, weight='600', fill='purple-strong', anchor='start')
    c.text(
        908,
        386,
        f'0.5 x {best_by_scenario[0]:.0f} + 0.5 x {best_by_scenario[1]:.0f} = {best_by_scenario.mean():.0f}',
        size=22,
        mono=True,
        fill='purple-strong',
        anchor='end',
    )
    c.text(
        28,
        455,
        'In practice, a continuous distribution replaces these two scenarios. The idea is the same.',
        size=15,
        fill='ink-muted',
        anchor='start',
    )
    return c.render()
