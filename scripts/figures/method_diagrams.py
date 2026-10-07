"""Guide illustrations for bias mitigation and robust cost correction."""

from __future__ import annotations

import numpy as np
from guide_diagrams import heading
from svg import Canvas


def bias_mitigation(theme: dict[str, str]) -> str:
    """Compare labels, row multiplicities and weights on the same training customers."""
    from sklearn.linear_model import LogisticRegression

    from empulse.models.bias_mitigation.bias_reweighing import _independent_sample_weights
    from empulse.samplers import BiasRelabler, BiasResampler

    rows = list('ABCDEFGH')
    X = np.array([0.8, 0.2, 0.1, 0.9, 0.7, 0.6, 0.3, 0.05])[:, None]
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    high_value = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    _, relabeled = BiasRelabler(LogisticRegression()).fit_resample(X, labels, sensitive_feature=high_value)
    # Exact target counts keep the small resampling example at two rows per group and label.
    sampler = BiasResampler(
        strategy=lambda _y, _group: np.array([[2, 2 / 3], [2 / 3, 2]]),
        random_state=0,
    )
    resampled, _ = sampler.fit_resample(X, labels, sensitive_feature=high_value)
    counts = np.array([(resampled[:, 0] == value).sum() for value in X[:, 0]])
    weights = _independent_sample_weights(labels, high_value)

    c = Canvas(960, 826, theme)
    heading(
        c,
        'Three ways to change the same training data',
        'Eight customers. Label 1 means an event; label 0 means no event.',
    )
    for x, indices, title in (
        (28, range(4), 'High-value group: 1 event out of 4'),
        (508, range(4, 8), 'Low-value group: 3 events out of 4'),
    ):
        c.box(x, 104, 424, 110, fill='blue-soft', stroke=None, radius=8)
        c.text(x + 20, 128, title, size=19, weight='600', anchor='start')
        for col, index in enumerate(indices):
            cx = x + 20 + 98 * col
            c.box(cx, 153, 90, 40, fill='paper', stroke='blue-strong', radius=5)
            c.text(cx + 45, 173, f'{rows[index]}: {labels[index]}', size=20, mono=True)

    c.line(480, 225, 480, 254, stroke='purple-strong', width=2)
    c.line(170, 254, 790, 254, stroke='purple-strong', width=2)
    titles = ('Relabeling', 'Resampling', 'Reweighing')
    subtitles = ('Change labels', 'Change row counts', 'Change training weights')
    for col, (title, subtitle) in enumerate(zip(titles, subtitles, strict=True)):
        x = 28 + col * 310
        c.arrow(x + 142, 254, x + 142, 286, stroke='purple-strong', width=2)
        c.box(x, 296, 284, 438, fill='surface', radius=8)
        c.text(x + 20, 325, title, size=23, weight='600', anchor='start')
        c.text(x + 20, 354, subtitle, size=16, fill='ink-muted', anchor='start')
        c.text(x + 20, 388, 'Row', size=15, mono=True, fill='ink-muted', anchor='start')
        if col == 0:
            c.text(x + 264, 388, 'Label before / after', size=14, mono=True, fill='ink-muted', anchor='end')
        else:
            c.text(x + 96, 388, 'Label', size=15, mono=True, fill='ink-muted')
            c.text(x + 264, 388, 'Copies' if col == 1 else 'Weight', size=15, mono=True, fill='ink-muted', anchor='end')
        for index, row in enumerate(rows):
            y = 422 + index * 27
            if index == 4:
                c.line(x + 16, y - 14, x + 268, y - 14, width=1, dash='3 4')
            changed = labels[index] != relabeled[index] if col == 0 else counts[index] != 1 if col == 1 else False
            if changed:
                c.box(x + 12, y - 11, 260, 23, fill='purple-soft', stroke=None, radius=3)
            c.text(x + 20, y, row, size=18, mono=True, anchor='start')
            if col == 0:
                c.text(x + 167, y, f'{labels[index]}', size=18, mono=True, fill='ink-muted')
                c.arrow(x + 189, y, x + 221, y, stroke='purple-strong' if changed else 'rule', head=5)
                c.text(
                    x + 254, y, f'{relabeled[index]}', size=18, mono=True, fill='purple-strong' if changed else 'ink'
                )
            else:
                c.text(x + 96, y, f'{labels[index]}', size=18, mono=True)
                if col == 1:
                    c.text(
                        x + 264,
                        y,
                        f'{counts[index]}',
                        size=18,
                        mono=True,
                        anchor='end',
                        fill='purple-strong' if changed else 'ink',
                    )
                else:
                    c.box(x + 148, y - 4, 48 * weights[index], 8, fill='purple-strong', stroke=None, radius=2)
                    c.text(x + 264, y, f'{weights[index]:.2f}', size=18, mono=True, fill='purple-strong', anchor='end')
        c.line(x + 16, 634, x + 268, 634, width=1)
        notes = (
            ('A is promoted; G is demoted.', 'Rows and features stay the same.'),
            ('A and F are omitted.', 'D and H each appear twice.'),
            ('All rows and labels stay.', 'D and H have three times the weight.'),
        )
        for y, text in zip((658, 688), notes[col], strict=True):
            c.text(x + 20, y, text, size=14, anchor='start')
        c.text(
            x + 20,
            716,
            'Weighted event share: 50% per group' if col == 2 else 'Event share: 50% per group',
            size=13,
            fill='purple-strong',
            anchor='start',
        )
    c.line(28, 761, 932, 761, width=1)
    c.text(
        28,
        783,
        'Holdout data keeps its original rows and labels. Apply mitigation within each training fold.',
        size=16,
        fill='ink-muted',
        anchor='start',
    )
    c.text(
        28,
        808,
        'Illustrative resampling target: two rows per group and label. Dashed rules separate the value groups.',
        size=13,
        fill='ink-muted',
        anchor='start',
    )
    return c.render()


def robust_cost_correction(theme: dict[str, str]) -> str:
    """Show a computed cost residual and the replacement used by RobustCS."""
    from empulse.metrics import Cost, CostMatrix, Metric
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    X = np.linspace(0, 10, 40)[:, None]
    labels = np.tile([0, 1], 20)
    observed = 25 + 4 * X[:, 0] + np.random.default_rng(4).normal(0, 1.2, len(labels))
    outlier = 28
    observed[outlier] = 100
    loss = Metric(CostMatrix().add_fn_cost('clv').add_fp_cost(5).mark_outlier_sensitive('clv'), Cost())
    model = RobustCSClassifier(CSLogitClassifier(loss=loss)).fit(X, labels, clv=observed)
    predicted = model.outlier_estimators_['clv'].predict(X)
    corrected = model.costs_['clv']
    residuals = np.abs(observed - predicted)
    standardized = residuals[outlier] / np.std(residuals)
    replacement = corrected[outlier]

    c = Canvas(960, 628, theme)
    heading(
        c,
        'Correct an outlier cost before training',
        'RobustCS predicts CLV from the features, flags a large residual, then replaces that cost.',
    )
    c.box(28, 104, 526, 424, fill='surface', radius=8)
    c.text(84, 132, 'Observed CLV', size=16, mono=True, anchor='start')

    def px(value: float) -> float:
        return 84 + (value - 20) / 50 * 430

    def py(value: float) -> float:
        return 470 - (value - 20) / 85 * 306

    for value in (25, 50, 75, 100):
        c.line(84, py(value), 514, py(value), width=1)
        c.text(74, py(value), str(value), size=14, mono=True, fill='ink-muted', anchor='end')
    for value in (25, 45, 65):
        c.line(px(value), 470, px(value), 476, width=1)
        c.text(px(value), 491, str(value), size=14, mono=True, fill='ink-muted')
    c.line(84, 164, 84, 470, width=1)
    c.line(84, 470, 514, 470, width=1)
    c.line(px(20), py(20), px(70), py(70), stroke='ink-muted', dash='5 5', width=1.5)
    c.text(193, 246, 'Observed = predicted', size=15, fill='ink-muted')
    c.line(195, 260, px(37), py(37) - 10, stroke='rule', width=1)
    for index, (estimate, original) in enumerate(zip(predicted, observed, strict=True)):
        if index != outlier:
            c.circle(px(estimate), py(original), 4, fill='blue-strong')
    ox, oy, cy = px(predicted[outlier]), py(observed[outlier]), py(replacement)
    c.box(ox - 6, oy - 6, 12, 12, fill='purple-strong', stroke=None, radius=0)
    c.text(ox + 16, oy, '100', size=17, mono=True, fill='purple-strong', anchor='start')
    c.arrow(ox, oy + 16, ox, cy - 12, stroke='purple-strong', width=2, head=8)
    c.circle(ox, cy, 8, fill='paper')
    c.path(f'M {ox - 8} {cy} a 8 8 0 1 0 16 0 a 8 8 0 1 0 -16 0', stroke='purple-strong', width=2)
    c.text(ox + 17, cy + 15, f'{replacement:.1f}', size=17, mono=True, fill='purple-strong', anchor='start')
    c.text(299, 514, 'Predicted CLV from X', size=16, mono=True)

    c.text(590, 127, '1. Predict a cost', size=21, weight='600', anchor='start')
    c.text(590, 160, 'A Huber regressor uses the', size=17, anchor='start')
    c.text(590, 185, 'training features to estimate CLV.', size=17, anchor='start')
    c.text(590, 232, '2. Flag the residual', size=21, weight='600', anchor='start')
    c.text(
        590,
        267,
        f'Observed {observed[outlier]:.0f}; predicted {predicted[outlier]:.1f}',
        size=17,
        mono=True,
        anchor='start',
    )
    c.text(590, 297, 'Standardized absolute residual:', size=17, anchor='start')
    c.text(
        590,
        327,
        f'{standardized:.1f} > {model.outlier_threshold:.1f} threshold',
        size=19,
        mono=True,
        fill='purple-strong',
        anchor='start',
    )
    c.text(590, 378, '3. Replace this cost', size=21, weight='600', anchor='start')
    c.text(590, 413, f'100 becomes {replacement:.1f}.', size=19, mono=True, fill='purple-strong', anchor='start')
    c.text(590, 444, 'Other costs keep their values.', size=17, anchor='start')
    c.box(580, 472, 352, 56, fill='purple-soft', stroke='purple-strong', radius=6)
    c.text(756, 500, 'Fit the classifier on corrected costs', size=17, weight='600', fill='purple-strong')
    c.circle(38, 557, 4, fill='blue-strong')
    c.text(53, 557, 'Unchanged cost', size=15, anchor='start')
    c.box(270, 551, 12, 12, fill='purple-strong', stroke=None, radius=0)
    c.text(294, 557, 'Flagged cost', size=15, anchor='start')
    c.line(28, 580, 932, 580, width=1)
    c.text(
        28,
        606,
        'Features, labels and scalar costs stay unchanged. Learn the correction within each training fold.',
        size=15,
        fill='ink-muted',
        anchor='start',
    )
    return c.render()
