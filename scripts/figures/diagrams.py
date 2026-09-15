"""The schematic figures: structural claims that no dataset can make for you."""

from __future__ import annotations

from svg import Canvas


def empulse_spine(theme: dict[str, str]) -> str:
    """The shape of the package: define a cost matrix once, then reuse it four ways.

    Blue is the definition side, purple the use side, matching the guide's own order.
    """
    c = Canvas(920, 300, theme)

    # --- definition side ------------------------------------------------------------------
    # A miniature 2x2 grid stands in for the cost matrix everywhere it appears.
    gx, gy, cell = 40, 108, 26
    c.box(gx, gy, cell * 2, cell * 2, fill='blue-soft', stroke='blue-strong', radius=4)
    c.line(gx + cell, gy, gx + cell, gy + cell * 2, stroke='blue-strong', width=1)
    c.line(gx, gy + cell, gx + cell * 2, gy + cell, stroke='blue-strong', width=1)
    c.text(gx + cell, gy + cell * 2 + 20, 'CostMatrix', fill='blue-strong', size=13, weight='600')
    c.text(gx + cell, gy + cell * 2 + 38, 'what an outcome is worth', fill='ink-muted', size=11)

    c.text(gx + cell * 2 + 40, gy + cell, '+', fill='ink-muted', size=20)

    sx, sy, sw, sh = 168, gy + 6, 118, 40
    c.box(sx, sy, sw, sh, fill='blue-soft', stroke='blue-strong', radius=20)
    c.text(sx + sw / 2, sy + sh / 2, 'Strategy', fill='blue-strong', size=13, weight='600')
    c.text(sx + sw / 2, gy + cell * 2 + 38, 'how it becomes a number', fill='ink-muted', size=11)

    c.arrow(sx + sw + 12, gy + cell, 350, gy + cell, stroke='blue-strong', width=2)

    mx, my, mw, mh = 356, gy + 2, 104, 48
    c.box(mx, my, mw, mh, fill='blue-strong', stroke='blue-strong', radius=6)
    c.text(mx + mw / 2, my + mh / 2, 'Metric', fill='paper', size=15, weight='600')

    # --- use side -------------------------------------------------------------------------
    uses = [
        ('Evaluate', 'score a model in money', 2, 46),
        ('Train', 'optimise it directly', 3, 116),
        ('Decide', 'pick the cut-off', 4, 186),
    ]
    ux, uw, uh = 604, 172, 46
    for label, sub, stage, uy in uses:
        c.elbow_arrow(mx + mw + 10, my + mh / 2, ux - 12, uy + uh / 2, stroke='purple-strong')
        c.box(ux, uy, uw, uh, fill='purple-soft', stroke='purple-strong', radius=6)
        c.text(ux + 16, uy + 17, label, fill='purple-strong', size=13, weight='600', anchor='start')
        c.text(ux + 16, uy + 33, sub, fill='ink-muted', size=11, anchor='start')
        c.circle(ux + uw - 18, uy + uh / 2, 10, fill='purple-strong')
        c.text(ux + uw - 18, uy + uh / 2 + 1, str(stage), fill='paper', size=11, weight='600')

    # The sampler reads the cost matrix directly: it never needs a strategy.
    ry = 256
    c.path(
        f'M {gx + cell} {gy + cell * 2 + 48} V {ry} H {ux - 12}',
        stroke='purple-strong',
        width=1.5,
        dash='4 4',
    )
    c._head(ux - 12, ry, 0.0, 6, 'purple-strong')
    c.box(ux, ry - 23, uw, uh, fill='paper', stroke='purple-strong', radius=6)
    c.text(ux + 16, ry - 6, 'Resample', fill='purple-strong', size=13, weight='600', anchor='start')
    c.text(ux + 16, ry + 10, 'no metric needed', fill='ink-muted', size=11, anchor='start')
    c.circle(ux + uw - 18, ry, 10, fill='purple-strong')
    c.text(ux + uw - 18, ry + 1, '5', fill='paper', size=11, weight='600')

    c.text(240, 26, 'define once', fill='blue-strong', size=12, weight='600', italic=True)
    c.text(690, 26, 'reuse everywhere', fill='purple-strong', size=12, weight='600', italic=True)
    c.line(40, 40, 460, 40, stroke='blue-strong', width=1, dash='3 4')
    c.line(478, 40, 880, 40, stroke='purple-strong', width=1, dash='3 4')

    return c.render()


def cost_matrix_anatomy(theme: dict[str, str]) -> str:
    """The four cells, the cost/benefit sign flip, and what an array changes."""
    c = Canvas(960, 430, theme)

    cell_w, cell_h = 118, 54
    left_x, right_x, grid_y = 142, 612, 86

    def grid(ox: float, labels: list[tuple[str, str]], title: str, *, row_labels: bool) -> None:
        c.text(ox + cell_w, grid_y - 46, title, fill='ink', size=14, weight='600')
        c.text(ox + cell_w * 0.5, grid_y - 17, 'actual 1', fill='ink-muted', size=11)
        c.text(ox + cell_w * 1.5, grid_y - 17, 'actual 0', fill='ink-muted', size=11)
        if row_labels:
            for row, name in enumerate(('predicted 1', 'predicted 0')):
                c.text(ox - 12, grid_y + cell_h * (row + 0.5), name, fill='ink-muted', size=11, anchor='end')
        for index, (name, kind) in enumerate(labels):
            row, col = divmod(index, 2)
            x, y = ox + col * cell_w, grid_y + row * cell_h
            correct = index in {0, 3}
            c.box(
                x,
                y,
                cell_w,
                cell_h,
                fill='blue-soft' if correct else 'surface',
                stroke='blue-strong' if correct else 'rule',
                radius=4,
            )
            c.text(x + cell_w / 2, y + cell_h / 2 - 7, name, fill='ink', size=12, weight='600', mono=True)
            c.text(x + cell_w / 2, y + cell_h / 2 + 12, kind, fill='ink-muted', size=10)

    grid(
        left_x,
        [('tp_cost', 'correct'), ('fp_cost', 'error'), ('fn_cost', 'error'), ('tn_cost', 'correct')],
        'as costs',
        row_labels=True,
    )
    grid(
        right_x,
        [
            ('tp_benefit', 'correct'),
            ('fp_cost', 'error'),
            ('fn_cost', 'error'),
            ('tn_benefit', 'correct'),
        ],
        'as costs and benefits',
        row_labels=False,
    )

    # The two spellings differ only on the diagonal, so badge exactly those two cells.
    for row, col in ((0, 0), (1, 1)):
        bx = right_x + col * cell_w + cell_w - 13
        by = grid_y + row * cell_h + 13
        c.circle(bx, by, 9, fill='blue-strong')
        # U+2212 MINUS SIGN, not a hyphen: this is a mathematical negation.
        c.text(bx, by + 1, '−', fill='paper', size=13, weight='700')  # ruff: ignore[ambiguous-unicode-character-string]

    mid = (left_x + cell_w * 2 + right_x) / 2
    c.arrow(
        left_x + cell_w * 2 + 18,
        grid_y + cell_h,
        right_x - 18,
        grid_y + cell_h,
        stroke='ink-muted',
        width=1.5,
        double=True,
    )
    c.text(mid, grid_y + cell_h - 17, 'same number,', fill='ink-muted', size=11)
    c.text(mid, grid_y + cell_h + 19, 'opposite sign', fill='ink-muted', size=11)

    c.line(60, 248, 900, 248, stroke='rule', width=1, dash='4 4')

    # --- scalar versus array --------------------------------------------------------------
    def mini(ox: float, oy: float) -> None:
        s = 21
        c.box(ox, oy, s * 2, s * 2, fill='blue-soft', stroke='blue-strong', radius=3, stroke_width=1.2)
        c.line(ox + s, oy, ox + s, oy + s * 2, stroke='blue-strong', width=0.9)
        c.line(ox, oy + s, ox + s * 2, oy + s, stroke='blue-strong', width=0.9)

    c.text(60, 288, 'One number per outcome, or one per row', fill='ink', size=14, weight='600', anchor='start')

    mini(150, 326)
    c.text(192, 398, 'scalar', fill='ink', size=12, weight='600', mono=True)
    c.text(192, 416, 'one matrix for every row', fill='ink-muted', size=11)
    c.text(258, 352, 'fp_cost=5', fill='ink-muted', size=11, mono=True, anchor='start')

    for offset in (0, 9, 18):
        mini(596 + offset, 314 + offset)
    c.text(656, 398, 'array', fill='ink', size=12, weight='600', mono=True)
    c.text(656, 416, 'one matrix per row', fill='ink-muted', size=11)
    c.text(722, 352, 'fp_cost=[5, 12, 3, ...]', fill='ink-muted', size=11, mono=True, anchor='start')

    return c.render()


def threshold_and_rate(theme: dict[str, str]) -> str:
    """One cut through a ranked population, named two ways."""
    c = Canvas(920, 386, theme)

    bx, by, bw, bh = 90, 136, 740, 56
    cut = 0.26  # the fraction the tutorial's retention campaign ends up targeting
    cut_x = bx + bw * cut
    mid_left = (bx + cut_x) / 2

    c.text(bx, 34, 'Instances sorted by score, highest first', fill='ink', size=14, weight='600', anchor='start')

    c.box(bx, by, bw, bh, fill='surface', stroke='rule', radius=4)
    c.box(bx, by, bw * cut, bh, fill='purple-soft', stroke='purple-strong', radius=4)

    # Ticks so the bar reads as a ranked population rather than a progress bar. They take the
    # colour of whichever side of the cut they fall on, so they stay visible on both fills.
    for i in range(1, 40):
        x = bx + bw * i / 40
        c.line(x, by + bh - 13, x, by + bh - 5, stroke='purple-strong' if x < cut_x else 'rule', width=1)

    c.line(cut_x, by - 16, cut_x, by + bh + 16, stroke='purple-strong', width=2)

    c.text(bx + 14, by + bh / 2, 'act', fill='purple-strong', size=12, weight='600', anchor='start')
    c.text(bx + bw - 14, by + bh / 2, 'do nothing', fill='ink-muted', size=12, anchor='end')

    # --- the rate view, above the bar -----------------------------------------------------
    c.bracket(bx, cut_x, by - 22, stroke='purple-strong')
    c.text(mid_left, by - 74, 'top 26%', fill='purple-strong', size=14, weight='600')
    c.text(mid_left, by - 52, 'optimal_rate', fill='ink-muted', size=11, mono=True)

    # --- the threshold view, below the bar ------------------------------------------------
    c.text(cut_x + 16, by + bh + 30, 'score ≥ 0.31', fill='purple-strong', size=14, weight='600', anchor='start')
    c.text(cut_x + 16, by + bh + 50, 'optimal_threshold', fill='ink-muted', size=11, mono=True, anchor='start')

    # --- the function that converts between them ------------------------------------------
    c.text(460, 288, 'classification_threshold', fill='ink-muted', size=11, mono=True)
    c.text(460, 306, 'turns a rate into the cut-off that achieves it', fill='ink-muted', size=11)

    # --- which estimator fixes which view -------------------------------------------------
    c.box(bx + 34, 336, 226, 32, fill='paper', stroke='purple-strong', radius=16)
    c.text(bx + 147, 353, 'CSRateClassifier', fill='purple-strong', size=12, weight='600', mono=True)

    c.box(bx + 400, 336, 262, 32, fill='paper', stroke='purple-strong', radius=16)
    c.text(bx + 531, 353, 'CSThresholdClassifier', fill='purple-strong', size=12, weight='600', mono=True)

    c.text(bx + 147, 326, 'fixes the fraction', fill='ink-muted', size=11)
    c.text(bx + 531, 326, 'fixes the cut-off', fill='ink-muted', size=11)

    return c.render()


DIAGRAMS = {
    'empulse_spine': empulse_spine,
    'cost_matrix_anatomy': cost_matrix_anatomy,
    'threshold_and_rate': threshold_and_rate,
}
