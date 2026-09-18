"""The schematic figures: structural claims that no dataset can make for you."""

from __future__ import annotations

from svg import Canvas


def empulse_spine(theme: dict[str, str], *, numbered: bool = True, opaque: bool = False) -> str:
    """The shape of the package: define a cost matrix once, then reuse it four ways.

    Blue is the definition side, purple the use side, matching the guide's own order.

    ``numbered=False`` drops the use-side stage badges (there is no numbered prose pointing at
    them outside the guide) and narrows those boxes by the width a badge took up. ``opaque=True``
    paints an explicit page-coloured background instead of leaving the canvas transparent, for a
    figure meant to sit on a fixed white page rather than a theme-switching docs canvas.
    """
    width = 920 if numbered else 890
    c = Canvas(width, 320, theme)
    if opaque:
        c.box(0, 0, width, c.height, fill='paper', stroke=None, radius=0)

    # --- definition side ------------------------------------------------------------------
    # A miniature 2x2 grid stands in for the cost matrix everywhere it appears. Centred on the
    # midpoint of the four use-side rows below, so the single arrow into Metric splits evenly.
    gx, gy, cell = 40, 144, 26
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
    # Preprocess reads the metric like every other use: the cost matrix alone cannot rebalance
    # anything without a strategy to turn it into per-row weights.
    uses = [
        ('Evaluate', 'score a model in money', 2, 42),
        ('Train', 'optimise it directly', 3, 112),
        ('Decide', 'pick the cut-off', 4, 182),
        ('Preprocess', 'rebalance the data', 5, 252),
    ]
    # A badge is a 20px circle sitting 18px in from the right edge; dropping it frees roughly that
    # much width, so the box (and the canvas that fits around it) can shrink by the same amount.
    ux, uw, uh = 604, 172 if numbered else 142, 46
    for label, sub, stage, uy in uses:
        c.elbow_arrow(mx + mw + 10, my + mh / 2, ux - 12, uy + uh / 2, stroke='purple-strong')
        c.box(ux, uy, uw, uh, fill='purple-soft', stroke='purple-strong', radius=6)
        c.text(ux + 16, uy + 17, label, fill='purple-strong', size=13, weight='600', anchor='start')
        c.text(ux + 16, uy + 33, sub, fill='ink-muted', size=11, anchor='start')
        if numbered:
            c.circle(ux + uw - 18, uy + uh / 2, 10, fill='purple-strong')
            c.text(ux + uw - 18, uy + uh / 2 + 1, str(stage), fill='paper', size=11, weight='600')

    right_edge = width - 40
    reuse_x = (478 + right_edge) / 2
    c.text(240, 18, 'define once', fill='blue-strong', size=15, weight='700')
    c.text(reuse_x, 18, 'reuse everywhere', fill='purple-strong', size=15, weight='700')
    c.line(40, 32, 460, 32, stroke='blue-strong', width=1, dash='3 4')
    c.line(478, 32, right_edge, 32, stroke='purple-strong', width=1, dash='3 4')

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


def churn_cost_benefit(theme: dict[str, str]) -> str:
    """Where the campaign's money goes: back to the customer base, or out with the churner.

    Redraws the same flow the legacy ``churn_cost_benefit.png`` did — contacted customers split
    by whether the incentive works, loyal customers contacted for nothing, both other cells
    costing nothing because no campaign touches them — in the package's own visual language
    instead of green/red arrows.
    """
    c = Canvas(920, 380, theme)

    base_x, base_y, base_w, base_h = 40, 90, 150, 190
    outflow_x, outflow_y, outflow_w, outflow_h = 730, 110, 150, 70
    mid_x, mid_y, mid_w, mid_h = 250, 40, 420, 260

    c.box(base_x, base_y, base_w, base_h, fill='paper', stroke='rule', radius=10)
    c.text(base_x + base_w / 2, base_y + base_h / 2, 'Customer base', fill='ink', size=13, weight='600')

    c.box(outflow_x, outflow_y, outflow_w, outflow_h, fill='paper', stroke='rule', radius=10)
    c.text(outflow_x + outflow_w / 2, outflow_y + outflow_h / 2, 'Outflow', fill='ink-muted', size=13, weight='600')

    c.box(mid_x, mid_y, mid_w, mid_h, fill='none', stroke='rule', radius=10)
    c.text(mid_x + mid_w / 2, mid_y + 24, 'Contacted', fill='ink', size=14, weight='600')

    bar_x, bar_w = mid_x + 20, mid_w - 40

    # --- real churners: split by whether the incentive works ----------------------------------
    churner_y = mid_y + 60
    c.text(bar_x, churner_y - 10, 'real churner', fill='ink-muted', size=11, anchor='start')
    bar_h = 30
    gamma = 0.3  # the accept_rate default in the code sample right below this figure
    c.box(bar_x, churner_y, bar_w, bar_h, fill='surface', stroke='rule', radius=6)
    accept_w = bar_w * gamma
    c.box(bar_x, churner_y, accept_w, bar_h, fill='blue-strong', stroke='blue-strong', radius=6)
    # Greek gamma and U+2212 MINUS SIGN below mirror the sympy symbols, not ASCII lookalikes.
    c.text(bar_x + 10, churner_y + bar_h / 2, 'accept · γ', fill='paper', size=11, weight='600', anchor='start')  # ruff: ignore[ambiguous-unicode-character-string]
    c.text(bar_x + bar_w - 10, churner_y + bar_h / 2, 'leave anyway', fill='ink-muted', size=11, anchor='end')

    arrow_y = churner_y + bar_h / 2
    c.arrow(bar_x, arrow_y, base_x + base_w, arrow_y, stroke='blue-strong', width=2)
    c.arrow(bar_x + bar_w, arrow_y, outflow_x, arrow_y, stroke='ink-muted', width=1.5)

    # Symbol names match the sympy code right below: clv, d (incentive_cost), f (contact_cost).
    for offset, label in enumerate(('+ clv', '− d', '− f')):  # ruff: ignore[ambiguous-unicode-character-string]
        c.text(
            base_x + base_w + 10,
            arrow_y + 10 + offset * 13,
            label,
            fill='ink-muted',
            size=10,
            mono=True,
            anchor='start',
        )
    c.text((bar_x + bar_w + outflow_x) / 2, arrow_y + 10, '− f', fill='ink-muted', size=10, mono=True)  # ruff: ignore[ambiguous-unicode-character-string]

    # --- loyal customers: contacted anyway, stay regardless ------------------------------------
    loyal_y = churner_y + bar_h + 40
    c.text(bar_x, loyal_y - 10, 'loyal customer', fill='ink-muted', size=11, anchor='start')
    c.box(bar_x, loyal_y, bar_w, bar_h, fill='blue-strong', stroke='blue-strong', radius=6)
    c.text(bar_x + bar_w / 2, loyal_y + bar_h / 2, 'stays regardless', fill='paper', size=11, weight='600')

    loyal_arrow_y = loyal_y + bar_h / 2
    c.arrow(bar_x, loyal_arrow_y, base_x + base_w, loyal_arrow_y, stroke='blue-strong', width=2)
    for offset, label in enumerate(('− d', '− f')):  # ruff: ignore[ambiguous-unicode-character-string]
        c.text(
            base_x + base_w + 10,
            loyal_arrow_y + 10 + offset * 13,
            label,
            fill='ink-muted',
            size=10,
            mono=True,
            anchor='start',
        )

    # --- not contacted: no campaign, so no cost either way -------------------------------------
    rest_y = mid_y + mid_h + 20
    c.box(mid_x, rest_y, mid_w, 40, fill='none', stroke='rule', radius=8, dash='4 4')
    c.text(mid_x + mid_w / 2, rest_y + 20, 'Not contacted — no campaign cost either way', fill='ink-muted', size=11)

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
    'churn_cost_benefit': churn_cost_benefit,
    'threshold_and_rate': threshold_and_rate,
}
