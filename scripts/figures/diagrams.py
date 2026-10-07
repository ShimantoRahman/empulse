"""The schematic figures: structural claims that no dataset can make for you."""

from __future__ import annotations

from guide_diagrams import (
    churn_outcomes,
    fold_aligned_costs,
    instance_thresholds,
    peak_and_area,
    threshold_rate_populations,
    uncertainty_order,
)
from method_diagrams import bias_mitigation, robust_cost_correction
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


DIAGRAMS = {
    'empulse_spine': empulse_spine,
    'churn_cost_benefit': churn_outcomes,
    'threshold_and_rate': threshold_rate_populations,
    'fold_aligned_costs': fold_aligned_costs,
    'instance_thresholds': instance_thresholds,
    'peak_and_area': peak_and_area,
    'uncertainty_order': uncertainty_order,
    'bias_mitigation': bias_mitigation,
    'robust_cost_correction': robust_cost_correction,
}
