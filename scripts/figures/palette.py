"""Visual language for the Empulse documentation figures.

The blue ramp is sampled from ``docs/_static/assets/empulse_logo_light.png``; the purple ramp is
the same hue rotation applied to the logo's own saturation and lightness, so it reads as a sibling
rather than an import.

Colour carries meaning and is not decorative:

* **blue** marks what you *write down* — cost matrices, strategies, metrics, parameters;
* **purple** marks what you *do with it* — evaluating, training, deciding, resampling.

Correct and incorrect outcomes inside a cost matrix are separated by tint (soft versus strong),
never by hue, because hue is reserved for the definition/use axis.

Each token has a light and a dark value because the logo's own colours cannot serve both themes:
``#244876`` scores 9.28 contrast on white but 1.92 on the theme's dark background.
"""

from typing import Any, Final

# Font names are single-quoted: this string is emitted inside a double-quoted XML attribute, and a
# nested double quote would terminate it and produce an unparseable SVG.
FONT_STACK: Final[str] = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

LIGHT: Final[dict[str, str]] = {
    'paper': '#FFFFFF',
    'surface': '#F4F7F9',
    'ink': '#244876',
    'ink-muted': '#5C6B80',
    'blue-strong': '#1B7DBF',
    'blue-soft': '#D3E7F6',
    'purple-strong': '#7F4ACF',
    'purple-soft': '#E4D9F7',
    'rule': '#C4D0DE',
}

DARK: Final[dict[str, str]] = {
    'paper': '#14181E',
    'surface': '#222832',
    'ink': '#E8EEF4',
    'ink-muted': '#9AA9BC',
    'blue-strong': '#37ABDF',
    'blue-soft': '#1D3A50',
    'purple-strong': '#A173DE',
    'purple-soft': '#332748',
    'rule': '#3A444F',
}

THEMES: Final[dict[str, dict[str, str]]] = {'light': LIGHT, 'dark': DARK}


def rc_params(theme: dict[str, str]) -> dict[str, Any]:
    """Matplotlib settings that make a plot look like the rest of the documentation.

    Every colour is transparent or drawn from ``theme`` so the figure sits on the page's own
    background rather than a white card.
    """
    return {
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'savefig.facecolor': 'none',
        'savefig.transparent': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 9,
        'text.color': theme['ink'],
        'axes.labelcolor': theme['ink'],
        'axes.edgecolor': theme['rule'],
        'axes.titlecolor': theme['ink'],
        'axes.titlesize': 10,
        'axes.titleweight': 'medium',
        'axes.labelsize': 9,
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': theme['ink-muted'],
        'ytick.color': theme['ink-muted'],
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'grid.color': theme['rule'],
        'grid.linewidth': 0.8,
        'legend.frameon': False,
        'legend.fontsize': 8,
        'lines.linewidth': 1.8,
        'lines.solid_capstyle': 'round',
        'patch.linewidth': 1.2,
        'svg.fonttype': 'path',
        # Deterministic element ids, so regenerating produces a byte-identical file.
        'svg.hashsalt': 'empulse-docs',
    }
