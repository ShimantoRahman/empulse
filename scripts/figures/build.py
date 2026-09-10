"""Render every documentation figure, in both themes, into ``docs/_static/assets``.

Each figure is written twice: ``<name>_light.svg`` and ``<name>_dark.svg``. Both variants come from
one definition, so they cannot drift apart. The ``.. themed-figure::`` directive picks the right one
at read time based on the reader's theme.

Run with::

    just figures

The output is deterministic: running twice leaves the working tree clean.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from diagrams import DIAGRAMS  # noqa: E402
from palette import THEMES  # noqa: E402
from plots import PLOTS  # noqa: E402

OUTPUT_DIR = HERE.parents[1] / 'docs' / '_static' / 'assets'

FIGURES = {**DIAGRAMS, **PLOTS}


def check_well_formed(name: str, content: str) -> None:
    """Parse the SVG as XML before writing it.

    An SVG served to a browser as ``image/svg+xml`` goes through a strict XML parser and shows an
    error page rather than a picture if anything is malformed. Inlining the same markup into an
    HTML page hides the problem, because the HTML parser is forgiving — so the only reliable check
    is to parse it here. The usual culprit is an unescaped double quote inside an attribute, for
    example a font stack containing ``"Segoe UI"``.
    """
    try:
        ET.fromstring(content)
    except ET.ParseError as error:
        raise SystemExit(f'{name}: generated SVG is not well-formed XML: {error}') from error


def main() -> int:
    """Render every figure and report what changed."""
    if not OUTPUT_DIR.is_dir():
        print(f'output directory not found: {OUTPUT_DIR}', file=sys.stderr)
        return 1

    written, unchanged = 0, 0
    for name, render in FIGURES.items():
        for theme_name, theme in THEMES.items():
            target = OUTPUT_DIR / f'{name}_{theme_name}.svg'
            content = render(theme)
            check_well_formed(target.name, content)
            previous = target.read_text(encoding='utf-8') if target.exists() else None
            if previous == content:
                unchanged += 1
                continue
            target.write_text(content, encoding='utf-8')
            written += 1
            print(f'wrote {target.relative_to(OUTPUT_DIR.parents[2])}')

    print(f'\n{written} written, {unchanged} unchanged ({len(FIGURES)} figures x 2 themes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
