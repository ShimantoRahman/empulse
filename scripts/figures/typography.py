"""Use the documentation's bundled fonts and outline lettering in SVG images."""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

FONT_DIR = Path(__file__).resolve().parents[2] / 'docs' / '_static' / 'fonts'
_FONT_CACHE = tempfile.TemporaryDirectory(prefix='empulse-figure-fonts-', ignore_cleanup_errors=True)


@cache
def font_file(mono: bool = False, weight: int = 400) -> str:
    """Convert a bundled variable WOFF2 face to a static font Matplotlib can read."""
    source = 'jetbrains-mono-variable.woff2' if mono else 'ShoraiSansStdNVar-Latin-Plus-Symbols.woff2'
    font = TTFont(FONT_DIR / source)
    static = instantiateVariableFont(font, {'wght': weight}, inplace=True)
    static.flavor = None
    target = Path(_FONT_CACHE.name) / f'{"mono" if mono else "sans"}-{weight}.ttf'
    static.save(target)
    font.close()
    font_manager.fontManager.addfont(str(target))
    return str(target)


def font_family(mono: bool = False) -> str:
    """Return the actual family name recorded in the bundled font."""
    return font_manager.FontProperties(fname=font_file(mono)).get_name()


@cache
def lettering(content: str, size: float, mono: bool, weight: int) -> tuple[str, tuple[float, float, float, float]]:
    """Return outline paths and their bounds, in a coordinate system with y pointing up."""
    prop = font_manager.FontProperties(fname=font_file(mono, weight))
    text_path = TextPath((0, 0), content, size=size, prop=prop, usetex=False)
    parts = []
    commands = {MplPath.MOVETO: 'M', MplPath.LINETO: 'L', MplPath.CURVE3: 'Q', MplPath.CURVE4: 'C'}
    for vertices, code in text_path.iter_segments(curves=True, simplify=False):
        if code == MplPath.CLOSEPOLY:
            parts.append('Z')
        else:
            parts.append(commands[code] + ' '.join(f'{value:.3f}' for value in vertices))
    bounds = text_path.get_extents()
    return ' '.join(parts), (bounds.x0, bounds.y0, bounds.x1, bounds.y1)
