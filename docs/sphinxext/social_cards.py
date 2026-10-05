"""Draw the social preview card of every page in the Empulse design language.

``sphinxext-opengraph`` ships a Matplotlib card with its own font and layout. This extension
replaces its renderer, so the cards keep the extension's metadata, hashing and ``<meta>`` tags but
are painted from the tokens in ``DESIGN.md``: the ``tint`` canvas with wind streaks, the cost-matrix
mark and wordmark, Shorai Sans text and JetBrains Mono labels.

The fonts are the self-hosted web fonts in ``_static/fonts``. Pillow cannot read WOFF2, so they are
unpacked once per build to TrueType files in a temporary directory.
"""

from __future__ import annotations

import hashlib
import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFont

if TYPE_CHECKING:
    from sphinx.application import Sphinx

FONT_DIR = Path(__file__).resolve().parent.parent / '_static' / 'fonts'
SANS_WOFF2 = 'ShoraiSansStdNVar-Latin-Plus-Symbols.woff2'
MONO_WOFF2 = 'jetbrains-mono-variable.woff2'

WIDTH, HEIGHT = 1146, 600  # the size sphinxext-opengraph announces in og:image:width / height
SCALE = 2  # drawn at twice the size and scaled down, which smooths the shapes and the text

TINT = '#EDF2F6'
INK = '#244876'
INK_MUTED = '#5C6B80'
BLUE_STRONG = '#1B7DBF'
BLUE_SOFT = '#D3E7F6'
PURPLE_STRONG = '#7F4ACF'
RULE = '#C4D0DE'
STREAK = (255, 255, 255, 242)

MARGIN = 72
TEXT_WIDTH = 900
EYEBROW = 'cost-sensitive learning for scikit-learn'

# side, offset from that side, top, width, height, faint?
WIND = (
    ('right', -20, 52, 470, 30, False),
    ('right', -60, 128, 330, 22, True),
    ('right', 40, 232, 250, 26, True),
    ('right', -40, 470, 360, 30, False),
    ('left', -40, 548, 330, 24, True),
)


@cache
def _ttf(woff2_name: str) -> str:
    font = TTFont(FONT_DIR / woff2_name)
    font.flavor = None
    directory = Path(tempfile.gettempdir()) / 'empulse_social_fonts'
    directory.mkdir(exist_ok=True)
    path = directory / (Path(woff2_name).stem + '.ttf')
    font.save(path)
    return str(path)


@cache
def _font(woff2_name: str, size: int, weight: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(_ttf(woff2_name), size)
    font.set_variation_by_axes([weight])
    return font


def _sans(size: float, weight: int) -> ImageFont.FreeTypeFont:
    return _font(SANS_WOFF2, round(size * SCALE), weight)


def _mono(size: float, weight: int) -> ImageFont.FreeTypeFont:
    return _font(MONO_WOFF2, round(size * SCALE), weight)


def _advance(font: ImageFont.FreeTypeFont, previous: str, char: str) -> float:
    """Width of ``char`` including its kerning against ``previous``."""
    if not previous:
        return font.getlength(char)
    return font.getlength(previous + char) - font.getlength(previous)


def _text_width(text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
    width = 0.0
    previous = ''
    for char in text:
        width += _advance(font, previous, char) + tracking
        previous = char
    return width - tracking if text else 0.0


def _draw_text(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    tracking: float = 0.0,
) -> None:
    previous = ''
    for char in text:
        x += _advance(font, previous, char)
        draw.text((x - font.getlength(char), y), char, font=font, fill=fill, anchor='ls')
        x += tracking
        previous = char


def _wrap(text: str, font: ImageFont.FreeTypeFont, tracking: float, width: float) -> list[str]:
    lines: list[str] = []
    line = ''
    for word in text.split():
        candidate = f'{line} {word}'.strip()
        if line and _text_width(candidate, font, tracking) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _ellipsise(lines: list[str], limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    return [*lines[: limit - 1], lines[limit - 1].rstrip('.,;: ') + '...']


def _wind(canvas: Image.Image) -> None:
    for side, offset, top, width, height, faint in WIND:
        w, h = width * SCALE, height * SCALE
        gradient = Image.linear_gradient('L').rotate(90, expand=True).resize((w, h))
        if side == 'left':
            gradient = gradient.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        # Fades in towards the page: transparent at the outer end, full at 72% of the length.
        gradient = gradient.point(lambda v: min(255, round(v / 0.72)))
        pill = Image.new('L', (w, h), 0)
        ImageDraw.Draw(pill).rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=255)
        alpha = ImageChops.multiply(gradient, pill).point(lambda v: round(v * (0.55 if faint else 1.0)))
        layer = Image.new('RGBA', (w, h), STREAK[:3] + (255,))
        layer.putalpha(alpha.point(lambda v: round(v * STREAK[3] / 255)))
        x = offset * SCALE if side == 'left' else canvas.width - w - offset * SCALE
        canvas.alpha_composite(layer, (x, top * SCALE))


def _logo(draw: ImageDraw.ImageDraw, x: float, baseline: float, size: float) -> None:
    """The cost-matrix mark and the lowercase wordmark, built as in the documentation header."""
    font = _sans(size, 600)
    mark = size * 0.92 * SCALE
    unit = mark / 32
    top = baseline - size * SCALE * 0.70
    for cx, cy, fill in ((3, 3, BLUE_SOFT), (17, 3, BLUE_STRONG), (3, 17, BLUE_STRONG), (17, 17, BLUE_SOFT)):
        draw.rounded_rectangle(
            (x + cx * unit, top + cy * unit, x + (cx + 12) * unit, top + (cy + 12) * unit),
            radius=3.2 * unit,
            fill=fill,
        )
    dot_x, dot_y, radius = x + 9 * unit, top + 9 * unit, 2.8 * unit
    draw.ellipse((dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius), fill=PURPLE_STRONG)
    _draw_text(draw, x + mark + 0.32 * size * SCALE, baseline, 'empulse', font, INK, -0.035 * size * SCALE)


def render_card(title: str, description: str, url_text: str) -> Image.Image:
    canvas = Image.new('RGBA', (WIDTH * SCALE, HEIGHT * SCALE), TINT)
    _wind(canvas)
    draw = ImageDraw.Draw(canvas)

    _logo(draw, MARGIN * SCALE, 108 * SCALE, 46)

    for size in (78, 70, 62, 54):
        font = _sans(size, 600)
        tracking = -0.04 * size * SCALE
        title_lines = _wrap(title, font, tracking, TEXT_WIDTH * SCALE)
        if len(title_lines) <= 2 or size == 54:
            break
    title_lines = _ellipsise(title_lines, 3)
    line_height = size * 1.06 * SCALE
    y = 216 * SCALE
    for line in title_lines:
        _draw_text(draw, MARGIN * SCALE, y, line, font, INK, tracking)
        y += line_height

    body = _sans(28, 400)
    y += (10 - 24) * SCALE + 28 * SCALE
    for line in _ellipsise(_wrap(description, body, 0, TEXT_WIDTH * SCALE), 3):
        _draw_text(draw, MARGIN * SCALE, y, line, body, INK_MUTED)
        y += 28 * 1.5 * SCALE

    rule_y = 512 * SCALE
    draw.line((MARGIN * SCALE, rule_y, (WIDTH - MARGIN) * SCALE, rule_y), fill=RULE, width=SCALE)
    _draw_text(draw, MARGIN * SCALE, 556 * SCALE, url_text, _mono(22, 400), INK_MUTED)
    label = _mono(17, 600)
    tracking = 0.09 * 17 * SCALE
    text = EYEBROW.upper()
    right_aligned_x = (WIDTH - MARGIN) * SCALE - _text_width(text, label, tracking)
    _draw_text(draw, right_aligned_x, 554 * SCALE, text, label, BLUE_STRONG, tracking)

    return canvas.convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def create_social_card(
    config_social: dict[str, Any],
    site_name: str,
    page_title: str,
    description: str,
    url_text: str,
    page_path: str,
    *,
    srcdir: str | Path,
    outdir: str | Path,
    **_: Any,
) -> Path:
    digest = hashlib.sha1(
        f'{site_name}{page_title}{description}{url_text}{config_social}{Path(__file__).read_bytes()!r}'.encode(),
        usedforsecurity=False,
    ).hexdigest()[:8]
    relative = Path('_images/social_previews') / f'summary_{page_path.replace("/", "_")}_{digest}.png'
    target = Path(outdir) / relative
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        text = description or config_social.get('default_description', '')
        render_card(page_title, text, url_text).save(target, optimize=True)
    return relative


def setup(app: Sphinx) -> dict[str, Any]:
    import sphinxext.opengraph as opengraph

    opengraph.create_social_card = create_social_card
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}
