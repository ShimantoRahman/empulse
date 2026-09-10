"""A very small SVG builder for the schematic figures.

Only what the diagrams in :mod:`diagrams` need: rounded boxes, text, straight and elbowed arrows,
brackets and grids. Everything is plain strings, so the generated files stay diffable and can be
hand-inspected.
"""

from __future__ import annotations

from typing import Literal
from xml.sax.saxutils import escape

from palette import FONT_STACK

Anchor = Literal['start', 'middle', 'end']
Baseline = Literal['auto', 'middle', 'hanging']


class Canvas:
    """Collects SVG elements and renders them into a scalable document."""

    def __init__(self, width: float, height: float, theme: dict[str, str]) -> None:
        self.width = width
        self.height = height
        self.theme = theme
        self._parts: list[str] = []

    def _c(self, token: str) -> str:
        """Resolve a palette token, passing through anything that is already a colour."""
        return self.theme.get(token, token)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = 'surface',
        stroke: str | None = 'rule',
        radius: float = 4,
        stroke_width: float = 1.5,
        dash: str | None = None,
    ) -> None:
        """Draw a rounded rectangle."""
        attrs = [
            f'x="{x}" y="{y}" width="{w}" height="{h}"',
            f'rx="{radius}" ry="{radius}"',
            f'fill="{self._c(fill)}"' if fill != 'none' else 'fill="none"',
        ]
        if stroke:
            attrs.append(f'stroke="{self._c(stroke)}" stroke-width="{stroke_width}"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self._parts.append(f'  <rect {" ".join(attrs)}/>')

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        fill: str = 'ink',
        size: float = 13,
        anchor: Anchor = 'middle',
        weight: str = 'normal',
        baseline: Baseline = 'middle',
        mono: bool = False,
        italic: bool = False,
    ) -> None:
        """Draw a single line of text."""
        family = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' if mono else FONT_STACK
        attrs = [
            f'x="{x}" y="{y}"',
            f'fill="{self._c(fill)}"',
            f'font-family="{family}"',
            f'font-size="{size}"',
            f'text-anchor="{anchor}"',
            f'dominant-baseline="{baseline}"',
        ]
        if weight != 'normal':
            attrs.append(f'font-weight="{weight}"')
        if italic:
            attrs.append('font-style="italic"')
        self._parts.append(f'  <text {" ".join(attrs)}>{escape(content)}</text>')

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = 'rule',
        width: float = 1.5,
        dash: str | None = None,
    ) -> None:
        """Draw a straight line."""
        attrs = [
            f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"',
            f'stroke="{self._c(stroke)}" stroke-width="{width}"',
            'stroke-linecap="round"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self._parts.append(f'  <line {" ".join(attrs)}/>')

    def path(
        self,
        d: str,
        *,
        stroke: str = 'rule',
        fill: str = 'none',
        width: float = 1.5,
        dash: str | None = None,
    ) -> None:
        """Draw an arbitrary path."""
        attrs = [
            f'd="{d}"',
            f'fill="{self._c(fill)}"' if fill != 'none' else 'fill="none"',
            f'stroke="{self._c(stroke)}" stroke-width="{width}"',
            'stroke-linecap="round"',
            'stroke-linejoin="round"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self._parts.append(f'  <path {" ".join(attrs)}/>')

    def circle(self, cx: float, cy: float, r: float, *, fill: str = 'ink') -> None:
        """Draw a filled circle."""
        self._parts.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{self._c(fill)}"/>')

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = 'rule',
        width: float = 1.5,
        head: float = 6,
        double: bool = False,
    ) -> None:
        """Draw a straight arrow with a filled head (optionally at both ends)."""
        import math

        angle = math.atan2(y2 - y1, x2 - x1)
        # Stop the shaft short of the head so the two do not overlap and thicken the tip.
        sx, sy = (x1 + head * math.cos(angle), y1 + head * math.sin(angle)) if double else (x1, y1)
        ex, ey = x2 - head * math.cos(angle), y2 - head * math.sin(angle)
        self.line(sx, sy, ex, ey, stroke=stroke, width=width)
        self._head(x2, y2, angle, head, stroke)
        if double:
            self._head(x1, y1, angle + math.pi, head, stroke)

    def _head(self, x: float, y: float, angle: float, size: float, stroke: str) -> None:
        import math

        spread = 0.42
        p1 = (x - size * math.cos(angle - spread), y - size * math.sin(angle - spread))
        p2 = (x - size * math.cos(angle + spread), y - size * math.sin(angle + spread))
        points = f'{x},{y} {p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}'
        self._parts.append(f'  <polygon points="{points}" fill="{self._c(stroke)}"/>')

    def elbow_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = 'rule',
        width: float = 1.5,
        radius: float = 8,
    ) -> None:
        """Draw an arrow that leaves horizontally, turns once, and arrives horizontally."""
        mid = (x1 + x2) / 2
        sweep_down = 1 if y2 > y1 else 0
        r = min(radius, abs(y2 - y1) / 2, abs(mid - x1))
        d = (
            f'M {x1} {y1} H {mid - r} '
            f'A {r} {r} 0 0 {sweep_down} {mid} {y1 + (r if sweep_down else -r)} '
            f'V {y2 - (r if sweep_down else -r)} '
            f'A {r} {r} 0 0 {1 - sweep_down} {mid + r} {y2} '
            f'H {x2 - 7}'
        )
        self.path(d, stroke=stroke, width=width)
        self._head(x2, y2, 0.0, 6, stroke)

    def bracket(
        self,
        x1: float,
        x2: float,
        y: float,
        *,
        depth: float = 7,
        stroke: str = 'rule',
        width: float = 1.5,
        below: bool = False,
    ) -> None:
        """Draw a square bracket spanning ``x1``..``x2``, opening towards the content."""
        tip = y + depth if below else y - depth
        self.path(
            f'M {x1} {tip} V {y} H {x2} V {tip}',
            stroke=stroke,
            width=width,
        )

    def render(self) -> str:
        """Serialise the canvas. The ``viewBox`` lets the figure scale to the article width."""
        body = '\n'.join(self._parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'width="100%" role="img">\n{body}\n</svg>\n'
        )
