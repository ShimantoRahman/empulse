"""A ``themed-figure`` directive for illustrations that come in a light and a dark variant.

Usage::

    .. themed-figure:: cost_matrix_anatomy
        :alt: Two cost matrices side by side, one written as costs and one as benefits.
        :width: 760

        Costs and benefits are the same four numbers with the diagonal negated.

The name refers to a pair of files in ``_static/assets`` produced by ``scripts/figures/build.py``:
``<name>_light.svg`` and ``<name>_dark.svg``.

Two complete figures are emitted, each carrying pydata-sphinx-theme's ``only-light`` /
``only-dark`` class on both the figure and the image inside it. The theme switches them on the
``data-theme`` attribute, so a figure follows the reader's manual light/dark toggle rather than
only their operating-system setting.

The class has to sit on the figure as well as the image for two separate reasons. On the figure,
because the theme's ``.only-dark ~ figcaption`` rule would otherwise hide the caption of whichever
variant is showing. On the image, because the theme paints a white card behind, and applies a
brightness filter to, every ``img`` in dark mode that is not marked ``only-dark``.

``:alt:`` is required. A figure without a text alternative fails the build rather than shipping.
"""

from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util.logging import getLogger

logger = getLogger(__name__)

ASSET_DIR = Path(__file__).resolve().parents[1] / '_static' / 'assets'


class ThemedFigure(Directive):
    """Render a light/dark pair of illustrations as a single logical figure."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = True
    option_spec = {  # noqa: RUF012
        'alt': directives.unchanged_required,
        'width': directives.unchanged,
        'align': lambda arg: directives.choice(arg, ('left', 'center', 'right')),
        'name': directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        """Build one figure node per theme."""
        name = self.arguments[0].strip()
        alt = self.options.get('alt', '').strip()
        if not alt:
            raise self.error(
                f'themed-figure "{name}" needs an :alt: description of what the figure shows.'
            )

        missing = [
            variant
            for variant in ('light', 'dark')
            if not (ASSET_DIR / f'{name}_{variant}.svg').is_file()
        ]
        if missing:
            raise self.error(
                f'themed-figure "{name}" is missing the {", ".join(missing)} variant in '
                f'{ASSET_DIR}. Run `just figures` to regenerate the illustrations.'
            )

        align = self.options.get('align', 'center')
        figures: list[nodes.Node] = []
        for variant in ('light', 'dark'):
            only = f'only-{variant}'
            image = nodes.image(
                uri=f'/_static/assets/{name}_{variant}.svg',
                alt=alt,
                classes=[only],
            )
            if 'width' in self.options:
                image['width'] = self.options['width']

            figure = nodes.figure('', image, classes=[only, f'align-{align}'])
            figure['align'] = align

            if self.content:
                caption_text = nodes.Element()
                self.state.nested_parse(self.content, self.content_offset, caption_text)
                if caption_text.children and isinstance(caption_text[0], nodes.paragraph):
                    caption = nodes.caption(caption_text[0].rawsource, '',
                                            *caption_text[0].children)
                    caption.source, caption.line = self.state_machine.get_source_and_line(
                        self.lineno
                    )
                    figure += caption

            figures.append(figure)

        self.add_name(figures[0])
        return figures


def setup(app) -> dict[str, object]:  # noqa: ANN001
    """Register the directive."""
    app.add_directive('themed-figure', ThemedFigure)
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}
