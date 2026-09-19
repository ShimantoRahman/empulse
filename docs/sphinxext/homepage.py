"""Render ``index`` as a custom landing page instead of as an article.

The rest of the documentation is prose, and pydata-sphinx-theme's article layout is exactly right
for it. The homepage is not prose: it has one job, which is to tell a first-time reader what
Empulse is for and hand them a next step. So ``index`` is rendered from ``_templates/homepage.html``
instead, which keeps the theme's header, footer and search but replaces the article column with a
full-width layout of its own.

What this extension does:

* swaps the template for ``index`` only, leaving every other page untouched;
* hands the template the copy from ``homepage_content.py``, with the code snippets already
  highlighted by Pygments, so the homepage needs no client-side highlighter and its code matches
  every other code block on the site;
* resolves the ``:ref:`` labels and document names the content module names into real URLs through
  Sphinx's own machinery, so a renamed label fails the ``-W`` build rather than shipping a dead
  link;
* looks up the download count and star count that the README already advertises.

The counts are fetched once per build and every failure is non-fatal: the stat is simply left off
the page. Set ``EMPULSE_DOCS_OFFLINE=1`` to skip the requests entirely, which is what you want when
iterating on the page itself.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import requests
from docutils import nodes
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from sphinx.application import Sphinx
from sphinx.util.logging import getLogger

import homepage_content as content

logger = getLogger(__name__)

HOMEPAGE = 'index'
TEMPLATE = 'homepage.html'

HERE = Path(__file__).resolve().parents[1]
HERO_DATA = HERE / '_static' / 'data' / 'homepage_hero.json'

GITHUB_REPO = 'ShimantoRahman/empulse'
GITHUB_API = f'https://api.github.com/repos/{GITHUB_REPO}'
# The same source the README's download badge uses. pepy's JSON API needs a key; the badge does
# not, and it already carries the rounded number ("54k") this page wants to print.
PEPY_BADGE = 'https://static.pepy.tech/badge/empulse'
TIMEOUT = 10

# `nowrap` because the template supplies the `<div class="highlight"><pre>` wrapper itself. The
# token classes are Pygments' standard ones, which is the point: Sphinx already writes CSS for
# them in both themes, so homepage code is coloured by the same rules as code anywhere else.
FORMATTER = HtmlFormatter(nowrap=True)


def _format_count(value: int) -> str:
    """Render a count the way a badge would: 942, 5.4k, 54k, 1.2M."""
    if value < 1000:
        return str(value)
    if value < 10_000:
        return f'{value / 1000:.1f}k'.replace('.0k', 'k')
    if value < 1_000_000:
        return f'{round(value / 1000)}k'
    return f'{value / 1_000_000:.1f}M'.replace('.0M', 'M')


def _github_stars() -> str | None:
    """The repository's star count, or ``None`` if GitHub cannot be reached."""
    response = requests.get(GITHUB_API, timeout=TIMEOUT)
    response.raise_for_status()
    return _format_count(int(response.json()['stargazers_count']))


def _latest_release() -> str | None:
    """The newest release tag published on GitHub."""
    response = requests.get(f'{GITHUB_API}/releases/latest', timeout=TIMEOUT)
    response.raise_for_status()
    return str(response.json()['tag_name'])


def _pypi_downloads() -> str | None:
    """The total download count, read out of pepy's badge.

    The badge is an SVG whose last ``<text>`` element holds the number. That is a scrape rather
    than an API call, so it is wrapped like one: any change in shape raises, and the caller drops
    the stat instead of failing the build.
    """
    response = requests.get(PEPY_BADGE, timeout=TIMEOUT)
    response.raise_for_status()
    labels = re.findall(r'<text[^>]*>([^<]+)</text>', response.text)
    if not labels:
        raise ValueError('no <text> elements in the pepy badge')
    return labels[-1].strip()


def _fetch_stats(release: str) -> list[dict[str, str]]:
    """The social-proof row: whatever of it could be looked up, plus what is known locally.

    ``release`` is the version of the checkout being built, which on the development branch is
    ahead of anything a reader can install. The row advertises the *latest release*, so on a dev
    build the number has to be looked up rather than taken from here — the same distinction
    ``generate_versions_json`` draws in ``conf.py``. If that lookup cannot be made, the stat is
    left off instead of printing a version nobody can ``pip install``.
    """
    stats: list[dict[str, str]] = []
    pypi = 'https://pypi.org/project/empulse/'

    lookups = [
        ('downloads', 'https://pepy.tech/projects/empulse', _pypi_downloads),
        ('GitHub stars', f'https://github.com/{GITHUB_REPO}', _github_stars),
    ]
    # A tagged build is itself the latest release, so it needs no lookup. Only a dev build does.
    if 'dev' in release:
        lookups.append(('latest release', pypi, _latest_release))

    if os.environ.get('EMPULSE_DOCS_OFFLINE') != '1':
        for label, url, lookup in lookups:
            try:
                value = lookup()
            except Exception as error:  # noqa: BLE001 - a missing stat must not fail a docs build
                logger.info(f'homepage: could not look up {label} ({error}); leaving it off the page')
                continue
            if value:
                stats.append({'value': value, 'label': label, 'url': url})

    if 'dev' not in release:
        stats.append({'value': release.split('+')[0], 'label': 'latest release', 'url': pypi})

    # Tags are published both with and without a leading `v`; the page prints one of them.
    for stat in stats:
        if stat['label'] == 'latest release':
            stat['value'] = f'v{stat["value"].lstrip("v")}'

    stats.append({'value': '3.11+', 'label': 'Python', 'url': pypi})
    return stats


class _Resolver:
    """Turns the labels and document names in ``homepage_content`` into URLs relative to ``index``.

    Both kinds go through Sphinx rather than being written out by hand, so a page that moves or a
    label that is renamed is reported as a warning — which the ``-W`` documentation build turns
    into a failure — instead of becoming a 404 on the busiest page of the site.
    """

    def __init__(self, app: Sphinx, pagename: str) -> None:
        self.app = app
        self.pagename = pagename

    def ref(self, label: str) -> str:
        """The URL of a ``.. _label:`` target."""
        labels = self.app.env.domains['std'].labels
        if label not in labels:
            logger.warning(f'homepage: unknown :ref: label {label!r}', type='homepage')
            return ''
        docname, labelid, _ = labels[label]
        uri = self.app.builder.get_relative_uri(self.pagename, docname)
        return f'{uri}#{labelid}' if labelid else uri

    def doc(self, docname: str) -> str:
        """The URL of a document."""
        if docname not in self.app.env.all_docs:
            logger.warning(f'homepage: unknown document {docname!r}', type='homepage')
            return ''
        return self.app.builder.get_relative_uri(self.pagename, docname)


def _highlight(code: str) -> str:
    """Colour a snippet with Pygments and wrap each line so it can be revealed in turn.

    The walkthrough types its code in one line at a time, which needs a per-line element to hang
    the delay on. The wrapping happens here rather than in the browser because splitting coloured
    HTML on newlines is only safe when no ``<span>`` crosses one — true for every snippet on the
    page, and checked below so that adding a triple-quoted string to one fails the build instead
    of shipping broken markup.
    """
    # Pygments always ends its output with a newline, which would otherwise become a blank line.
    lines = highlight(code.strip('\n'), PythonLexer(), FORMATTER).rstrip('\n').split('\n')
    for number, line in enumerate(lines, start=1):
        if line.count('<span') != line.count('</span>'):
            raise ValueError(
                f'homepage snippet line {number} has a highlight span crossing a line break, which '
                f'the per-line reveal cannot wrap: {line!r}'
            )
    # Joined by real newlines, which is what a copied selection and a screen reader read. The spans
    # are `inline-block` rather than `block` for exactly this reason: a block would supply the line
    # break itself and turn each of these newlines into a second, empty line. An empty line stays
    # an empty span, sized by `min-height` in the stylesheet, so a copied snippet carries no
    # invisible filler character.
    return '\n'.join(
        f'<span class="eds-line" style="--eds-line-index: {index}">{line}</span>'
        for index, line in enumerate(lines)
    )


def _build_context(app: Sphinx, pagename: str) -> dict[str, Any]:
    """Everything ``homepage.html`` needs, with refs resolved and code highlighted."""
    resolve = _Resolver(app, pagename)

    return {
        'hero_eyebrow': content.EYEBROW,
        'hero_motto': content.MOTTO,
        'hero_lead': content.LEAD,
        'install_command': content.INSTALL_COMMAND,
        # Resolved rather than written with `pathto` so that a page which moves is reported by the
        # `-W` build instead of turning into a 404 on the busiest page of the site.
        'url_getting_started': resolve.doc('getting_started'),
        'url_tutorial': resolve.doc('tutorial'),
        'url_installation': resolve.doc('getting_started/installation'),
        'hero_data': HERO_DATA.read_text(encoding='utf-8').strip() if HERO_DATA.is_file() else '',
        'stats': app.env.empulse_homepage_stats,
        'pillars': [{**pillar._asdict(), 'url': resolve.ref(pillar.link_ref)} for pillar in content.PILLARS],
        'setup_caption': content.SETUP_CAPTION,
        'steps': [
            {
                **step._asdict(),
                'html': _highlight(step.code),
                'lines': len(step.code.strip('\n').splitlines()),
                'url': resolve.ref(step.link_ref),
            }
            for step in content.STEPS
        ],
        'destinations': [
            {**destination._asdict(), 'url': resolve.doc(destination.ref)} for destination in content.DESTINATIONS
        ],
    }


def _collect_stats(app: Sphinx) -> None:
    """Look the counts up once, before any page is written."""
    app.env.empulse_homepage_stats = _fetch_stats(app.config.release)


def _use_homepage_template(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> str | None:
    """Swap the template and fill the context, for ``index`` only."""
    if pagename != HOMEPAGE:
        return None
    if not HERO_DATA.is_file():
        logger.warning(
            f'homepage: {HERO_DATA.name} is missing, so the hero chart will be empty; run `just figures`',
            type='homepage',
        )
    context.update(_build_context(app, pagename))
    return TEMPLATE


def _homepage_is_always_outdated(app: Sphinx, env: Any, added: set[str], changed: set[str], removed: set[str]) -> list[str]:
    """Rebuild the homepage on every build.

    Nothing Sphinx tracks changes when ``homepage_content.py`` or the hero data does, because
    ``index.rst`` holds none of it — so an incremental build would keep serving the previous
    wording. Its content is also partly looked up from the network, which is stale the moment it
    is written. Rebuilding one page every time costs nothing and removes both problems.
    """
    return [HOMEPAGE] if HOMEPAGE in env.all_docs else []


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the homepage renderer."""
    app.connect('builder-inited', _collect_stats)
    app.connect('env-get-outdated', _homepage_is_always_outdated)
    app.connect('html-page-context', _use_homepage_template)
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}
