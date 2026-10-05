/**
 * Behaviour for the documentation landing page (see _templates/homepage.html).
 *
 * Independent pieces, each of which leaves a complete page behind if it does not run:
 *
 *   * the copy buttons beside the install command;
 *   * the signature's profit chart. Its headline and both model rows are already printed by the
 *     template from the same JSON, so without this script a reader still sees the numbers; the
 *     script adds the chart, the threshold control and the switch between tuned and default;
 *   * the quick tour's step tabs, which are ordinary buttons over panels that are all in the page.
 *   * the three bento figures: the cost matrix's readout, the threshold over example customers,
 *     and the dataset tiles' loader line. Each is drawn by the template and only made responsive
 *     here.
 *
 * No colour is named here. Every mark the chart draws carries a class that
 * _static/scss/homepage.scss paints from the design system's tokens, so the figure follows the
 * reader's light/dark toggle without the script knowing anything about it.
 */

(() => {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';

  /* -- Copy buttons -------------------------------------------------------------------------- */

  function setUpCopyButtons() {
    document.querySelectorAll('.eds-home [data-copy]').forEach((button) => {
      const label = button.getAttribute('aria-label');
      button.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(button.dataset.copy);
        } catch {
          return; // Clipboard denied or unavailable; the command is on screen to select anyway.
        }
        button.classList.add('is-copied');
        button.setAttribute('aria-label', 'Copied');
        window.setTimeout(() => {
          button.classList.remove('is-copied');
          button.setAttribute('aria-label', label);
        }, 1600);
      });
    });
  }

  /* -- Signature ------------------------------------------------------------------------------- */

  const VIEW = { width: 560, height: 250 };
  const PLOT = { left: 46, right: 546, top: 14, bottom: 220 };
  const KEYS = ['baseline', 'cost_sensitive'];

  function node(name, attributes, parent) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (parent) parent.appendChild(element);
    return element;
  }

  function euro(value) {
    const sign = value < 0 ? '−' : '';
    return `${sign}€${Math.abs(Math.round(value)).toLocaleString('en-US')}`;
  }

  function signedEuro(value) {
    return (value < 0 ? '−' : '+') + euro(Math.abs(value));
  }

  function shortEuro(value) {
    if (value === 0) return '€0';
    return `${value < 0 ? '−' : ''}€${Math.abs(value) / 1000}k`;
  }

  /** A "nice" tick step that splits `span` into roughly `count` intervals. */
  function tickStep(span, count) {
    const raw = span / count;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const residual = raw / magnitude;
    const nice = residual >= 5 ? 10 : residual >= 2 ? 5 : residual >= 1 ? 2 : 1;
    return nice * magnitude;
  }

  function setUpSignature(figure) {
    const dataNode = figure.querySelector('[data-eds-sig-data]');
    if (!dataNode) return;
    let data;
    try {
      data = JSON.parse(dataNode.textContent);
    } catch {
      return; // The rows the template printed stay as they are.
    }

    const svg = figure.querySelector('[data-eds-sig-svg]');
    const scrub = figure.querySelector('[data-eds-sig-scrub]');
    const plot = figure.querySelector('[data-eds-sig-plot]');
    const modes = figure.querySelector('[data-eds-sig-modes]');
    const hint = figure.querySelector('[data-eds-sig-hint]');
    const hintText = figure.querySelector('[data-eds-sig-hint-text]');
    const delta = figure.querySelector('[data-eds-sig-delta]');
    const note = figure.querySelector('[data-eds-sig-note]');
    const thresholds = data.thresholds;
    const last = thresholds.length - 1;
    const models = data.models;

    // The y axis spans the interesting region: from just above the best profit down to a modest
    // loss. Contacting nearly everyone loses far more than that, and those tails are clipped by
    // the plot rather than squashing every other value against the axis.
    const best = Math.max(...KEYS.flatMap((key) => models[key].profit));
    const step = tickStep(best, 2);
    const yMax = Math.ceil((best * 1.08) / step) * step;
    const yMin = -yMax * 1.1;

    const x = (index) => PLOT.left + (index / last) * (PLOT.right - PLOT.left);
    const y = (value) => {
      const clamped = Math.max(yMin - (yMax - yMin) * 0.1, Math.min(yMax, value));
      return PLOT.top + (1 - (clamped - yMin) / (yMax - yMin)) * (PLOT.bottom - PLOT.top);
    };

    // Axes and grid.
    const clip = node('clipPath', { id: 'eds-sig-clip' }, node('defs', {}, svg));
    node('rect', { x: PLOT.left, y: PLOT.top, width: PLOT.right - PLOT.left, height: PLOT.bottom - PLOT.top }, clip);
    const grid = node('g', { class: 'eds-grid' }, svg);
    const axis = node('g', { class: 'eds-axis' }, svg);
    for (let value = -Math.floor(-yMin / step) * step; value <= yMax; value += step) {
      if (value !== 0) node('line', { x1: PLOT.left, x2: PLOT.right, y1: y(value), y2: y(value) }, grid);
      node('text', { x: PLOT.left - 8, y: y(value) + 3.5, 'text-anchor': 'end' }, axis).textContent = shortEuro(value);
    }
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const anchor = tick === 0 ? 'start' : tick === 1 ? 'end' : 'middle';
      const label = node('text', { x: x(tick * last), y: VIEW.height - 8, 'text-anchor': anchor }, axis);
      label.textContent = tick % 0.5 === 0 ? tick.toFixed(1) : tick.toFixed(2);
    });
    node('line', { class: 'eds-zero', x1: PLOT.left, x2: PLOT.right, y1: y(0), y2: y(0) }, svg);

    // The two curves, each labelled directly rather than through a legend.
    const curves = node('g', { 'clip-path': 'url(#eds-sig-clip)' }, svg);
    KEYS.forEach((key) => {
      const d = models[key].profit.map((value, index) => `${index ? 'L' : 'M'}${x(index).toFixed(1)} ${y(value).toFixed(1)}`);
      node('path', { class: `eds-curve--${key}`, d: d.join('') }, curves);
    });
    const csTuned = models.cost_sensitive.tuned_index;
    // Label the cost-sensitive curve just under its lowest point across the label's width, so the
    // text never sits on the line.
    const csLabelIndex = Math.round(last * 0.24);
    const underLabel = models.cost_sensitive.profit.slice(csLabelIndex, csLabelIndex + Math.round(last * 0.18));
    node('text', {
      class: 'eds-label eds-label--cost_sensitive',
      x: x(csLabelIndex),
      y: y(Math.min(...underLabel)) + 18,
    }, svg).textContent = 'cost-sensitive';
    const flat = models.baseline.contacted.findIndex((count) => count === 0);
    const baseLabelIndex = Math.max(flat, Math.round(last * 0.45));
    node('text', { class: 'eds-label eds-label--baseline', x: x(baseLabelIndex), y: y(0) + 16 }, svg).textContent =
      flat >= 0 ? 'standard: contacts no one' : 'standard';

    const scrubLine = node('line', { class: 'eds-scrub-line', y1: PLOT.top, y2: PLOT.bottom }, svg);
    const scrubTag = node('text', { class: 'eds-scrub-tag', y: PLOT.top + 10 }, svg);
    const points = Object.fromEntries(KEYS.map((key) => [key, node('circle', { class: 'eds-point', r: 5.5 }, svg)]));

    const rows = Object.fromEntries(
      KEYS.map((key) => {
        const row = figure.querySelector(`[data-eds-sig-row="${key}"]`);
        return [
          key,
          {
            value: row.querySelector('[data-eds-sig-value]'),
            meta: row.querySelector('[data-eds-sig-meta]'),
            bar: row.querySelector('[data-eds-sig-bar]'),
          },
        ];
      }),
    );

    function render(indices, message) {
      const profits = KEYS.map((key) => models[key].profit[indices[key]]);
      const scale = Math.max(...profits, 1);
      KEYS.forEach((key, i) => {
        const index = indices[key];
        rows[key].value.textContent = euro(profits[i]);
        rows[key].meta.textContent =
          `threshold ${thresholds[index].toFixed(2)} · ${models[key].contacted[index].toLocaleString('en-US')} contacted`;
        rows[key].bar.style.width = `${(Math.max(profits[i], 0) / scale) * 100}%`;
        points[key].setAttribute('cx', x(index));
        points[key].setAttribute('cy', y(models[key].profit[index]));
      });
      delta.textContent = signedEuro(profits[1] - profits[0]);
      note.textContent = message;

      const shared = indices.baseline === indices.cost_sensitive;
      scrubLine.style.display = shared ? '' : 'none';
      scrubTag.style.display = shared ? '' : 'none';
      if (shared) {
        const index = indices.baseline;
        scrubLine.setAttribute('x1', x(index));
        scrubLine.setAttribute('x2', x(index));
        const right = index > last * 0.8;
        scrubTag.setAttribute('x', x(index) + (right ? -7 : 7));
        scrubTag.setAttribute('text-anchor', right ? 'end' : 'start');
        scrubTag.textContent = `t = ${thresholds[index].toFixed(2)}`;
      }
    }

    const buttons = modes.querySelectorAll('button');
    function setMode(mode) {
      buttons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
      if (mode === 'tuned') {
        scrub.value = String(csTuned);
        hintText.textContent = 'Drag across the chart to try any threshold';
        render(
          { baseline: models.baseline.tuned_index, cost_sensitive: csTuned },
          'more profit than the tuned baseline',
        );
      } else {
        const index = data.default_index;
        scrub.value = String(index);
        hintText.textContent = `Both models at the default threshold of ${thresholds[index].toFixed(1)}`;
        render({ baseline: index, cost_sensitive: index }, 'more profit at the default threshold');
      }
    }

    scrub.max = String(last);
    scrub.addEventListener('input', () => {
      const index = Number(scrub.value);
      buttons.forEach((button) => button.setAttribute('aria-pressed', 'false'));
      hintText.textContent = `Both models at threshold ${thresholds[index].toFixed(2)}`;
      render({ baseline: index, cost_sensitive: index }, 'more profit at this threshold');
    });
    buttons.forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));

    plot.hidden = false;
    modes.hidden = false;
    hint.hidden = false;
    setMode('tuned');
  }

  /* -- Quick tour ------------------------------------------------------------------------------ */

  function setUpTour(section) {
    const tabs = Array.from(section.querySelectorAll('[data-eds-tab]'));
    const panels = Array.from(section.querySelectorAll('[data-eds-panel]'));

    function select(index, focus) {
      tabs.forEach((tab, i) => {
        const selected = i === index;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
        panels[i].hidden = !selected;
      });
      if (focus) tabs[index].focus();
    }

    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(index, false));
      tab.addEventListener('keydown', (event) => {
        const forward = event.key === 'ArrowDown' || event.key === 'ArrowRight';
        const backward = event.key === 'ArrowUp' || event.key === 'ArrowLeft';
        if (!forward && !backward) return;
        event.preventDefault();
        select((index + (forward ? 1 : tabs.length - 1)) % tabs.length, true);
      });
    });
  }

  /* -- Bento: the cost matrix ------------------------------------------------------------------- */

  // Pointing at or focusing a cell names its outcome in the readout under the matrix.
  function setUpMatrix(card) {
    const cells = card.querySelectorAll('[data-eds-cell]');
    const name = card.querySelector('[data-eds-matrix-name]');
    const note = card.querySelector('[data-eds-matrix-note]');
    const show = (cell) => {
      cells.forEach((other) => other.classList.toggle('is-active', other === cell));
      name.textContent = cell.dataset.name;
      note.textContent = cell.dataset.note;
    };
    cells.forEach((cell) => {
      cell.addEventListener('mouseenter', () => show(cell));
      cell.addEventListener('focus', () => show(cell));
      cell.addEventListener('click', () => show(cell));
    });
  }

  /* -- Bento: the threshold -------------------------------------------------------------------- */

  // Everyone at or right of the line is contacted. Priced with the matrix on the measuring card:
  // reaching a churner keeps the €200 they would take with them, and every loyal customer reached
  // costs a €10 incentive.
  const SAVED_PER_CHURNER = 200;
  const COST_PER_STAYER = 10;
  const AXIS = { left: 16, width: 368 };

  function setUpDecide(card) {
    const dots = Array.from(card.querySelectorAll('.eds-cutoff__dot')).map((dot) => ({
      dot,
      score: Number(dot.dataset.score),
      churns: dot.dataset.churns === '1',
    }));
    const line = card.querySelector('[data-eds-decide-line]');
    const scrub = card.querySelector('[data-eds-decide-scrub]');
    const readout = card.querySelector('[data-eds-decide-readout]');
    const x = (threshold) => AXIS.left + threshold * AXIS.width;

    const profitAt = (threshold) =>
      dots.reduce((total, d) => {
        if (d.score < threshold) return total;
        return total + (d.churns ? SAVED_PER_CHURNER : -COST_PER_STAYER);
      }, 0);

    // The best line sits just left of the lowest-scoring customer worth contacting.
    const candidates = [1, ...dots.map((d) => d.score)];
    const optimum = candidates.reduce((top, t) => (profitAt(t) > profitAt(top) ? t : top), 1);

    function render(threshold) {
      let contacted = 0;
      let churners = 0;
      dots.forEach((d) => {
        const reached = d.score >= threshold;
        d.dot.classList.toggle('is-contacted', reached);
        contacted += reached ? 1 : 0;
        churners += reached && d.churns ? 1 : 0;
      });
      line.setAttribute('x1', x(threshold));
      line.setAttribute('x2', x(threshold));
      const profit = profitAt(threshold);
      const sign = profit < 0 ? '−' : '';
      readout.innerHTML =
        `Contact <b>${contacted}</b> · reach ${churners} ${churners === 1 ? 'churner' : 'churners'} · ` +
        `profit <b>${sign}€${Math.abs(profit).toLocaleString('en-US')}</b>` +
        (profit === profitAt(optimum) ? ' · the most profitable line' : '');
    }

    scrub.addEventListener('input', () => render(Number(scrub.value) / 100));
    render(Number(scrub.value) / 100);
  }

  /* -- Bento: the datasets --------------------------------------------------------------------- */

  // Pointing at or focusing a tile puts the call that loads that dataset in the line below it.
  function setUpShelf(card) {
    const tiles = card.querySelectorAll('[data-loader]');
    const call = card.querySelector('[data-eds-shelf-call]');
    const loader = card.querySelector('[data-eds-shelf-loader]');
    const show = (tile) => {
      tiles.forEach((other) => other.classList.toggle('is-active', other === tile));
      loader.textContent = tile.dataset.loader;
      if (tile.dataset.loaderUrl) call.href = tile.dataset.loaderUrl;
    };
    tiles.forEach((tile) => {
      tile.addEventListener('mouseenter', () => show(tile));
      tile.addEventListener('focus', () => show(tile));
    });
    if (tiles.length) show(tiles[0]);
  }

  function start() {
    setUpCopyButtons();
    document.querySelectorAll('[data-eds-sig]').forEach(setUpSignature);
    document.querySelectorAll('[data-eds-tour]').forEach(setUpTour);
    document.querySelectorAll('[data-eds-matrix]').forEach(setUpMatrix);
    document.querySelectorAll('[data-eds-decide]').forEach(setUpDecide);
    document.querySelectorAll('[data-eds-shelf]').forEach(setUpShelf);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
