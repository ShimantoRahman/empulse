/**
 * Behaviour for the documentation landing page (see _templates/homepage.html).
 *
 * Three independent pieces, each of which degrades to something readable if it does not run:
 *
 *   * the install box's copy button, which is simply not shown as copied without it;
 *   * the hero chart, whose numbers are inlined into the page by sphinxext/homepage.py and drawn
 *     here — without the script the figure's caption still states the claim in words;
 *   * the code walkthrough, whose first step is already rendered and whose tabs are ordinary
 *     buttons, so the page never depends on the autoplay to show its code.
 *
 * No colour is named here. Every part of the chart carries a class that _static/scss/homepage.scss
 * paints from the design system's tokens, so the figure follows the reader's light/dark toggle
 * without the script knowing anything about it.
 */

(() => {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  /* -- Install command ----------------------------------------------------------------------- */

  function setUpCopyButtons() {
    document.querySelectorAll('[data-copy]').forEach((button) => {
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
          button.setAttribute('aria-label', 'Copy the install command');
        }, 1600);
      });
    });
  }

  /* -- Hero chart ---------------------------------------------------------------------------- */

  const VIEW = { width: 440, height: 260 };
  const PLOT = { left: 46, right: 430, top: 30, bottom: 212 };

  function element(name, attributes, className) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (className) node.setAttribute('class', className);
    return node;
  }

  function money(value) {
    return `€${value.toFixed(2)}`;
  }

  function drawChart(figure) {
    const dataNode = figure.querySelector('[data-eds-chart-data]');
    const svg = figure.querySelector('[data-eds-chart-svg]');
    if (!dataNode || !svg) return;

    let data;
    try {
      data = JSON.parse(dataNode.textContent);
    } catch {
      return; // Leave the written caption standing rather than drawing something wrong.
    }

    const { thresholds, costs } = data;
    if (!Array.isArray(thresholds) || thresholds.length !== costs.length) return;

    // A padded domain rather than one anchored at zero: this is a line, not a bar, and the whole
    // point of the figure is the shape of the dip. Both axes are labelled so the padding is
    // visible rather than implied. The padding is deeper below the minimum than above the
    // maximum, which is what clears the room the cost-optimal callout sits in.
    const lowest = Math.min(...costs);
    const highest = Math.max(...costs);
    const span = highest - lowest;
    const yMin = lowest - span * 0.34;
    const yMax = highest + span * 0.12;

    const x = (threshold) => PLOT.left + threshold * (PLOT.right - PLOT.left);
    const y = (cost) => PLOT.bottom - ((cost - yMin) / (yMax - yMin)) * (PLOT.bottom - PLOT.top);

    const axes = element('g', {}, 'eds-chart-axes');
    axes.appendChild(
      element('line', { x1: PLOT.left, y1: PLOT.bottom, x2: PLOT.right, y2: PLOT.bottom }, 'eds-chart-axis'),
    );
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const label = element(
        'text',
        { x: x(tick), y: PLOT.bottom + 16, 'text-anchor': tick === 0 ? 'start' : tick === 1 ? 'end' : 'middle' },
        'eds-chart-tick',
      );
      label.textContent = tick.toFixed(2);
      axes.appendChild(label);
    });
    [lowest, highest].forEach((cost) => {
      const label = element('text', { x: PLOT.left - 8, y: y(cost) + 4, 'text-anchor': 'end' }, 'eds-chart-tick');
      label.textContent = money(cost);
      axes.appendChild(label);
    });

    const axisTitle = element('text', { x: PLOT.left, y: PLOT.bottom + 34 }, 'eds-chart-label');
    axisTitle.textContent = 'decision threshold';
    axes.appendChild(axisTitle);

    const yTitle = element('text', { x: PLOT.left - 40, y: PLOT.top - 12, 'text-anchor': 'start' }, 'eds-chart-label');
    yTitle.textContent = `€ ${data.unit || 'per instance'}`;
    axes.appendChild(yTitle);
    svg.appendChild(axes);

    const points = thresholds.map((threshold, index) => `${x(threshold).toFixed(2)},${y(costs[index]).toFixed(2)}`);
    const curve = element('polyline', { points: points.join(' ') }, 'eds-chart-curve');
    svg.appendChild(curve);

    // Markers, drawn after the curve so they sit on top of it. Each callout is placed away from
    // the curve rather than centred on its dot: the cost-optimal one sits in the empty wedge
    // under the rising side, the default one in the empty space above the curve.
    const markers = element('g', {}, 'eds-chart-reveal');
    const marker = (point, kind, label, place) => {
      const px = x(point.threshold);
      const py = y(point.cost);
      markers.appendChild(element('line', { x1: px, y1: py, x2: px, y2: PLOT.bottom }, `eds-chart-${kind}-line`));
      markers.appendChild(element('circle', { cx: px, cy: py, r: 5 }, `eds-chart-${kind}-dot`));

      const text = (content, dy, className) => {
        const node = element('text', { x: px + place.dx, y: py + dy, 'text-anchor': place.anchor }, className);
        node.textContent = content;
        markers.appendChild(node);
      };
      text(money(point.cost), place.dy, 'eds-chart-value');
      text(label, place.dy + 14, `eds-chart-label${kind === 'optimal' ? ' eds-chart-label--optimal' : ''}`);
    };
    marker(data.optimal, 'optimal', `cost-optimal · ${data.optimal.threshold}`, {
      dx: 12,
      dy: 14,
      anchor: 'start',
    });
    marker(data.default, 'default', 'default 0.5', { dx: 0, dy: -26, anchor: 'middle' });
    svg.appendChild(markers);

    // The pointer read-out, hidden until the reader actually moves over the plot.
    const hover = element('g', { visibility: 'hidden' }, 'eds-chart-hover');
    const hoverLine = element('line', { y1: PLOT.top, y2: PLOT.bottom }, 'eds-chart-hover-line');
    const hoverDot = element('circle', { r: 4 }, 'eds-chart-hover-dot');
    hover.append(hoverLine, hoverDot);
    svg.appendChild(hover);

    const readout = figure.querySelector('[data-eds-chart-readout]');
    const source = figure.querySelector('[data-eds-chart-source]');
    const caption = figure.querySelector('[data-eds-chart-caption]');

    if (source && data.dataset) {
      source.textContent = `${data.dataset} · ${data.samples.toLocaleString('en')} customers`;
    }
    if (caption && typeof data.reduction === 'number') {
      caption.innerHTML =
        `Moving the decision threshold from 0.5 to ${data.optimal.threshold} reduces expected cost from ${money(data.default.cost)} ` +
        `to <strong>${money(data.optimal.cost)}</strong> per customer, a ` +
        `<strong>${Math.round(data.reduction * 100)}% reduction</strong> on the same test predictions. ` +
        `Drag across the chart to inspect any threshold.`;
    }

    if (!reduceMotion.matches) {
      const length = curve.getTotalLength ? curve.getTotalLength() : 0;
      if (length) {
        curve.style.setProperty('--eds-path-length', length);
        curve.style.strokeDasharray = length;
        curve.style.strokeDashoffset = length;
        curve.classList.add('eds-chart-curve--animated');
      }
    }

    const track = (event) => {
      const box = svg.getBoundingClientRect();
      if (!box.width) return;
      const svgX = ((event.clientX - box.left) / box.width) * VIEW.width;
      const fraction = Math.min(1, Math.max(0, (svgX - PLOT.left) / (PLOT.right - PLOT.left)));
      const index = Math.round(fraction * (thresholds.length - 1));
      const px = x(thresholds[index]);
      const py = y(costs[index]);

      hover.setAttribute('visibility', 'visible');
      hoverLine.setAttribute('x1', px);
      hoverLine.setAttribute('x2', px);
      hoverDot.setAttribute('cx', px);
      hoverDot.setAttribute('cy', py);

      if (readout) {
        readout.hidden = false;
        readout.textContent = `${thresholds[index].toFixed(2)} → ${money(costs[index])}`;
        readout.style.left = `${(px / VIEW.width) * 100}%`;
        readout.style.top = `${(py / VIEW.height) * 100}%`;
      }
    };

    const clear = () => {
      hover.setAttribute('visibility', 'hidden');
      if (readout) readout.hidden = true;
    };

    svg.addEventListener('pointermove', track);
    svg.addEventListener('pointerdown', track);
    svg.addEventListener('pointerleave', clear);
    svg.addEventListener('pointercancel', clear);
  }

  /* -- Code walkthrough ---------------------------------------------------------------------- */

  // Long enough to read a five-line snippet and its result, short enough that a reader who is
  // watching rather than reading does not lose patience. Paused whenever the reader takes over.
  const STEP_MS = 7000;

  function setUpTour(tour) {
    const tabs = Array.from(tour.querySelectorAll('[data-eds-tab]'));
    const panels = Array.from(tour.querySelectorAll('[data-eds-panel]'));
    if (tabs.length === 0 || tabs.length !== panels.length) return;

    let current = 0;
    let timer = null;
    let autoplay = !reduceMotion.matches;

    const show = (index) => {
      current = index;
      tabs.forEach((tab, position) => {
        const selected = position === index;
        tab.setAttribute('aria-selected', String(selected));
        tab.setAttribute('tabindex', selected ? '0' : '-1');
        // The progress bar is a CSS transition, and a transition only runs if the element is
        // laid out with its starting value first. Clearing the duration on every other tab
        // snaps them back so the active one always animates from zero.
        tab.style.setProperty('--eds-tab-duration', selected && autoplay ? `${STEP_MS}ms` : '0ms');
      });
      panels.forEach((panel, position) => {
        const selected = position === index;
        panel.hidden = !selected;
        panel.classList.toggle('is-active', selected);
        panel.classList.remove('is-playing');
        if (selected && !reduceMotion.matches) {
          // Force a reflow so the line animation restarts on a panel that was shown before.
          void panel.offsetWidth;
          panel.classList.add('is-playing');
        }
      });
    };

    const stop = () => {
      window.clearTimeout(timer);
      timer = null;
    };

    const schedule = () => {
      stop();
      if (!autoplay) return;
      timer = window.setTimeout(() => {
        show((current + 1) % tabs.length);
        schedule();
      }, STEP_MS);
    };

    const takeOver = (index) => {
      autoplay = false;
      stop();
      show(index);
    };

    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => takeOver(index));
      tab.addEventListener('keydown', (event) => {
        const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
        if (!step) return;
        event.preventDefault();
        const next = (index + step + tabs.length) % tabs.length;
        takeOver(next);
        tabs[next].focus();
      });
    });

    // Autoplay is a courtesy, not a demand: it pauses while the pointer or the keyboard is inside
    // the section, and never runs while the section is off screen.
    tour.addEventListener('pointerenter', stop);
    tour.addEventListener('pointerleave', () => autoplay && schedule());
    tour.addEventListener('focusin', stop);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              show(current);
              schedule();
            } else {
              stop();
            }
          });
        },
        { threshold: 0.35 },
      ).observe(tour);
    } else {
      show(0);
      schedule();
    }
  }

  /* -- Boot ---------------------------------------------------------------------------------- */

  function start() {
    setUpCopyButtons();
    document.querySelectorAll('[data-eds-chart]').forEach(drawChart);
    document.querySelectorAll('[data-eds-tour]').forEach(setUpTour);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
