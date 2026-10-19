"""Summary tables and matplotlib figures built from in-memory benchmark results."""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from .data import MODELS
from .emp_integration import DIM_N_SAMPLES, ERROR_N_SAMPLES, METHODS

INK = '#244876'
INK_MUTED = '#5C6B80'
RULE = '#C4D0DE'
PAPER = '#FFFFFF'
PRIMARY = '#1B7DBF'
PRIMARY_FILL = '#D3E7F6'
COMPARISON = '#C8622A'
COMPARISON_FILL = '#F5E1D6'

RC_PARAMS: dict[str, Any] = {
    'figure.facecolor': PAPER,
    'axes.facecolor': PAPER,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'text.color': INK,
    'axes.labelcolor': INK,
    'axes.edgecolor': RULE,
    'axes.titlecolor': INK,
    'axes.titlesize': 10,
    'axes.titleweight': 'medium',
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'grid.color': RULE,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'lines.linewidth': 1.8,
    'lines.solid_capstyle': 'round',
}

MODEL_LABEL = {'cslogit': 'CSLogit', 'cstree': 'CSTree', 'csforest': 'CSForest'}
LEGACY_LABEL = {'cslogit': 'Reference', 'cstree': 'CostCla', 'csforest': 'CostCla'}
IMPL_COLOR = {'current': PRIMARY, 'legacy': COMPARISON}
IMPL_FILL = {'current': PRIMARY_FILL, 'legacy': COMPARISON_FILL}
IMPL_MARKER_FACE = {'current': None, 'legacy': PAPER}
MODEL_MARKER = {'cslogit': 'o', 'cstree': 's', 'csforest': '^'}
MODEL_LINESTYLE = {'cslogit': 'solid', 'cstree': (0, (4, 1.5)), 'csforest': (0, (1, 1))}

METHOD_COLOR = {'Exact': PRIMARY, 'Quad': COMPARISON, 'QMC': COMPARISON, 'MC': COMPARISON}
METHOD_MARKER = {'Exact': 'o', 'Quad': 's', 'QMC': '^', 'MC': 'D'}
METHOD_LINESTYLE = {'Exact': 'solid', 'Quad': (0, (4, 1.5)), 'QMC': (0, (1, 1)), 'MC': (0, (3, 1, 1, 1))}


# ------------------------------------------------------------------------------------------------
# Training time
# ------------------------------------------------------------------------------------------------


def summarize_timing(timing: pd.DataFrame) -> pd.DataFrame:
    """Mean fit time and 95% confidence interval half-width over seeds, per (impl, model, N).

    The interval uses the Student-t distribution and is only reported from 5 seeds onwards; with
    fewer, the t multiplier (12.7 at two seeds) makes the interval uninformative.
    """

    def ci95(x: pd.Series) -> float:
        n = len(x)
        if n < 5:
            return float('nan')
        return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)

    grouped = timing.groupby(['impl', 'model', 'n_samples'])['fit_seconds']
    summary = grouped.agg(mean='mean', n='count').reset_index()
    summary['ci95'] = grouped.apply(ci95).reset_index(drop=True)
    return summary


def speedup_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean fit time of both implementations and their ratio, where both have data."""
    wide = summary.pivot_table(index=['model', 'n_samples'], columns='impl', values='mean')
    wide = wide.dropna(subset=['legacy', 'current'])
    wide['speedup'] = wide['legacy'] / wide['current']
    return wide.reset_index()[['model', 'n_samples', 'legacy', 'current', 'speedup']]


def plot_scaling(summary: pd.DataFrame) -> plt.Figure:
    """Mean training time against N on log-log axes, with 95% confidence bands."""
    with mpl.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for model in MODELS:
            for impl in ('legacy', 'current'):
                rows = summary[(summary['model'] == model) & (summary['impl'] == impl)].sort_values('n_samples')
                if rows.empty:
                    continue
                label = 'Empulse' if impl == 'current' else LEGACY_LABEL[model]
                ax.plot(
                    rows['n_samples'],
                    rows['mean'],
                    color=IMPL_COLOR[impl],
                    marker=MODEL_MARKER[model],
                    markerfacecolor=IMPL_MARKER_FACE[impl],
                    linestyle=MODEL_LINESTYLE[model],
                    markersize=4,
                    label=f'{MODEL_LABEL[model]} ({label})',
                )
                lower = (rows['mean'] - rows['ci95'].fillna(0)).clip(lower=1e-6)
                upper = rows['mean'] + rows['ci95'].fillna(0)
                ax.fill_between(rows['n_samples'], lower, upper, color=IMPL_FILL[impl], linewidth=0)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of samples (N)')
        ax.set_ylabel('Fit time (seconds)')
        ax.grid(True, which='major', linewidth=0.6)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncols=len(MODELS), bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return fig


# ------------------------------------------------------------------------------------------------
# EMP integration methods
# ------------------------------------------------------------------------------------------------


def summarize_emp_integration(size: pd.DataFrame, dimensions: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Mean execution time and relative error per method, by N and by number of stochastic variables."""
    tables = {'by size': size.pivot_table(index='n_samples', columns='method', values=['time_seconds', 'error_pct'])}
    if dimensions is not None:
        tables['by dimension'] = dimensions.pivot_table(
            index='n_stochastic', columns='method', values=['time_seconds', 'error_pct']
        )
    return tables


def _plot_method_lines(ax: plt.Axes, df: pd.DataFrame, x: str, skip: tuple[str, ...] = ()) -> None:
    mean_time = df.groupby(['method', x])['time_seconds'].mean().reset_index()
    for name in METHODS:
        rows = mean_time[mean_time['method'] == name].sort_values(x)
        if rows.empty or name in skip:
            continue
        ax.plot(
            rows[x],
            rows['time_seconds'],
            color=METHOD_COLOR[name],
            marker=METHOD_MARKER[name],
            linestyle=METHOD_LINESTYLE[name],
            markersize=5,
            label=name,
        )


def _sci_label(value: float) -> str:
    mantissa, exponent = f'{value:.1e}'.split('e')
    return rf'${mantissa} \times 10^{{{int(exponent)}}}$%'


def plot_emp_integration(size: pd.DataFrame, dimensions: pd.DataFrame | None) -> plt.Figure:
    """Accuracy (a), scaling with N (b) and, if available, scaling with the number of variables (c)."""
    with mpl.rc_context(RC_PARAMS):
        n_panels = 2 if dimensions is None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2))
        ax_error, ax_time = axes[0], axes[1]

        # The error hardly depends on N, so one size represents it. Exact is the reference, so its
        # error is zero by construction and is left out.
        error_n_samples = ERROR_N_SAMPLES if ERROR_N_SAMPLES in set(size['n_samples']) else int(size['n_samples'].max())
        error = size[size['n_samples'] == error_n_samples].groupby('method')['error_pct'].mean()
        error = error.reindex([name for name in METHODS if name != 'Exact'])
        bars = ax_error.bar(error.index, error.to_numpy(), color=COMPARISON, width=0.6)
        ax_error.bar_label(bars, labels=[_sci_label(value) for value in error], padding=3)
        ax_error.set_yscale('log')
        ax_error.set_ylim(1e-7, error.max() * 5)
        ax_error.set_xlabel('Integration method')
        ax_error.set_ylabel('Relative error vs. Exact (%)')
        ax_error.set_title(f'(a) Approximation accuracy (N = {error_n_samples:,})')
        ax_error.grid(True, which='major', axis='y', linewidth=0.6)

        _plot_method_lines(ax_time, size, 'n_samples')
        sizes = sorted(size['n_samples'].unique())
        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        ax_time.set_xticks(sizes)
        ax_time.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _p: f'{int(x) // 1000}k' if x >= 1000 else str(int(x)))
        )
        ax_time.xaxis.set_minor_locator(mticker.NullLocator())
        ax_time.set_xlabel('Number of samples (N)')
        ax_time.set_ylabel('Execution time (seconds)')
        ax_time.set_title('(b) Scaling with dataset size')
        ax_time.legend(loc='upper left', ncols=2)
        ax_time.grid(True, which='major', linewidth=0.6)

        if dimensions is not None:
            ax_dim = axes[2]
            # Exact exists only for a single stochastic variable, where panel (b) already shows it.
            _plot_method_lines(ax_dim, dimensions, 'n_stochastic', skip=('Exact',))
            ax_dim.set_yscale('log')
            dims = sorted(dimensions['n_stochastic'].unique())
            ax_dim.set_xticks(dims)
            ax_dim.set_xlim(dims[0] - 0.25, dims[-1] + 0.25)
            ax_dim.set_xlabel('Number of stochastic variables')
            ax_dim.set_ylabel('Execution time (seconds)')
            dim_n_samples = int(dimensions['n_samples'].iloc[0]) if 'n_samples' in dimensions else DIM_N_SAMPLES
            ax_dim.set_title(f'(c) Scaling with dimension (N = {dim_n_samples:,})')
            ax_dim.legend(loc='upper left')
            ax_dim.grid(True, which='major', linewidth=0.6)

        fig.tight_layout()
    return fig
