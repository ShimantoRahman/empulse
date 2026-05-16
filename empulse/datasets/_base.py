from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic

from narwhals.typing import IntoDataFrameT, IntoSeriesT

if TYPE_CHECKING:
    import numpy as np

    from ..metrics.metric.cost_matrix import CostMatrix


def get_data_home(data_home: str | Path | None = None) -> Path:
    """Return the path to the empulse data directory.

    By default, this is ``~/empulse_data``.
    The directory is created if it does not exist.

    The path can be overridden by setting the ``EMPULSE_DATA_HOME``
    environment variable.

    Parameters
    ----------
    data_home : str or Path, optional
        Explicit path to the data directory. Overrides ``EMPULSE_DATA_HOME``.

    Returns
    -------
    data_home : Path
    """
    if data_home is None:
        data_home = os.environ.get('EMPULSE_DATA_HOME', Path.home() / 'empulse_data')
    data_home = Path(data_home)
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


@dataclass(frozen=True)
class Dataset(Generic[IntoDataFrameT, IntoSeriesT]):
    """
    Container object for datasets returned by the load / fetch functions.

    Attributes
    ----------
    data : :class:`pandas:pandas.DataFrame`, :class:`numpy:numpy.ndarray`, or \
           any dataframe supported by narwhals
        Feature matrix.
    target : :class:`pandas:pandas.Series`, :class:`numpy:numpy.ndarray`, or \
             any series supported by narwhals
        Binary classification labels.
    cost_matrix : :class:`~empulse.metrics.CostMatrix`
        Symbolic cost matrix with default values pre-filled via
        :meth:`~empulse.metrics.CostMatrix.set_default`.
        The formula only uses deterministic variables.
    instance_costs : dict[str, numpy.ndarray] or None
        Per-instance cost drivers (e.g. ``{'clv': array}``).
        Keys match the symbol names (or aliases) in ``cost_matrix``.
        Pass as keyword arguments to :class:`~empulse.metrics.Metric`::

            metric(y_true, y_score, **dataset.instance_costs)

        ``None`` when no instance-dependent costs are available.
    feature_names : list[str]
        Column names of ``data``.
    target_names : list[str]
        Human-readable label names.
    name : str
        Dataset name.
    DESCR : str
        Full description of the dataset.
    """

    data: IntoDataFrameT
    target: IntoSeriesT
    cost_matrix: CostMatrix
    instance_costs: dict[str, np.ndarray] | None
    feature_names: list[str]
    target_names: list[str]
    name: str
    DESCR: str
