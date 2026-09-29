"""Benchmark code for the Empulse JSS paper: the EMP integration comparison and the scaling benchmark.

The modules are driven from ``replication_jss.ipynb`` one directory up, but each can also be run on
its own from the ``examples`` directory:

    python -m jss_benchmark.timing --impl current
    python -m jss_benchmark.timing --impl legacy
    python -m jss_benchmark.emp_integration

- ``data``: the synthetic datasets and instance-dependent costs.
- ``models``: the benchmarked models, for Empulse (``current``) and for the reference
  implementations they were ported from (``legacy``).
- ``reference_cslogit`` / ``reference_costcla``: those reference implementations, extracted with
  only the edits current Python, NumPy and scikit-learn require.
- ``timing`` / ``emp_integration``: the two experiments. Each writes raw rows to a
  CSV under ``results/``, and resumes where it stopped if interrupted.
- ``figures``: summary tables and the paper's figures, drawn only from those CSVs.
"""
