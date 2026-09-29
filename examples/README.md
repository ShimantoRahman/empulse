# Replication materials for the Empulse JSS paper

- `replication_jss.ipynb` runs every code example of the paper, reproduces the case study (Section 8)
  with its printed output, and regenerates the two benchmark figures: the EMP integration comparison
  (Section 5.4) and the computational performance comparison (Section 9).
- `jss_benchmark/` holds the benchmark code the notebook drives, including the reference
  implementations Empulse is compared against:
  - `reference_cslogit.py`: the Python reimplementation of `CSLogit` (Höppner et al., 2022) by
    Vanderschueren et al. (2022), from
    <https://github.com/toonvds/CostSensitiveLearning> (MIT License);
  - `reference_costcla.py`: the cost-sensitive decision tree and random forest of CostCla, from
    <https://github.com/albahnsen/CostSensitiveClassification> (BSD 3-clause License), with the edits
    needed to run on current Python, NumPy and scikit-learn marked `# CHANGE:`.
- `results/` holds the raw benchmark results reported in the paper, one row per run.

## Running

```bash
pip install empulse[boosting] matplotlib jupyter
jupyter notebook replication_jss.ipynb
```

Run the notebook from this directory. Everything up to Section 9 takes about a minute. Section 9
redraws the figures from `results/` unless one of its `RUN_*` flags is switched on; the notebook states
how long each experiment takes. The experiments can also be run from the command line in this
directory, for example `python -m jss_benchmark.timing --impl legacy`.
