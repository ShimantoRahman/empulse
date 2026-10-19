# Empulse JSS replication

Install the dependencies from PyPI and run the script:

```bash
python -m pip install -r requirements.txt
python replication_jss.py
```

`main()` calls three functions:

- `run_examples()` runs the code examples from the paper.
- `run_emp_benchmark()` runs all EMP benchmarking experiment from section 6.4 (figure 2).
- `run_training_times()` run the training times experiments from section 10 (figure 3).

To keep the runtime reasonable, the default settings in main() limit the number of stochastic variables, training samples, and repetitions. You can edit these settings in `main()` for larger experiments.

These settings replicate the full experiments in the paper (but may take a long time to run):

```python
max_emp_variables = 3
max_training_samples = 1_000_000
training_repetitions = 10
```

The helper modules in `jss_benchmark/` contain the experiment definitions, comparison models, and plotting functions.
