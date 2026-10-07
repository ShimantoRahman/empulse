"""
What works without each optional extra.

Each check runs in a fresh interpreter with the extra's package made unimportable, so blocking it
cannot leak into other tests sharing an xdist worker.
"""

import subprocess
import sys
import textwrap

import pytest


def _run(code: str, *, block: str = 'imblearn') -> subprocess.CompletedProcess[str]:
    blocker = f'import sys\nsys.modules[{block!r}] = None\n'
    return subprocess.run(
        [sys.executable, '-c', blocker + textwrap.dedent(code)], capture_output=True, text=True, check=False
    )


def test_bias_mitigation_models_work_without_imbalanced_learn():
    result = _run("""
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from empulse.models import BiasRelabelingClassifier, BiasResamplingClassifier

        X, y = make_classification(random_state=0)
        sensitive_feature = np.random.default_rng(0).integers(0, 2, y.shape)
        for cls in (BiasRelabelingClassifier, BiasResamplingClassifier):
            cls(LogisticRegression()).fit(X, y, sensitive_feature=sensitive_feature).predict(X)
    """)
    assert result.returncode == 0, result.stderr


def test_samplers_without_imbalanced_learn_name_the_extra():
    result = _run("""
        import empulse.samplers
    """)
    assert result.returncode != 0
    assert 'pip install empulse[sampling]' in result.stderr


@pytest.mark.parametrize('module', ['empulse.metrics', 'empulse.models', 'empulse.optimizers', 'empulse.datasets'])
def test_imports_without_imbalanced_learn(module):
    result = _run(f'import {module}')
    assert result.returncode == 0, result.stderr


def test_openml_datasets_with_pandas_without_pyarrow_name_the_extra(tmp_path):
    result = _run(
        f"""
        import pandas
        from empulse.datasets import fetch_home_equity

        fetch_home_equity(backend=pandas, data_home={str(tmp_path)!r})
        """,
        block='pyarrow',
    )
    assert result.returncode != 0
    assert 'pip install empulse[datasets]' in result.stderr


def test_openml_datasets_with_polars_work_without_pyarrow(tmp_path):
    pytest.importorskip('polars')
    result = _run(
        f"""
        import polars
        from empulse.datasets import fetch_home_equity

        polars.DataFrame({{
            'BAD': [1, 0], 'LOAN': [1100, 1700], 'REASON': ['HomeImp', None], 'DEBTINC': [None, 37.1]
        }}).write_parquet({str(tmp_path / 'home_equity.parquet')!r})
        dataset = fetch_home_equity(backend=polars, data_home={str(tmp_path)!r}, download_if_missing=False)
        assert dataset.data.shape == (2, 3), dataset.data.shape
        """,
        block='pyarrow',
    )
    assert result.returncode == 0, result.stderr
