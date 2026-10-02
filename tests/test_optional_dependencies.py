"""
What works without each optional extra.

Each check runs in a fresh interpreter with the extra's package made unimportable, so blocking it
cannot leak into other tests sharing an xdist worker.
"""

import subprocess
import sys
import textwrap

import pytest

BLOCK_IMBLEARN = textwrap.dedent("""
    import sys
    sys.modules['imblearn'] = None
""")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', BLOCK_IMBLEARN + textwrap.dedent(code)], capture_output=True, text=True, check=False
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
