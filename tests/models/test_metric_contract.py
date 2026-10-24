"""What models need of a metric is declared on BaseMetric, so a new kind of metric knows what to implement."""

import re
from pathlib import Path

import empulse
from empulse.metrics import BaseMetric

_PACKAGE = Path(empulse.__file__).parent
# A loss or metric object as the models, samplers and optimizers name it, followed by a private member.
_PRIVATE_MEMBER = re.compile(r'\b(?:loss_?|metric_loss|self\.loss|metric)\.(_[a-z][a-z_]*)\b')


def _private_members_used_on_metrics() -> dict[str, list[str]]:
    used: dict[str, list[str]] = {}
    for folder in ('models', 'samplers', 'optimizers', '_common'):
        for path in sorted((_PACKAGE / folder).rglob('*.py')):
            for name in _PRIVATE_MEMBER.findall(path.read_text(encoding='utf-8')):
                used.setdefault(name, []).append(str(path.relative_to(_PACKAGE)))
    return used


def test_models_use_only_private_metric_members_that_base_metric_declares():
    used = _private_members_used_on_metrics()
    assert '_loss' in used, 'the scan found nothing; has the naming of losses changed?'
    undeclared = {name: files for name, files in used.items() if not hasattr(BaseMetric, name)}
    assert not undeclared, f'Declare these on BaseMetric, or stop using them: {undeclared}'


def test_base_metric_documents_every_private_member_models_use():
    documented = set(re.findall(r'`(_[a-z][a-z_]*)`', BaseMetric.__doc__ or ''))
    missing = sorted(set(_private_members_used_on_metrics()) - documented)
    assert not missing, f'List these in the BaseMetric docstring: {missing}'
