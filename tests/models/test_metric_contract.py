"""Models reach a metric through its public ModelInterface, and the hooks behind it are declared on BaseMetric."""

import ast
import inspect
import re
from pathlib import Path

import empulse
from empulse.metrics import BaseMetric, ModelInterface

_PACKAGE = Path(empulse.__file__).parent
# A loss or metric object as the models, samplers and optimizers name it, followed by a private member.
_PRIVATE_MEMBER = re.compile(r'\b(?:loss_?|metric_loss|self\.loss|metric)\.(_[a-z][a-z_]*)\b')
# Returns a compiled object only ProfTree's native code can consume, so it is not part of ModelInterface.
_INTERNAL_HOOKS = {'_closed_form_expected_max_profit'}


def _private_members_used_on_metrics() -> dict[str, list[str]]:
    used: dict[str, list[str]] = {}
    for folder in ('models', 'samplers', 'optimizers', '_common'):
        for path in sorted((_PACKAGE / folder).rglob('*.py')):
            for name in _PRIVATE_MEMBER.findall(path.read_text(encoding='utf-8')):
                used.setdefault(name, []).append(str(path.relative_to(_PACKAGE)))
    return used


def _hooks_behind_model_interface() -> set[str]:
    """The private members of the metric that ModelInterface's methods and properties read."""
    tree = ast.parse(inspect.getsource(ModelInterface))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == '_metric'
        and node.attr.startswith('_')
    }


def test_models_use_no_private_metric_members():
    used = _private_members_used_on_metrics()
    assert set(used) >= _INTERNAL_HOOKS, 'the scan found nothing; has the naming of losses changed?'
    private = {name: files for name, files in sorted(used.items()) if name not in _INTERNAL_HOOKS}
    assert not private, f'Use ModelInterface (`loss.model`) instead of these: {private}'


def test_base_metric_declares_every_hook_model_interface_uses():
    hooks = _hooks_behind_model_interface()
    assert '_loss' in hooks, 'the scan found nothing; has ModelInterface stopped naming its metric `_metric`?'
    undeclared = sorted(name for name in hooks if not hasattr(BaseMetric, name))
    assert not undeclared, f'Declare these on BaseMetric: {undeclared}'


def test_base_metric_documents_every_hook():
    documented = set(re.findall(r'`(_[a-z][a-z_]*)`', BaseMetric.__doc__ or ''))
    missing = sorted((_hooks_behind_model_interface() | _INTERNAL_HOOKS) - documented)
    assert not missing, f'List these in the BaseMetric docstring: {missing}'
