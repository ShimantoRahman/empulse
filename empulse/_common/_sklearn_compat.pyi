# The vendored module defines different names for different scikit-learn versions, so mypy cannot
# see them; every attribute is typed as Any.
from typing import Any

def __getattr__(name: str) -> Any: ...
