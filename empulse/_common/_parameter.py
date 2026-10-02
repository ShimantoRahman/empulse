from enum import Enum
from typing import Self


class Parameter(Enum):
    """Used to know if parameters have been set."""

    UNCHANGED = 'unchanged'

    # Negating an unset cost argument returns the sentinel instead of raising.
    def __neg__(self) -> Self:
        return self
