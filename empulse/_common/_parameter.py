from enum import Enum


class Parameter(Enum):
    """Used to know if parameters have been set."""

    UNCHANGED = 'unchanged'

    # This is a bit of a hack to make sure that if someone tries to negate the parameter, it doesn't change it.
    def __neg__(self):
        return self
