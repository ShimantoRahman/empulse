from enum import Enum


class Parameter(Enum):
    """
    Default of a cost argument the caller did not pass, told apart from any value it could pass.

    It supports no arithmetic, so an unset cost that leaks into a calculation fails at once.
    """

    UNCHANGED = 'unchanged'
