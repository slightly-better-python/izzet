import sys
from typing import Any

from izzet.exceptions import TypeCheckError
from izzet.memo import TypeCheckMemo
from izzet.type_checkers import _check_type

__title__ = 'Izzet'
__version__ = "1.0.1"
__author__ = 'Peter Wensel'
__license__ = 'MIT'
__credits__ = 'Slightly Better Python'


class Izzet:

    def __init__(self, value: Any):
        self.value = value
        frame = sys._getframe(1)  # noqa
        try:
            _globals = frame.f_gloabls
        except AttributeError:
            _globals = {'value': {}}

        try:
            _locals = frame.f_locals
        except AttributeError:
            _locals = {'value': {}}

        self.memo: TypeCheckMemo = TypeCheckMemo(_globals, _locals)

    def a(self, expected_type: Any) -> bool:
        """
        :param expected_type: a class or generic type instance
        :return: bool
        """
        try:
            _check_type(self.value, expected_type, self.memo)
            return True
        except TypeError:
            return False

    def not_a(self, unexpected_type: Any):
        return not self.a(unexpected_type)


def izzet(value: Any) -> Izzet:
    return Izzet(value)
