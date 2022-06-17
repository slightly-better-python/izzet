import unittest
from typing import AnyStr
from typing import Dict
from typing import List
from typing import Union

from izzet import izzet

JSONType = Union[str, int, float, bool, None, List['JSONType'], Dict[str, 'JSONType']]


class TestIzzetA(unittest.TestCase):

    @staticmethod
    def test_izzet_a_bytes():
        assert izzet(b'byte_string').a(bytes) is True

    @staticmethod
    def test_izzet_a_str():
        assert izzet('bar').a('str') is True

    @staticmethod
    def test_izzet_a_anystr_false():
        assert izzet(int).a(AnyStr) is False
