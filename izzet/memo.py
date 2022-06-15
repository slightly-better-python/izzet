from __future__ import annotations

from typing import Any
from typing import Dict


class TypeCheckMemo:
    __slots__ = 'globals', 'locals'

    def __init__(self, _globals: Dict[str, Any], _locals: Dict[str, Any]):
        self.globals = _globals
        self.locals = _locals
